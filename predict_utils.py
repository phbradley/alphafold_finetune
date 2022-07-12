######################################################################################88
import sys
import os
from os.path import exists
import pickle
from collections import OrderedDict
from sys import exit
import numpy as np
import pandas as pd
import tensorflow as tf
import train_utils
import random
from timeit import default_timer as timer
import haiku as hk
from alphafold.common import residue_constants
from alphafold.common import protein
from alphafold.data import pipeline
from alphafold.data import templates
from alphafold.model import data
from alphafold.model import config
from alphafold.model import model

# super-simple, stripped down pdb reader
# not good with messy pdbs
def load_pdb_coords(
        pdbfile,
        allow_chainbreaks=False,
        allow_skipped_lines=False,
        verbose=False,
):
    ''' returns: chains, all_resids, all_coords, all_name1s
    '''

    chains = []
    all_resids = {}
    all_coords = {}
    all_name1s = {}

    if verbose:
        print('reading:', pdbfile)
    skipped_lines = False
    with open(pdbfile,'r') as data:
        for line in data:
            if (line[:6] in ['ATOM  ','HETATM'] and line[17:20] != 'HOH' and
                line[16] in ' A1'):
                if ( line[17:20] in residue_constants.restype_3to1
                     or line[17:20] == 'MSE'): # 2022-03-31 change to include MSE
                    name1 = ('M' if line[17:20] == 'MSE' else
                             residue_constants.restype_3to1[line[17:20]])
                    resid = line[22:27]
                    chain = line[21]
                    if chain not in all_resids:
                        all_resids[chain] = []
                        all_coords[chain] = {}
                        all_name1s[chain] = {}
                        chains.append(chain)
                    if line.startswith('HETATM'):
                        print('WARNING: HETATM', pdbfile, line[:-1])
                    atom = line[12:16].split()[0]
                    if resid not in all_resids[chain]:
                        all_resids[chain].append(resid)
                        all_coords[chain][resid] = {}
                        all_name1s[chain][resid] = name1

                    all_coords[chain][resid][atom] = np.array(
                        [float(line[30:38]), float(line[38:46]), float(line[46:54])])
                else:
                    print('skip ATOM line:', line[:-1], pdbfile)
                    skipped_lines = True

    # check for chainbreaks
    maxdis = 1.75
    for chain in chains:
        for res1, res2 in zip(all_resids[chain][:-1], all_resids[chain][1:]):
            coords1 = all_coords[chain][res1]
            coords2 = all_coords[chain][res2]
            if 'C' in coords1 and 'N' in coords2:
                dis = np.sqrt(np.sum(np.square(coords1['C']-coords2['N'])))
                if dis>maxdis:
                    print('WARNING chainbreak:', chain, res1, res2, dis, pdbfile)
                    if not allow_chainbreaks:
                        print('STOP: chainbreaks', pdbfile)
                        print('DONE')
                        exit()

    if skipped_lines and not allow_skipped_lines:
        print('STOP: skipped lines:', pdbfile)
        print('DONE')
        exit()

    return chains, all_resids, all_coords, all_name1s


def fill_afold_coords(
        chain_order,
        all_resids,
        all_coords,
):
    ''' returns: all_positions, all_positions_mask

    these are 'atom37' coords (not 'atom14' coords)

    '''
    assert residue_constants.atom_type_num == 37 #HACK/SANITY
    crs = [(chain,resid) for chain in chain_order for resid in all_resids[chain]]
    num_res = len(crs)
    all_positions = np.zeros([num_res, residue_constants.atom_type_num, 3])
    all_positions_mask = np.zeros([num_res, residue_constants.atom_type_num],
                                  dtype=np.int64)
    for res_index, (chain,resid) in enumerate(crs):
        pos = np.zeros([residue_constants.atom_type_num, 3], dtype=np.float32)
        mask = np.zeros([residue_constants.atom_type_num], dtype=np.float32)
        for atom_name, xyz in all_coords[chain][resid].items():
            x,y,z = xyz
            if atom_name in residue_constants.atom_order.keys():
                pos[residue_constants.atom_order[atom_name]] = [x, y, z]
                mask[residue_constants.atom_order[atom_name]] = 1.0
            elif atom_name != 'NV': # PRO NV OK to skip
                # this is just debugging/verbose output:
                name = atom_name[:]
                while name[0] in '123':
                    name = name[1:]
                if name[0] != 'H':
                    print('unrecognized atom:', atom_name, chain, resid)
            # elif atom_name.upper() == 'SE' and res.get_resname() == 'MSE':
            #     # Put the coordinates of the selenium atom in the sulphur column.
            #     pos[residue_constants.atom_order['SD']] = [x, y, z]
            #     mask[residue_constants.atom_order['SD']] = 1.0

        all_positions[res_index] = pos
        all_positions_mask[res_index] = mask
    return all_positions, all_positions_mask



def run_alphafold_prediction(
        query_sequence: str,
        msa: list,
        deletion_matrix: list,
        chainbreak_sequence: str,
        template_features: dict,
        model_runners: dict,
        out_prefix: str,
        crop_size=None,
        dump_pdbs=True,
        dump_metrics=True,
):
    '''msa should be a list. If single seq is provided, it should be a list of str.

    returns a dictionary with keys= model_name, values= dictionary
    indexed by metric_tag
    '''
    # gather features for running with only template information
    feature_dict = {
        **pipeline.make_sequence_features(sequence=query_sequence,
                                          description="none",
                                          num_res=len(query_sequence)),
        **pipeline.make_msa_features(msas=[msa],
                                     deletion_matrices=[deletion_matrix]),
        **template_features
    }

    # add big enough number to residue index to indicate chain breaks

    # Ls: number of residues in each chain
    # Ls = [ len(split) for split in chainbreak_sequence.split('/') ]
    Ls = [ len(split) for split in chainbreak_sequence.split('/') ]
    idx_res = feature_dict['residue_index']
    L_prev = 0
    for L_i in Ls[:-1]:
        idx_res[L_prev+L_i:] += 200
        L_prev += L_i
    feature_dict['residue_index'] = idx_res

    all_metrics = predict_structure(
        out_prefix, feature_dict, model_runners, crop_size=crop_size,
        dump_pdbs=dump_pdbs, dump_metrics=dump_metrics,
    )

    #np.save('{}_plddt.npy'.format(out_prefix), plddts['model_1'])

    return all_metrics


def predict_structure(
        prefix,
        feature_dict,
        model_runners,
        random_seed=0,
        crop_size=None,
        dump_pdbs=True,
        dump_metrics=True,
):
    """Predicts structure using AlphaFold for the given sequence.

    returns a dictionary with keys= model_name, values= dictionary
    indexed by metric_tag
    """

    # Run the models.
    #plddts = []
    unrelaxed_pdb_lines = []
    relaxed_pdb_lines = []
    model_names = []

    metric_tags = 'plddt ptm predicted_aligned_error'.split()

    all_metrics = {} # eventual return value

    metrics = {} # stupid duplication

    for model_name, model_runner in model_runners.items():
        start = timer()
        print(f"running {model_name}")

        processed_feature_dict = model_runner.process_features(
            feature_dict, random_seed=random_seed)

        prediction_result = model_runner.predict(processed_feature_dict)

        unrelaxed_protein = protein.from_prediction(
            processed_feature_dict, prediction_result)
        unrelaxed_pdb_lines.append(protein.to_pdb(unrelaxed_protein))
        model_names.append(model_name)

        all_metrics[model_name] = {}
        for tag in metric_tags:
            result = prediction_result.get(tag, None)
            metrics.setdefault(tag, []).append(result)
            if result is not None:
                all_metrics[model_name][tag] = result

        print(f"{model_name} pLDDT: {np.mean(prediction_result['plddt'])} "
              f"Time: {timer() - start}")

    # rerank models based on predicted lddt
    plddts = metrics['plddt']
    lddt_rank = np.mean(plddts,-1).argsort()[::-1]
    #plddts_ranked = {}
    for n, r in enumerate(lddt_rank):
        print(f"model_{n+1} {np.mean(plddts[r])}")

        if dump_pdbs:
            unrelaxed_pdb_path = f'{prefix}_model_{n+1}_{model_names[r]}.pdb'
            with open(unrelaxed_pdb_path, 'w') as f: f.write(unrelaxed_pdb_lines[r])


        #plddts_ranked[f"model_{n+1}"] = plddts[r]

        if dump_metrics:
            metrics_prefix = f'{prefix}_model_{n+1}_{model_names[r]}'
            for tag in metric_tags:
                m = metrics[tag][r]
                if m is not None:
                    np.save(f'{metrics_prefix}_{tag}.npy', m)

    return all_metrics


def load_model_runners(
        model_names,
        crop_size,
        data_dir,
        num_recycle = 3,
        num_ensemble = 1,
        model_params_files = None,
        resample_msa_in_recycling = True,
        small_msas = True,
):
    if model_params_files is None:
        model_params_files = [None]*len(model_names)

    assert len(model_names) == len(model_params_files)

    model_runners = OrderedDict()
    for model_name, model_params_file in zip(model_names, model_params_files):
        print('config:', model_name)
        af_model_name = (model_name[:model_name.index('_ft')] if '_ft' in model_name
                         else model_name)
        model_config = config.model_config(af_model_name)

        # since this is not set automatically based on nres in this vers of alphafold
        model_config.data.eval.crop_size = crop_size
        model_config.data.eval.num_ensemble = num_ensemble
        model_config.data.common.num_recycle = num_recycle
        model_config.model.num_recycle = num_recycle
        if small_msas:
            print('load_model_runners:: small_msas==True setting small',
                  'max_extra_msa and max_msa_clusters')
            model_config.data.common.max_extra_msa = 1 #############
            model_config.data.eval.max_msa_clusters = 5 ###############
        if not resample_msa_in_recycling:
            model_config.data.common.resample_msa_in_recycling = False
            model_config.model.resample_msa_in_recycling = False


        if model_params_file != 'classic' and model_params_file is not None:
            print('loading', model_name, 'params from file:', model_params_file)
            with open(model_params_file, 'rb') as f:
                model_params = pickle.load(f)

            model_params, other_params = hk.data_structures.partition(
                lambda m, n, p: m[:9] == "alphafold", model_params)
            print('ignoring other_params:', other_params)

        else:
            assert '_ft' not in model_name
            model_params = data.get_model_haiku_params(
                model_name=model_name, data_dir=data_dir)

        model_runners[model_name] = model.RunModel(
            model_config, model_params)
    return model_runners


def create_single_template_features(
        target_sequence,
        template_pdbfile,
        target_to_template_alignment,
        template_name, # goes into template_domain_names, .encode()'ed
        allow_chainbreaks=True,
        allow_skipped_lines=True,
        expected_identities=None,
        expected_template_len=None,
):
    num_res = len(target_sequence)
    chains_tmp, all_resids_tmp, all_coords_tmp, all_name1s_tmp = load_pdb_coords(
        template_pdbfile, allow_chainbreaks=allow_chainbreaks,
        allow_skipped_lines=allow_skipped_lines,
    )

    crs_tmp = [(c,r) for c in chains_tmp for r in all_resids_tmp[c]]
    num_res_tmp = len(crs_tmp)
    template_full_sequence = ''.join(all_name1s_tmp[c][r] for c,r in crs_tmp)
    if expected_template_len:
        assert len(template_full_sequence) == expected_template_len

    all_positions_tmp, all_positions_mask_tmp = fill_afold_coords(
        chains_tmp, all_resids_tmp, all_coords_tmp)

    identities = sum(target_sequence[i] == template_full_sequence[j]
                     for i,j in target_to_template_alignment.items())
    if expected_identities:
        assert identities == expected_identities

    all_positions = np.zeros([num_res, residue_constants.atom_type_num, 3])
    all_positions_mask = np.zeros([num_res, residue_constants.atom_type_num],
                                  dtype=np.int64)

    template_alseq = ['-']*num_res
    for i,j in target_to_template_alignment.items(): # i=target, j=template
        template_alseq[i] = template_full_sequence[j]
        all_positions[i] = all_positions_tmp[j]
        all_positions_mask[i] = all_positions_mask_tmp[j]

    template_sequence = ''.join(template_alseq)
    assert len(template_sequence) == len(target_sequence)
    assert identities == sum(a==b for a,b in zip(template_sequence, target_sequence))

    template_aatype = residue_constants.sequence_to_onehot(
        template_sequence, residue_constants.HHBLITS_AA_TO_ID)

    template_features = {
        'template_all_atom_positions': all_positions,
        'template_all_atom_masks': all_positions_mask,
        'template_sequence': template_sequence.encode(),
        'template_aatype': template_aatype,
        'template_domain_names': template_name.encode(),
        'template_sum_probs': [identities],
        }
    return template_features


def compile_template_features(template_features_list):
    all_template_features = {}
    for name, dtype in templates.TEMPLATE_FEATURES.items():
        all_template_features[name] = np.stack(
            [f[name] for f in template_features_list], axis=0).astype(dtype)
    return all_template_features


def create_batch_for_training(
        target_chainseq, # has '/' between chains
        target_trim_positions, # 0-indexed wrt full target sequence
        templates_alignfile, # tsv file with cols given below, alignments to templates
        native_pdbfile,
        native_align, # dict (trg_pos, nat_pos), 0-indexed, wrt full target sequence
        crop_size, # sanity
        model_runner, # for feature processing
        native_identities=None, # for sanity checking
        native_len=None, # for sanity checking
        debug=False,
        verbose=False,
        random_seed=None, # if None, randomize
):
    ''' alignfile cols are:

    template_pdbfile
    target_to_template_alignstring
    identities
    target_len
    template_len
    '''
    assert len(target_trim_positions) <= crop_size
    assert None not in target_trim_positions
    if verbose:
        print('create_batch_for_training:', target_chainseq, target_trim_positions,
              templates_alignfile,
              native_pdbfile, native_align, crop_size, native_identities,
              native_len)
    target_trim_positions = sorted(set(target_trim_positions)) # sanity
    full_pos_to_trim_pos = {pos:i for i,pos in enumerate(target_trim_positions)}

    target_full_sequence = target_chainseq.replace('/','')
    target_sequence = ''.join(target_full_sequence[x] for x in target_trim_positions)
    target_cs = target_chainseq.split('/')

    residue_index = np.arange(len(target_full_sequence))
    for ch in range(1,len(target_cs)):
        chain_begin = sum(len(x) for x in target_cs[:ch])
        residue_index[chain_begin:] += 200
    #print('chain_lens:', [len(x) for x in target_cs])
    #print('full residue_index:', residue_index)
    trim_residue_index = residue_index[target_trim_positions]
    #print('trim_residue_index:', trim_residue_index)

    # templates stuff
    template_features_list = []
    templates_df = pd.read_table(templates_alignfile)
    for l in templates_df.itertuples():
        align_full = {int(x.split(':')[0]):int(x.split(':')[1])
                      for x in l.target_to_template_alignstring.split(';')}
        if debug:
            create_single_template_features(
                target_full_sequence, l.template_pdbfile, align_full, f'temp{l.Index}',
                expected_identities = l.identities,
                expected_template_len = l.template_len)
        align = {full_pos_to_trim_pos[x]:y for x,y in align_full.items()
                 if x in target_trim_positions}
        features = create_single_template_features(
            target_sequence, l.template_pdbfile, align, f'temp{l.Index}',
            expected_template_len = l.template_len)
        template_features_list.append(features)
    all_template_features = compile_template_features(template_features_list)

    msa = [target_sequence]
    deletions = [[0]*len(target_sequence)]

    feature_dict = {
        **pipeline.make_sequence_features(
            sequence=target_sequence, description="none", num_res=len(target_sequence)),
        **pipeline.make_msa_features(msas=[msa], deletion_matrices=[deletions]),
        **all_template_features,
    }

    if verbose:
        print('features_after_creation:', ' '.join(feature_dict.keys()))
    old_ri = feature_dict['residue_index']
    feature_dict['residue_index'] = trim_residue_index.astype(old_ri.dtype)

    if random_seed is None:
        random_seed = np.random.randint(0,999999)
    with tf.device('cpu:0'):
        processed_feature_dict = model_runner.process_features(
            feature_dict, random_seed=random_seed)
    if verbose:
        print('features_after_initial_processing:',
              ' '.join(processed_feature_dict.keys()))

    # get native coords
    if debug:
        native_features = create_single_template_features(
            target_full_sequence, native_pdbfile, native_align, 'dummy',
            expected_identities=native_identities,
            expected_template_len=native_len)
    native_align_trimmed = {full_pos_to_trim_pos[x]:y for x,y in native_align.items()
                            if x in target_trim_positions}
    native_features = create_single_template_features(
        target_sequence, native_pdbfile, native_align_trimmed, 'dummy',
        expected_template_len=native_len)

    L = processed_feature_dict['aatype'].shape[1]
    assert L == crop_size
    L0 = len(target_trim_positions)
    atom_positions = np.concatenate([
        native_features['template_all_atom_positions'],
        np.zeros([L-L0, 37, 3])], 0)
    atom_mask = np.concatenate([
        native_features['template_all_atom_masks'],
        np.zeros([L-L0, 37])], 0)
    aatype = processed_feature_dict['aatype'][0]
    #print('old aatype:', aatype)
    aatype = np.concatenate([
        processed_feature_dict['aatype'][0][:L0],
        20*np.ones([L-L0])],0).astype(np.int32)
    #print('new aatype:', aatype)

    pseudo_beta, pseudo_beta_mask = train_utils.pseudo_beta_fn_np(
        aatype, atom_positions, atom_mask)
    protein_dict = {'aatype': aatype,
            'all_atom_positions': atom_positions,
            'all_atom_mask': atom_mask}
    protein_dict = train_utils.make_atom14_positions(protein_dict)
    del protein_dict['aatype']
    for key_, value_ in protein_dict.items():
        protein_dict[key_] = np.array(value_)[None,]
    processed_feature_dict['pseudo_beta'] = np.array(pseudo_beta)[None,]
    processed_feature_dict['pseudo_beta_mask'] = np.array(pseudo_beta_mask)[None,]
    processed_feature_dict['all_atom_mask'] = np.array(atom_mask)[None,]
    processed_feature_dict['resolution'] = np.array(1.0)[None,]
    processed_feature_dict.update(protein_dict)
    n, ca, c = [residue_constants.atom_order[a] for a in ('N', 'CA', 'C')]
    rot, trans = train_utils.make_transform_from_reference_np(
        n_xyz =processed_feature_dict['all_atom_positions'][0, :, n , :],
        ca_xyz=processed_feature_dict['all_atom_positions'][0, :, ca, :],
        c_xyz =processed_feature_dict['all_atom_positions'][0, :, c , :])
    processed_feature_dict['backbone_translation'] = trans[None,]
    processed_feature_dict['backbone_rotation'] = rot[None,]
    #processed_feature_dict['backbone_affine_mask'] = np.concatenate(
    #    [np.ones([1,L0]), np.zeros([1,L-L0])], 1)
    # this code borrowed from modules.py:2013
    # Backbone affine mask: whether the residue has C, CA, N
    processed_feature_dict['backbone_affine_mask'] = (
        processed_feature_dict['all_atom_mask'][0, :, n ] *
        processed_feature_dict['all_atom_mask'][0, :, ca] *
        processed_feature_dict['all_atom_mask'][0, :, c ])[None,]

    if verbose:
        print('features_at_end:', ' '.join(processed_feature_dict.keys()))
    return processed_feature_dict



