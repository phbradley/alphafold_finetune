######################################################################################88

FREDHUTCH_HACKS = True # silly stuff Phil added for running on Hutch servers
if FREDHUTCH_HACKS:
    import os
    from shutil import which
    os.environ['XLA_FLAGS']='--xla_gpu_force_compilation_parallelism=1'
    os.environ["TF_FORCE_UNIFIED_MEMORY"] = '1'
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = '2.0'
    assert which('ptxas') is not None


import argparse

parser = argparse.ArgumentParser(
    description="Run simple template-based alphafold inference",
    epilog = f'''
Examples:

# this command will build models and compute confidence scores for
# all 10mer peptides in HCV_POLG77 bound to HLA-A*02:01, using the default
# alphafold model_2_ptm parameters. You would need to change the --data_dir
# argument to point to the location of the folder containing the alphafold
# params/ subfolder.

python3 run_prediction.py --targets examples/pmhc_hcv_polg_10mers/targets.tsv --data_dir /home/pbradley/csdat/alphafold/data/ --outfile_prefix polg_test1 --model_names model_2_ptm --ignore_identities


    ''',
    formatter_class=argparse.RawDescriptionHelpFormatter,
)

parser.add_argument('--outfile_prefix',
                    help='Prefix that will be prepended to the output '
                    'filenames')
parser.add_argument('--final_outfile_prefix',
                    help='Prefix that will be prepended to the final output '
                    'tsv filename')
parser.add_argument('--targets', required=True, help='File listing the targets to '
                    'be modeled. See description of file format in the github '
                    'README and also examples in the examples/*/*tsv')
parser.add_argument('--data_dir', help='Location of AlphaFold params/ folder')

parser.add_argument('--model_names', type=str, nargs='*', default=['model_2_ptm'])
parser.add_argument('--model_params_files', type=str, nargs='*')

parser.add_argument('--verbose', action='store_true')
parser.add_argument('--ignore_identities', action='store_true',
                    help='Ignore the sequence identities column in the templates '
                    'alignment files. Useful when modeling many different peptides '
                    'using the same alignment file.')
parser.add_argument('--no_pdbs', action='store_true', help='Dont write out pdbs')
parser.add_argument('--terse', action='store_true', help='Dont write out pdbs or '
                    'matrices with alphafold confidence values')
parser.add_argument('--no_resample_msa', action='store_true', help='Dont randomly '
                    'resample from the MSA during recycling. Perhaps useful for '
                    'testing...')

args = parser.parse_args()

import os
import sys
from os.path import exists
import itertools
import numpy as np
import pandas as pd
import predict_utils

targets = pd.read_table(args.targets)

lens = [len(x.target_chainseq.replace('/',''))
        for x in targets.itertuples()]
crop_size = max(lens)

if args.verbose:
    import jax
    from os import popen # just to get hostname for logging, not necessary
    # print some logging info
    platform = jax.local_devices()[0].platform
    hostname = popen('hostname').readlines()[0].strip()

    print('cmd:', ' '.join(sys.argv))
    print('local_device:', platform, 'hostname:', hostname, 'num_targets:',
          targets.shape[0], 'max_len=', crop_size)

sys.stdout.flush()

model_runners = predict_utils.load_model_runners(
    args.model_names,
    crop_size,
    args.data_dir,
    model_params_files=args.model_params_files,
    resample_msa_in_recycling = not args.no_resample_msa,
)

final_dfl = []
for counter, targetl in targets.iterrows():
    print('START:', counter, 'of', targets.shape[0])

    alignfile = targetl.templates_alignfile
    assert exists(alignfile)

    query_chainseq = targetl.target_chainseq
    if 'outfile_prefix' in targetl:
        outfile_prefix = targetl.outfile_prefix
    else:
        assert args.outfile_prefix is not None
        if 'targetid' in targetl:
            outfile_prefix = args.outfile_prefix+'_'+targetl.targetid
        else:
            outfile_prefix = f'{args.outfile_prefix}_T{counter}'

    query_sequence = query_chainseq.replace('/','')
    num_res = len(query_sequence)

    data = pd.read_table(alignfile)
    cols = ('template_pdbfile target_to_template_alignstring identities '
            'target_len template_len'.split())
    template_features_list = []
    for tnum, row in data.iterrows():
        #(template_pdbfile, target_to_template_alignstring,
        # identities, target_len, template_len) = line[cols]

        assert row.target_len == len(query_sequence)
        target_to_template_alignment = {
            int(x.split(':')[0]) : int(x.split(':')[1]) # 0-indexed
            for x in row.target_to_template_alignstring.split(';')
        }

        template_name = f'T{tnum:03d}' # dont think this matters
        template_features = predict_utils.create_single_template_features(
            query_sequence, row.template_pdbfile, target_to_template_alignment,
            template_name, allow_chainbreaks=True, allow_skipped_lines=True,
            expected_identities = None if args.ignore_identities else row.identities,
            expected_template_len = row.template_len,
        )
        template_features_list.append(template_features)

    all_template_features = predict_utils.compile_template_features(
        template_features_list)

    msa=[query_sequence]
    deletion_matrix=[[0]*len(query_sequence)]

    all_metrics = predict_utils.run_alphafold_prediction(
        query_sequence=query_sequence,
        msa=msa,
        deletion_matrix=deletion_matrix,
        chainbreak_sequence=query_chainseq,
        template_features=all_template_features,
        model_runners=model_runners,
        out_prefix=outfile_prefix,
        crop_size=crop_size,
        dump_pdbs = not (args.no_pdbs or args.terse),
        dump_metrics = not args.terse,
    )


    outl = targetl.copy()
    for model_name, metrics in all_metrics.items():
        plddts = metrics['plddt']
        paes = metrics.get('predicted_aligned_error', None)

        cs = query_chainseq.split('/')
        chain_stops = list(itertools.accumulate(len(x) for x in cs))
        chain_starts = [0]+chain_stops[:-1]
        nres = chain_stops[-1]
        assert nres == num_res
        outl[model_name+'_plddt'] = np.mean(plddts[:nres])
        if paes is not None:
            outl[model_name+'_pae'] = np.mean(paes[:nres,:nres])
        for chain1,(start1,stop1) in enumerate(zip(chain_starts, chain_stops)):
            outl[f'{model_name}_plddt_{chain1}'] = np.mean(plddts[start1:stop1])

            if paes is not None:
                for chain2 in range(len(cs)):
                    start2, stop2 = chain_starts[chain2], chain_stops[chain2]
                    pae = np.mean(paes[start1:stop1,start2:stop2])
                    outl[f'{model_name}_pae_{chain1}_{chain2}'] = pae
    final_dfl.append(outl)

if args.final_outfile_prefix:
    outfile_prefix = args.final_outfile_prefix
elif args.outfile_prefix:
    outfile_prefix = args.outfile_prefix
elif 'outfile_prefix' in targets.columns:
    outfile_prefix = targets.outfile_prefix.iloc[0]
else:
    outfile_prefix = None

if outfile_prefix:
    outfile = f'{outfile_prefix}_final.tsv'
    pd.DataFrame(final_dfl).to_csv(outfile, sep='\t', index=False)
    print('made:', outfile)

