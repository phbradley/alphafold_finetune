######################################################################################88
FREDHUTCH_HACKS = False # silly stuff Phil added for running on Hutch servers

if FREDHUTCH_HACKS:
    import os
    # I think this has to happen before importing tensorflow?
    os.environ['XLA_FLAGS']='--xla_gpu_force_compilation_parallelism=1'

import os
from os import popen
import sys
import pandas as pd
import numpy as np
import pickle
import optax
import jax
import jax.numpy as jnp
import haiku as hk
from alphafold.common import protein
from alphafold.common import confidence
from alphafold.common.protein import Protein
from alphafold.data import pipeline
from alphafold.data import templates
from alphafold.model import data
from alphafold.model import config
from alphafold.model import model
from alphafold.model.all_atom import atom37_to_torsion_angles, atom37_to_frames
import torch # dataloader/dataset stuff
import tensorflow.compat.v1 as tf1 # cmdline args
import warnings
warnings.filterwarnings("ignore")

## fine-tuning specific helper codes
import train_utils
import predict_utils
print('done importing') ; sys.stdout.flush()

flags = tf1.app.flags

flags.DEFINE_string('outprefix', 'testrun', help='string prefix for all outfiles')
flags.DEFINE_string('model_name', 'model_2_ptm', help='like model_1 or model_2_ptm')
flags.DEFINE_string('train_dataset', None, help='tsv file with training dataset. See '
                    'README for format info')
flags.DEFINE_string('valid_dataset', None, help='tsv file with validation dataset')
flags.DEFINE_integer('crop_size', 190, help='Max size of training example; set this '
                     'as low as possible for memory and speed')
flags.DEFINE_integer('num_epochs', 10, help='number of epochs')
flags.DEFINE_integer('batch_size', 1, help='total batch size')
flags.DEFINE_integer('apply_every', 1, help='how often to apply gradient updates')
flags.DEFINE_bool('notrain', False, help='if True, dont do any training')
flags.DEFINE_bool('debug', False, help='debug')
flags.DEFINE_bool('verbose', False, help='verbose')
flags.DEFINE_bool('test_load', False,
                  help='if True, loop through datasets to test loading')
flags.DEFINE_bool('dump_pdbs', False, help='if True, write out PDB files of '
                  'modeled structures during training')
flags.DEFINE_integer('print_steps', 25,
                     help='printed averaged results every print_steps')
flags.DEFINE_integer('save_steps', 100, help='save model + optimizer every save_steps')
flags.DEFINE_integer('valid_steps', 100000,
                     help='calc loss on whole valid set every valid_steps')
flags.DEFINE_integer('num_cpus', 0, help='number of (extra?) cpus for loading')
flags.DEFINE_float('lr_coef', 0.025, help='learning rate coefficient')
flags.DEFINE_float('fake_native_weight', 0.25, help='weight to apply to the alphafold '
                   'loss when using a predicted structure as the native')
flags.DEFINE_float('struc_viol_weight', 1.0, help='structural violation weight')
flags.DEFINE_integer('msa_clusters', 5, help='number of msa cluster sequences')
flags.DEFINE_integer('extra_msa', 1, help='number of extra msa sequences')
flags.DEFINE_integer('num_evo_blocks', 48, help='number of evoformer blocks')
flags.DEFINE_float('grad_norm_clip', 10.0, help='value to clip gradient norms, '
                   'per update')
flags.DEFINE_string('data_dir', "/home/pbradley/csdat/alphafold/data/",
                    help='location of alphafold params; passed to '
                    'data.get_model_haiku_params; should contain params/ subfolder')

flags.DEFINE_bool('only_fit_binder', False, help='if True, dont fit alphafold params')
flags.DEFINE_bool('freeze_binder', False, help='if True, dont fit binder params')
flags.DEFINE_bool('freeze_everything', False, help='if True, dont fit anything')
flags.DEFINE_bool('no_ramp', False, help='if True, dont ramp')
flags.DEFINE_bool('no_valid', False, help='if True, dont compute valid stats')
flags.DEFINE_bool('no_random', False, help='if True, dont randomize')
flags.DEFINE_bool('random_recycling', False, help='if True, set num_iter_recycling '
                  'randomly during training')
flags.DEFINE_bool('plddt_binder', False, help='if True, fit model based on pLDDT')
flags.DEFINE_float('pae_binder_slope', -7.9019634017508675, # from logistic regression
                   help='initial slope for PAE binder model')
flags.DEFINE_float('plddt_binder_slope', 16.690735, # from logistic regression
                   help='initial slope for pLDDT binder model')
flags.DEFINE_multi_float('binder_intercepts', None, required=True,
                         help='Initial values for the switchpoint parameter in '
                         'the logistic regression binder layer. This is the point '
                         'where the prediction switches from binder to non-binder. '
                         'Should be in values of 0.01*pLDDT or 0.1*PAE, where '
                         'pLDDT ranges 0-100 and PAE has values in the 2-20 ish range.'
                         'See example command lines on the github README.')
FLAGS = flags.FLAGS
print('binder_intercepts:',FLAGS.binder_intercepts)

assert FLAGS.msa_clusters >= 5 # since reduce_msa_clusters_by_max_templates is True for model_1/model_1_ptm

batch_size = FLAGS.batch_size
if batch_size>1:
    print('WARNING\n'*12,'phil has not tested batch_size>1')

assert FLAGS.apply_every >= 1

jax_key = jax.random.PRNGKey(0)
model_name = FLAGS.model_name

platform = jax.local_devices()[0].platform
hostname = popen('hostname').readlines()[0].strip()


print('cmd:', ' '.join(sys.argv))
print('local_device:', platform, hostname)
print('model_name:', model_name)
print('outprefix:', FLAGS.outprefix)
sys.stdout.flush()

model_config = config.model_config(model_name)
model_config.data.common.resample_msa_in_recycling = True
model_config.model.resample_msa_in_recycling = True
model_config.data.common.max_extra_msa = FLAGS.extra_msa
model_config.data.eval.max_msa_clusters = FLAGS.msa_clusters
model_config.data.eval.crop_size = FLAGS.crop_size
model_config.model.heads.structure_module.structural_violation_loss_weight = FLAGS.struc_viol_weight
model_config.model.embeddings_and_evoformer.evoformer_num_block = FLAGS.num_evo_blocks

### Binder stuff #######################################################################

class PlddtBinderClassifier(hk.Module):
    ''' Uses average pLDDT value over the peptide

    assumes we have defined in the input_features

    * peptide_mask
    * binder_class_1hot
    * predicted_lddt dict

    '''

    def __init__(self, *args, **kwargs):
        super().__init__(name="PlddtBinderClassifier")

    def __call__(self, input_features):
        ''' Not sure this will work for batches of size > 1 ??
        '''
        mask = input_features['peptide_mask']
        binder_class_1hot = input_features['binder_class_1hot']

        plddts = train_utils.compute_plddt_jax(
            input_features['predicted_lddt']['logits'])

        # multiply by 0.01 so goes from 0 to 1
        plddt = 0.01 * jnp.sum(plddts * mask, keepdims=True)/jnp.sum(mask)

        if FLAGS.only_fit_binder:
            plddt = jax.lax.stop_gradient(plddt)

        x_intercept_init_values = jnp.array(FLAGS.binder_intercepts, dtype=plddt.dtype)
        num_binder_classes = len(FLAGS.binder_intercepts)
        slope_init_value = FLAGS.plddt_binder_slope
        assert binder_class_1hot.shape[-1] == num_binder_classes

        x_intercept_init = lambda s,d:jnp.stack([x_intercept_init_values]*s[0])
        x_intercept = hk.get_parameter(
            "x_intercept", shape=[plddt.shape[-1], num_binder_classes],
            dtype=plddt.dtype, init = x_intercept_init)
        slope_init = lambda s,d:jnp.full(s, slope_init_value, dtype=d)
        slope = hk.get_parameter(
            "slope", shape=[plddt.shape[-1]], dtype=plddt.dtype, init=slope_init)
        if FLAGS.freeze_binder:
            x_intercept = jax.lax.stop_gradient(x_intercept)
            slope = jax.lax.stop_gradient(slope)

        #print(plddt.shape, x_intercept.shape, binder_class_1hot.shape)
        assert x_intercept.shape == binder_class_1hot.shape
        x_intercept = jnp.sum(x_intercept * binder_class_1hot, axis=-1)

        binder_logits = (plddt - x_intercept)*slope
        nonbinder_logits = jnp.zeros(binder_logits.shape, binder_logits.dtype)
        logits = jnp.concatenate([nonbinder_logits, binder_logits], axis=-1)


        return logits, plddt

class PaeBinderClassifier(hk.Module):
    ''' Uses average PAE value between peptide and MHC

    assumes we have, defined in the input_features

    * partner1_mask
    * partner2_mask
    * binder_class_1hot
    * predicted_aligned_error dict

    eg partner1_mask might be the MHC and partner2_mask might be the peptide

    symmetrizes: we sum [partner1_mask][partner2_mask] and also
    [partner2_mask][partner1_mask]

    '''

    def __init__(self, *args, **kwargs):
        super().__init__(name="PaeBinderClassifier")

    def __call__(self, input_features):
        ''' Not sure this will work for batches of size > 1 ??
        '''
        mask1 = input_features['partner1_mask']
        mask2 = input_features['partner2_mask']
        binder_class_1hot = input_features['binder_class_1hot']

        paes = train_utils.compute_predicted_aligned_error_jax(
            input_features['predicted_aligned_error']['logits'],
            input_features['predicted_aligned_error']['breaks'].ravel(),
        )['predicted_aligned_error']

        paes12 = jnp.sum((paes * mask1[...,None])*mask2[...,None,:], keepdims=True)
        paes21 = jnp.sum((paes * mask2[...,None])*mask1[...,None,:], keepdims=True)
        # multiply by 0.1 to reduce range
        pae = 0.1 * (paes12 + paes21)[0]/(2*jnp.sum(mask1)*jnp.sum(mask2))
        print('pae:', pae)

        if FLAGS.only_fit_binder:
            pae = jax.lax.stop_gradient(pae)

        #x_intercept_init_value = 0.80367635 # Class I, from logistic regression
        x_intercept_init_values = jnp.array(FLAGS.binder_intercepts, dtype=pae.dtype)
        num_binder_classes = len(FLAGS.binder_intercepts)
        slope_init_value = FLAGS.pae_binder_slope
        ###############

        x_intercept_init = lambda s,d:jnp.stack([x_intercept_init_values]*s[0])
        x_intercept = hk.get_parameter(
            "x_intercept", shape=[pae.shape[-1], num_binder_classes],
            dtype=pae.dtype, init = x_intercept_init)

        slope_init = lambda s,d:jnp.full(s, slope_init_value, dtype=d)
        slope = hk.get_parameter(
            "slope", shape=[pae.shape[-1]], dtype=pae.dtype, init=slope_init)
        if FLAGS.freeze_binder:
            x_intercept = jax.lax.stop_gradient(x_intercept)
            slope = jax.lax.stop_gradient(slope)

        assert x_intercept.shape == binder_class_1hot.shape
        x_intercept = jnp.sum(x_intercept * binder_class_1hot, axis=-1)

        binder_logits = (pae - x_intercept)*slope
        nonbinder_logits = jnp.zeros(binder_logits.shape, binder_logits.dtype)
        logits = jnp.concatenate([nonbinder_logits, binder_logits], axis=-1)

        return logits, pae

if not FLAGS.plddt_binder:
    # PAE values coming into the binder model are multiplied by 0.1, to reduce
    # the dynamic range. So the binder_intercepts values should be something in the
    # range of 0.4 - 0.8 (ish)
    assert all(x<2.0 for x in FLAGS.binder_intercepts), \
        'binder model PAE values are multiplied by 0.1'
    def binder_classification_fn(input_features):
        model = PaeBinderClassifier()(input_features)
        return model
else:
    # pLDDT values coming into the binder model are multiplied by 0.01, to reduce
    # the dynamic range. So the binder_intercepts values should be something in the
    # range of 0.75 (ish), and definitely less than 1.0
    assert all(x<1.0 for x in FLAGS.binder_intercepts), \
        'binder model pLDDT values are multiplied by 0.01 (ie, run from 0-1)'
    def binder_classification_fn(input_features):
        model = PlddtBinderClassifier()(input_features)
        return model

binder_classifier = hk.transform(binder_classification_fn, apply_rng=True)

# create the initial model params, this part is a little hacky
if not FLAGS.plddt_binder: ## PAE model
    rng = jax.random.PRNGKey(42)

    classifier_dict = {}
    crop_size = 200 # doesnt matter
    num_bins = 64 # doesnt matter
    num_binder_classes = len(FLAGS.binder_intercepts)

    classifier_dict['partner1_mask'] = jnp.ones([crop_size], dtype=jnp.float32)
    classifier_dict['partner2_mask'] = jnp.ones([crop_size], dtype=jnp.float32)
    classifier_dict['predicted_aligned_error'] = {
        'logits': jnp.zeros([1, crop_size, crop_size, num_bins], dtype=jnp.float32),
        'breaks': jnp.zeros([1, num_bins-1], dtype=jnp.float32),
    }
    classifier_dict['binder_class_1hot'] = jnp.ones([1, num_binder_classes],
                                                    dtype=jnp.float32)

    binder_model_params = binder_classifier.init(
        rng,
        input_features=classifier_dict,
    )

else: # pLDDT model ################
    rng = jax.random.PRNGKey(42)

    classifier_dict = {}
    crop_size = 200 # doesnt matter
    num_bins = 50
    num_binder_classes = len(FLAGS.binder_intercepts)

    classifier_dict['peptide_mask'] = jnp.ones([1, crop_size], dtype=jnp.float32)
    classifier_dict['binder_class_1hot'] = jnp.ones([1, num_binder_classes],
                                                    dtype=jnp.float32)
    classifier_dict['predicted_lddt'] = {
        'logits': jnp.zeros([crop_size, num_bins], dtype=jnp.float32)
    }

    binder_model_params = binder_classifier.init(
        rng,
        input_features=classifier_dict,
    )


af2_model_params = data.get_model_haiku_params(
    model_name=model_name, data_dir=FLAGS.data_dir)

print('initial binder params:', binder_model_params)

model_params = hk.data_structures.merge(af2_model_params, binder_model_params)


model_runner = model.RunModel(model_config, af2_model_params)


############################ end of binder stuff ############################

def softmax_cross_entropy(logits, labels):
    """Computes softmax cross entropy given logits and one-hot class labels."""
    loss = -jnp.sum(labels * jax.nn.log_softmax(logits), axis=-1)
    return jnp.asarray(loss), jax.nn.softmax(logits)

def get_loss_fn(model_params, key, processed_feature_dict, structure_flag):
    binder_model_params, af2_model_params = hk.data_structures.partition(
        lambda m, n, p: m[:9] != "alphafold", model_params)
    classifier_dict = {}
    binder_labels = jnp.array(processed_feature_dict['labels'], dtype=jnp.float32)
    classifier_dict['binder_class_1hot'] = processed_feature_dict['binder_class_1hot']
    classifier_dict['peptide_mask'] = processed_feature_dict['peptide_mask']
    classifier_dict['partner1_mask'] = processed_feature_dict['partner1_mask']
    classifier_dict['partner2_mask'] = processed_feature_dict['partner2_mask']
    del processed_feature_dict['binder_class_1hot']
    del processed_feature_dict['peptide_mask']
    del processed_feature_dict['partner1_mask']
    del processed_feature_dict['partner2_mask']
    del processed_feature_dict['labels']
    predicted_dict, loss = model_runner.apply(
        af2_model_params, key, processed_feature_dict)
    classifier_dict.update(predicted_dict)
    binder_logits, binder_features = binder_classifier.apply(
        binder_model_params, key, classifier_dict)
    binder_loss, binder_probs = softmax_cross_entropy(binder_logits, binder_labels)
    binder_loss_mean = binder_loss.mean()
    if FLAGS.only_fit_binder:
        loss = jax.lax.stop_gradient(loss)
    fake_native_weight = jnp.array(FLAGS.fake_native_weight, jnp.float32)
    sf = jnp.array(structure_flag, jnp.float32) # 1.0 or 0.0
    not_sf = jnp.ones(sf.shape, dtype=sf.dtype) - sf
    loss = (sf + not_sf*fake_native_weight)*loss[0] + 1.0*binder_loss_mean
    return loss, (predicted_dict, binder_loss_mean, binder_probs, binder_features)

def train_step(model_params, key, batch, structure_flag):
    (loss, (predicted_dict, binder_loss, prob, features)), grads = jax.value_and_grad(
        get_loss_fn, has_aux=True)(model_params, key, batch, structure_flag)
    # moved the grad-norming outside, search for apply_every down below
    #grads = norm_grads_per_example(grads, l2_norm_clip=FLAGS.grad_norm_clip)
    grads = jax.lax.pmean(grads, axis_name='model_ax')
    loss = jax.lax.pmean(loss, axis_name='model_ax')
    return loss, grads, predicted_dict, binder_loss, prob, features

def norm_grads_per_example(grads, l2_norm_clip=0.1):
    nonempty_grads, tree_def = jax.tree_util.tree_flatten(grads)
    total_grad_norm = jnp.linalg.norm([jnp.linalg.norm(neg.ravel()) for neg in nonempty_grads])
    divisor = jnp.maximum(total_grad_norm / l2_norm_clip, 1.)
    normalized_nonempty_grads = [g / divisor for g in nonempty_grads]
    grads = jax.tree_util.tree_unflatten(tree_def, normalized_nonempty_grads)
    return grads


def create_batch_from_dataset_row(row):
    ''' row is a pandas Series that has fields:

    required_fields:
      target_chainseq  ('/'-separated chain amino acid sequences)
      templates_alignfile
      native_pdbfile
      native_alignstring (';'-separated intpairs eg '0:0;1:1;3:2;4:3')
      binder  (0 or 1)
      peptide_positions (';'-separated intlist eg '201;202;203;204;205')
      native_exists (True or False or 0 or 1)

    optional fields:
      binder_class (needed if len(FLAGS.binder_intercepts)>1 )
      target_trim_positions (used to subset the target for memory reasons)
      native_identities (for debugging PDB IO)
      native_len (for debugging PDB IO)

    '''

    debug = FLAGS.debug
    verbose = FLAGS.verbose
    if True: #debug:
        print('create_batch_from_dataset_row:',
              getattr(row,'targetid','unk'),
              getattr(row,'pdbid','unk'))

    nres = len(row.target_chainseq.replace('/',''))
    if hasattr(row, 'target_trim_positions'):
        target_trim_positions = [int(x) for x in row.target_trim_positions.split(';')]
        if target_trim_positions != list(range(nres)):
            print('WARNING: target_trimming:',
                  [x for x in range(nres) if x not in target_trim_positions])
    else:
        target_trim_positions = list(range(nres))
    native_align = {int(x.split(':')[0]):int(x.split(':')[1])
                    for x in row.native_alignstring.split(';')}

    native_identities = getattr(row, 'native_identities', None)
    native_identities = None if pd.isna(native_identities) else native_identities
    native_len = getattr(row, 'native_len', None)
    native_len = None if pd.isna(native_len) else native_len

    random_seed = 0 if FLAGS.no_random else None
    batch = predict_utils.create_batch_for_training(
        row.target_chainseq, target_trim_positions, row.templates_alignfile,
        row.native_pdbfile, native_align, FLAGS.crop_size, model_runner,
        native_identities = native_identities,
        native_len = native_len,
        debug=debug,
        verbose=verbose,
        random_seed=random_seed,
    )

    # binder stuff here:
    num_binder_classes = len(FLAGS.binder_intercepts)
    peptide_positions = [int(x) for x in row.peptide_positions.split(';')]
    nonpeptide_positions = [x for x in range(nres) if x not in peptide_positions]
    assert all(0 <= x < nres for x in peptide_positions)
    assert row.binder in [0,1]
    if num_binder_classes>1:
        binder_class_int = int(row.binder_class)
    else:
        binder_class_int = 0
    assert 0 <= binder_class_int < num_binder_classes
    labels = np.ones(1, np.int32) if row.binder else np.zeros(1, np.int32)
    binder_class = np.full(1, binder_class_int, np.int32)
    batch['labels'] = np.eye(2)[labels] # has shape (1,2) I think
    batch['binder_class_1hot'] = np.eye(num_binder_classes)[binder_class]
    assert batch['binder_class_1hot'].shape == (1, num_binder_classes)

    # peptide_mask is used by PlddtBinderClassifier
    batch['peptide_mask'] = np.zeros((FLAGS.crop_size,))
    batch['peptide_mask'][peptide_positions] = 1.

    # partner1/partner2_mask is used by PaeBinderClassifier
    batch['partner1_mask'] = np.zeros((FLAGS.crop_size,))
    batch['partner1_mask'][nonpeptide_positions] = 1.

    batch['partner2_mask'] = np.zeros((FLAGS.crop_size,))
    batch['partner2_mask'][peptide_positions] = 1.

    batch['native_exists'] = bool(row.native_exists)

    return batch




class CustomPandasDataset(torch.utils.data.Dataset):
    def __init__(self, df, loader):
        self.df = df.copy()
        self.loader = loader
    def __len__(self):
        return self.df.shape[0]
    def __getitem__(self, index):
        if FLAGS.debug:
            print('CustomPandasDataset:get_item: index=', index)
        out = self.loader(self.df.iloc[index])
        return out

def collate(samples):
    out_dict = {}
    for name in train_utils.list_a + train_utils.list_a_templates:
        values = [item[name][0,] for item in samples]
        out_dict[name] = np.stack(values, axis=0)
    for name in train_utils.list_b + train_utils.list_b_templates:
        values = [item[name] for item in samples]
        out_dict[name] = np.stack(values, axis=0)
    aatype_ = [item['aatype'][0,] for item in samples]
    out_dict['aatype_'] = np.stack(aatype_, axis=0)
    # binder stuff:
    out_dict['binder_class_1hot' ] = [item['binder_class_1hot' ] for item in samples]
    out_dict['peptide_mask' ] = [item['peptide_mask' ] for item in samples]
    out_dict['partner1_mask'] = [item['partner1_mask'] for item in samples]
    out_dict['partner2_mask'] = [item['partner2_mask'] for item in samples]
    out_dict['labels'] = [item['labels'] for item in samples]
    out_dict['native_exists'] = [item['native_exists'] for item in samples]
    return out_dict


def prep_batch_for_step(batch, training):
    torsion_dict = atom37_to_torsion_angles(
        jnp.array(batch['aatype_']),
        jnp.array(batch['all_atom_positions']),
        jnp.array(batch['all_atom_mask']))
    batch['chi_mask'] = torsion_dict['torsion_angles_mask'][:,:,3:] #[B, N, 4] for 4 chi
    sin_chi = torsion_dict['torsion_angles_sin_cos'][:,:,3:,0]
    cos_chi = torsion_dict['torsion_angles_sin_cos'][:,:,3:,1]
    batch['chi_angles'] = jnp.arctan2(sin_chi, cos_chi) #[B, N, 4] for 4 chi angles
    rigidgroups_dict = atom37_to_frames(
        jnp.array(batch['aatype_']),
        jnp.array(batch['all_atom_positions']),
        jnp.array(batch['all_atom_mask']))
    batch.update(rigidgroups_dict)
    for key_, value_ in batch.items(): # add the 'ensemble' dimension??
        if key_ in train_utils.list_a + train_utils.list_a_templates:
            batch[key_] = value_[:,None,]
        if key_ in train_utils.list_c:
            batch[key_] = value_[:,None,]
    for item_ in train_utils.pdb_key_list_int:
        batch[item_] = jnp.array(batch[item_], jnp.int32)
    if training:
        if FLAGS.random_recycling:
            batch['num_iter_recycling'] = jnp.array(np.tile(
                np.random.randint(0, model_config.model.num_recycle+1, 1)[None,],
                (batch_size, model_config.model.num_recycle)), jnp.int32)
            print('num_iter_recycling:', batch['num_iter_recycling'])
        else:
            print('not setting num_iter_recycling!!! will do',
                  model_config.model.num_recycle,'recycles')

    # binder stuff:
    batch['binder_class_1hot'] = jnp.array(batch['binder_class_1hot'],
                                           dtype=jnp.float32)
    batch['peptide_mask' ] = jnp.array(batch['peptide_mask' ], dtype=jnp.float32)
    batch['partner1_mask'] = jnp.array(batch['partner1_mask'], dtype=jnp.float32)
    batch['partner2_mask'] = jnp.array(batch['partner2_mask'], dtype=jnp.float32)
    #del batch['labels']
    del batch['native_exists'] # already got this
    #del batch['peptide_mask']
    if FLAGS.verbose:
        print('prep_batch_for_step_final_features:', ' '.join(batch.keys()))



def protein_from_prediction(batch, predicted_dict, b_factors=None):
    fold_output = predicted_dict['structure_module']
    if b_factors is None:
        b_factors = np.zeros_like(fold_output['final_atom_mask'][0,])

    return Protein(
        aatype=batch['aatype'][0,0,:],
        atom_positions=fold_output['final_atom_positions'][0,],
        atom_mask=fold_output['final_atom_mask'][0,],
        residue_index=batch['residue_index'][0,0,:] + 1,
        b_factors=b_factors)

def show(name, thing):
    print(name, end=': ')
    if hasattr(thing, 'items'):
        for k,v in thing.items():
            show(k, v)
    elif hasattr(thing, 'shape'):
        print('shape=', thing.shape, end=', ')

def compute_valid_stats(valid_loader, replicated_params, jax_key):
    if FLAGS.no_valid:
        print('no valid stats')
        return
    temp_train_loss = []
    temp_binder_loss = []
    temp_lddt_ca = []
    temp_distogram = []
    temp_masked_msa = []
    temp_exper_res = []
    temp_pred_lddt = []
    temp_chi_loss = []
    temp_fape = []
    temp_sidechain_fape = []
    for n, batch in enumerate(valid_loader):
        print('test_epoch:', e, 'batch:', n)
        structure_flag = batch['native_exists'][0]
        binder_labels = batch['labels']
        peptide_mask = batch['peptide_mask']
        prep_batch_for_step(batch, True)#False)
        if FLAGS.no_random:
            subkey = jax.random.PRNGKey(0)
        else:
            jax_key, subkey = jax.random.split(jax_key)

        loss, grads, predicted_dict, binder_loss, binder_probs, binder_features \
            = jax.pmap(train_step, in_axes=(0,None,0,None), axis_name='model_ax')(
                replicated_params, subkey, batch, structure_flag)
        print('test_epoch_n=', e, n, 'loss=', loss[0],
              'lddt_ca=', np.mean(predicted_dict['predicted_lddt']['lddt_ca']),
              'fape=', np.mean(predicted_dict['structure_module']['fape']),
              'binder_probs=', binder_probs,
              'binder_loss=', binder_loss[0],
              'binder_features=', binder_features,
              'binder_labels=', binder_labels)

        temp_train_loss.append(np.mean(loss[0]))
        temp_binder_loss.append(np.mean(binder_loss[0]))
        temp_lddt_ca.append(np.mean(predicted_dict['predicted_lddt']['lddt_ca']))
        temp_distogram.append(np.mean(predicted_dict['distogram']['loss']))
        temp_masked_msa.append(np.mean(predicted_dict['masked_msa']['loss']))
        #temp_exper_res.append(np.mean(predicted_dict['experimentally_resolved']['loss']))
        temp_pred_lddt.append(np.mean(predicted_dict['predicted_lddt']['loss']))
        temp_chi_loss.append(np.mean(predicted_dict['structure_module']['chi_loss']))
        temp_fape.append(np.mean(predicted_dict['structure_module']['fape']))
        temp_sidechain_fape.append(np.mean(predicted_dict['structure_module']['sidechain_fape']))
        mean_loss = round(float(np.mean(temp_train_loss)),4)
        mean_binder_loss = round(float(np.mean(temp_binder_loss)),4)
        lddt_ca = round(float(np.mean(temp_lddt_ca)),4)
        distogram = round(float(np.mean(temp_distogram)),4)
        masked_msa = round(float(np.mean(temp_masked_msa)),4)
        fape = round(float(np.mean(temp_fape)),4)
        sidechain_fape = round(float(np.mean(temp_sidechain_fape)),4)
        chi_loss = round(float(np.mean(temp_chi_loss)),4)
        print(f'test_Step: {global_step} {n}, loss: {mean_loss}, binder_loss: {mean_binder_loss} lddt: {lddt_ca}, fape: {fape}, sc_fape: {sidechain_fape}, chi: {chi_loss}, disto: {distogram}, msa: {masked_msa}', flush=True)
        sys.stdout.flush()



######################################################################################88
######################################################################################88
######################################################################################88
######################################################################################88
######################################################################################88
######################################################################################88
######################################################################################88


train_df = pd.read_table(FLAGS.train_dataset)
valid_df = pd.read_table(FLAGS.valid_dataset)

training_set  = CustomPandasDataset(train_df, create_batch_from_dataset_row)
valid_set = CustomPandasDataset(valid_df, create_batch_from_dataset_row)

params_loader = {
    'shuffle': True,
    'num_workers': FLAGS.num_cpus,
    'pin_memory': False,
    'batch_size': batch_size,
    'collate_fn': collate,
}

train_loader = torch.utils.data.DataLoader(training_set, **params_loader)
valid_loader = torch.utils.data.DataLoader(valid_set, **params_loader)

if FLAGS.no_ramp:
    scheduler = optax.linear_schedule(1e-3, 1e-3, 1000, 0)
else:
    scheduler = optax.linear_schedule(0.0, 1e-3, 1000, 0)

# Combining gradient transforms using `optax.chain`.
chain_me = [
    optax.scale_by_adam(b1=0.9, b2=0.999, eps=1e-6),  # Use the updates from adam.
    optax.scale_by_schedule(scheduler),  # Use the learning rate from the scheduler.
    # Scale updates by -1 since optax.apply_updates is additive and we want to descend on the loss.
    optax.scale(-1.0*FLAGS.lr_coef),
]

gradient_transform = optax.chain(*chain_me)

n_devices = jax.local_device_count()
assert n_devices == 1, 'Not tested with multiple devices yet'
replicated_params = jax.tree_map(lambda x: jnp.array([x] * n_devices), model_params)

opt_state = gradient_transform.init(replicated_params)
global_step = 0

if FLAGS.test_load:
    # test the setup
    for n, batch in enumerate(train_loader):
        print('trainload:',n)
        sys.stdout.flush()
    for n, batch in enumerate(valid_loader):
        print('validload:',n)
        sys.stdout.flush()

if FLAGS.notrain:
    num_epochs = 1
else:
    num_epochs = FLAGS.num_epochs

for e in range(num_epochs):
    temp_train_loss = []
    temp_binder_loss = []
    temp_lddt_ca = []
    temp_distogram = []
    temp_masked_msa = []
    temp_exper_res = []
    temp_pred_lddt = []
    temp_chi_loss = []
    temp_fape = []
    temp_sidechain_fape = []
    grads_sum, grads_sum_count = None, 0

    for n, batch in enumerate(train_loader):
        if FLAGS.notrain:
            break
        structure_flag = batch['native_exists'][0]
        print('train_epoch:', e, 'batch:', n)
        if 2:
            tmp_params = jax.tree_map(lambda x: x[0,], replicated_params)
            b_params, _ = hk.data_structures.partition(
                lambda m, x, p: m[:9] != "alphafold", tmp_params)
            print('binder_params:', b_params)
        binder_labels = batch['labels']
        prep_batch_for_step(batch, True)
        peptide_mask = batch['peptide_mask']

        if FLAGS.verbose:
            for key_, value_ in batch.items():
                print('before_train_step:', key_, type(value_),
                      getattr(value_, 'shape', 'unk_shape'),
                      getattr(value_, 'dtype', 'unk_dtype'))
        sys.stdout.flush()


        #quoting from the docs: "Like with vmap, we can use in_axes to specify whether an argument to the parallelized function should be broadcast (None), or whether it should be split along a given axis. Note, however, that unlike vmap, only the leading axis (0) is supported by pmap"

        if FLAGS.no_random:
            subkey = jax.random.PRNGKey(0)
        else:
            jax_key, subkey = jax.random.split(jax_key)

        loss, grads, predicted_dict, binder_loss, binder_probs, binder_features \
            = jax.pmap(train_step, in_axes=(0,None,0,None), axis_name='model_ax')(
                replicated_params, subkey, batch, structure_flag)


        plddts = confidence.compute_plddt(predicted_dict['predicted_lddt']['logits'])
        print('train_epoch_n=', e, n, 'loss=', loss[0],
              'structure_flag:', structure_flag,
              'lddt_ca=', np.mean(predicted_dict['predicted_lddt']['lddt_ca']),
              'fape=', np.mean(predicted_dict['structure_module']['fape']),
              'binder_probs=', binder_probs,
              'binder_loss=', binder_loss,
              'peptide_plddt=', np.sum(peptide_mask*plddts)/9,
              'binder_features=', binder_features,
              'binder_labels=', binder_labels,
              'binder_params=', b_params,
        )

        if FLAGS.dump_pdbs:
            unrelaxed_protein = protein_from_prediction(
                batch, predicted_dict)
            outfile = f'{FLAGS.outprefix}train_result_{e:02d}_{n:04d}.pdb'
            with open(outfile, 'w') as f:
                f.write(protein.to_pdb(unrelaxed_protein))
                print('made:', outfile)


        temp_train_loss.append(np.mean(loss[0]))
        temp_binder_loss.append(np.mean(binder_loss[0]))
        temp_lddt_ca.append(np.mean(predicted_dict['predicted_lddt']['lddt_ca']))
        temp_distogram.append(np.mean(predicted_dict['distogram']['loss']))
        temp_masked_msa.append(np.mean(predicted_dict['masked_msa']['loss']))
        temp_pred_lddt.append(np.mean(predicted_dict['predicted_lddt']['loss']))
        temp_chi_loss.append(np.mean(predicted_dict['structure_module']['chi_loss']))
        temp_fape.append(np.mean(predicted_dict['structure_module']['fape']))
        temp_sidechain_fape.append(np.mean(predicted_dict['structure_module']['sidechain_fape']))
        global_step += 1

        # accumulate
        print('grad accumulate:', global_step, grads_sum_count)
        if grads_sum_count == 0:
            grads_sum = grads
        else:
            grads_sum = jax.tree_multimap(lambda x, y: x+y, grads_sum, grads)
        grads_sum_count += 1

        if (grads_sum_count >= FLAGS.apply_every and # time to update
            not FLAGS.freeze_everything):
            print('grad update!', global_step, grads_sum_count)
            grads_sum = jax.tree_map(lambda x: x/grads_sum_count, grads_sum)
            grads_sum = norm_grads_per_example(grads_sum,
                                               l2_norm_clip=FLAGS.grad_norm_clip)

            updates, opt_state = gradient_transform.update(grads_sum, opt_state)
            replicated_params = optax.apply_updates(replicated_params, updates)

            grads_sum, grads_sum_count = None, 0

        if (n+1) % FLAGS.print_steps == 0:
            mean_loss = round(float(np.mean(temp_train_loss)),4)
            binder_loss = round(float(np.mean(temp_binder_loss)),4)
            lddt_ca = round(float(np.mean(temp_lddt_ca)),4)
            distogram = round(float(np.mean(temp_distogram)),4)
            masked_msa = round(float(np.mean(temp_masked_msa)),4)
            fape = round(float(np.mean(temp_fape)),4)
            sidechain_fape = round(float(np.mean(temp_sidechain_fape)),4)
            chi_loss = round(float(np.mean(temp_chi_loss)),4)
            print(f'Step: {global_step}, loss: {mean_loss}, binder_loss: {binder_loss} lddt: {lddt_ca}, fape: {fape}, sc_fape: {sidechain_fape}, chi: {chi_loss}, disto: {distogram}, msa: {masked_msa}', flush=True)
            temp_train_loss = []
            temp_binder_loss = []
            temp_lddt_ca = []
            temp_distogram = []
            temp_masked_msa = []
            temp_exper_res = []
            temp_pred_lddt = []
            temp_chi_loss = []
            temp_fape = []
            temp_sidechain_fape = []
        if (n+1) % FLAGS.save_steps == 0:
            prefix = FLAGS.outprefix
            step_fname = f'{prefix}_af_mhc_global_step.npy'
            np.save(step_fname, global_step)

            param_fname = f'{prefix}_af_mhc_params_{global_step}.pkl'
            save_params = jax.tree_map(lambda x: x[0,], replicated_params)
            with open(param_fname, 'wb') as f:
                pickle.dump(save_params, f)

            state_fname = f'{prefix}_af_mhc_state_{global_step}.pkl'
            with open(state_fname, 'wb') as f:
                pickle.dump(opt_state, f)

            config_fname = f'{prefix}_af_mhc_config.pkl'
            with open(config_fname, 'wb') as f:
                pickle.dump(model_config, f)

        if (n+1)%FLAGS.valid_steps==0:
            compute_valid_stats(valid_loader, replicated_params, jax_key)
        sys.stdout.flush()

    compute_valid_stats(valid_loader, replicated_params, jax_key)
