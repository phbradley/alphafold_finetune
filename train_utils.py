import jax
import jax.numpy as jnp
import numpy as np
from typing import Tuple
from alphafold.common import residue_constants

# this is mostly Justas magic with some Phil hacks at the end
# some of this code is repurposed from the alphafold package

pdb_key_list = ['atom14_atom_exists',
                'residx_atom14_to_atom37',
                'residx_atom37_to_atom14',
                'atom37_atom_exists',
                'pseudo_beta',
                'pseudo_beta_mask',
                'all_atom_mask',
                'chi_mask',
                'chi_angles',
                'all_atom_positions',
                'atom14_gt_exists',
                'atom14_gt_positions',
                'atom14_alt_gt_positions',
                'atom14_alt_gt_exists',
                'atom14_atom_is_ambiguous',
                'rigidgroups_gt_frames',
                'rigidgroups_gt_exists',
                'rigidgroups_group_exists',
                'rigidgroups_group_is_ambiguous',
                'rigidgroups_alt_gt_frames',
                'backbone_translation',
                'backbone_rotation',
                'backbone_affine_mask']



pdb_key_list_int = ['residx_atom14_to_atom37',
                    'residx_atom37_to_atom14']


list_a = ['atom14_atom_exists',
        'residx_atom14_to_atom37',
        'residx_atom37_to_atom14',
        'atom37_atom_exists',
        'pseudo_beta',
        'pseudo_beta_mask',
        'all_atom_mask',
        'resolution',
        'all_atom_positions',
        'atom14_gt_exists',
        'atom14_gt_positions',
        'atom14_alt_gt_positions',
        'atom14_alt_gt_exists',
        'atom14_atom_is_ambiguous',
        'backbone_translation',
        'backbone_rotation',
        'backbone_affine_mask']

list_a_templates = [
  'template_aatype',
  'template_all_atom_masks',
  'template_all_atom_positions',
  #'template_backbone_affine_mask',
  #'template_backbone_affine_tensor',
  'template_pseudo_beta',
  'template_pseudo_beta_mask',
  'template_sum_probs',
]
list_b_templates = [
  'template_mask',
]

list_b = ['aatype',
        'residue_index',
        'seq_length',
        'is_distillation',
        'seq_mask',
        'msa_mask',
        'msa_row_mask',
        'random_crop_to_size_seed',
        'extra_msa',
        'extra_msa_mask',
        'extra_msa_row_mask',
        'bert_mask',
        'true_msa',
        'extra_has_deletion',
        'extra_deletion_value',
        'msa_feat',
        'target_feat']


list_c = ['chi_mask',
        'chi_angles',
        'rigidgroups_gt_frames',
        'rigidgroups_gt_exists',
        'rigidgroups_group_exists',
        'rigidgroups_group_is_ambiguous',
        'rigidgroups_alt_gt_frames']


pdb_key_list_int = ['residx_atom14_to_atom37',
                    'residx_atom37_to_atom14']


def pseudo_beta_fn_np(aatype, all_atom_positions, all_atom_masks):
  """Create pseudo beta features."""

  is_gly = np.equal(aatype, residue_constants.restype_order['G'])
  ca_idx = residue_constants.atom_order['CA']
  cb_idx = residue_constants.atom_order['CB']
  pseudo_beta = np.where(
      np.tile(is_gly[..., None], [1] * len(is_gly.shape) + [3]),
      all_atom_positions[..., ca_idx, :],
      all_atom_positions[..., cb_idx, :])

  if all_atom_masks is not None:
    pseudo_beta_mask = np.where(
        is_gly, all_atom_masks[..., ca_idx], all_atom_masks[..., cb_idx])
    pseudo_beta_mask = pseudo_beta_mask.astype(np.float32)
    return pseudo_beta, pseudo_beta_mask
  else:
    return pseudo_beta

def apply_rot_to_vec(rot, vec, unstack=False):
  """Multiply rotation matrix by a vector."""
  if unstack:
    x, y, z = [vec[:, i] for i in range(3)]
  else:
    x, y, z = vec
  return [rot[0][0] * x + rot[0][1] * y + rot[0][2] * z,
          rot[1][0] * x + rot[1][1] * y + rot[1][2] * z,
          rot[2][0] * x + rot[2][1] * y + rot[2][2] * z]

def _multiply(a, b):
  return np.stack([
      np.array([a[0][0]*b[0][0] + a[0][1]*b[1][0] + a[0][2]*b[2][0],
                 a[0][0]*b[0][1] + a[0][1]*b[1][1] + a[0][2]*b[2][1],
                 a[0][0]*b[0][2] + a[0][1]*b[1][2] + a[0][2]*b[2][2]]),

      np.array([a[1][0]*b[0][0] + a[1][1]*b[1][0] + a[1][2]*b[2][0],
                 a[1][0]*b[0][1] + a[1][1]*b[1][1] + a[1][2]*b[2][1],
                 a[1][0]*b[0][2] + a[1][1]*b[1][2] + a[1][2]*b[2][2]]),

      np.array([a[2][0]*b[0][0] + a[2][1]*b[1][0] + a[2][2]*b[2][0],
                 a[2][0]*b[0][1] + a[2][1]*b[1][1] + a[2][2]*b[2][1],
                 a[2][0]*b[0][2] + a[2][1]*b[1][2] + a[2][2]*b[2][2]])])

def make_canonical_transform(
    n_xyz: np.ndarray,
    ca_xyz: np.ndarray,
    c_xyz: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
  """Returns translation and rotation matrices to canonicalize residue atoms.
  Note that this method does not take care of symmetries. If you provide the
  atom positions in the non-standard way, the N atom will end up not at
  [-0.527250, 1.359329, 0.0] but instead at [-0.527250, -1.359329, 0.0]. You
  need to take care of such cases in your code.
  Args:
    n_xyz: An array of shape [batch, 3] of nitrogen xyz coordinates.
    ca_xyz: An array of shape [batch, 3] of carbon alpha xyz coordinates.
    c_xyz: An array of shape [batch, 3] of carbon xyz coordinates.
  Returns:
    A tuple (translation, rotation) where:
      translation is an array of shape [batch, 3] defining the translation.
      rotation is an array of shape [batch, 3, 3] defining the rotation.
    After applying the translation and rotation to all atoms in a residue:
      * All atoms will be shifted so that CA is at the origin,
      * All atoms will be rotated so that C is at the x-axis,
      * All atoms will be shifted so that N is in the xy plane.
  """
  assert len(n_xyz.shape) == 2, n_xyz.shape
  assert n_xyz.shape[-1] == 3, n_xyz.shape
  assert n_xyz.shape == ca_xyz.shape == c_xyz.shape, (
      n_xyz.shape, ca_xyz.shape, c_xyz.shape)

  # Place CA at the origin.
  translation = -ca_xyz
  n_xyz = n_xyz + translation
  c_xyz = c_xyz + translation

  # Place C on the x-axis.
  c_x, c_y, c_z = [c_xyz[:, i] for i in range(3)]
  # Rotate by angle c1 in the x-y plane (around the z-axis).
  sin_c1 = -c_y / np.sqrt(1e-20 + c_x**2 + c_y**2)
  cos_c1 = c_x / np.sqrt(1e-20 + c_x**2 + c_y**2)
  zeros = np.zeros_like(sin_c1)
  ones = np.ones_like(sin_c1)
  # pylint: disable=bad-whitespace
  c1_rot_matrix = np.stack([np.array([cos_c1, -sin_c1, zeros]),
                             np.array([sin_c1,  cos_c1, zeros]),
                             np.array([zeros,    zeros,  ones])])

  # Rotate by angle c2 in the x-z plane (around the y-axis).
  sin_c2 = c_z / np.sqrt(1e-20 + c_x**2 + c_y**2 + c_z**2)
  cos_c2 = np.sqrt(c_x**2 + c_y**2) / np.sqrt(
      1e-20 + c_x**2 + c_y**2 + c_z**2)
  c2_rot_matrix = np.stack([np.array([cos_c2,  zeros, sin_c2]),
                             np.array([zeros,    ones,  zeros]),
                             np.array([-sin_c2, zeros, cos_c2])])

  c_rot_matrix = _multiply(c2_rot_matrix, c1_rot_matrix)
  n_xyz = np.stack(apply_rot_to_vec(c_rot_matrix, n_xyz, unstack=True)).T

  # Place N in the x-y plane.
  _, n_y, n_z = [n_xyz[:, i] for i in range(3)]
  # Rotate by angle alpha in the y-z plane (around the x-axis).
  sin_n = -n_z / np.sqrt(1e-20 + n_y**2 + n_z**2)
  cos_n = n_y / np.sqrt(1e-20 + n_y**2 + n_z**2)
  n_rot_matrix = np.stack([np.array([ones,  zeros,  zeros]),
                            np.array([zeros, cos_n, -sin_n]),
                            np.array([zeros, sin_n,  cos_n])])
  # pylint: enable=bad-whitespace

  return (translation,
          np.transpose(_multiply(n_rot_matrix, c_rot_matrix), [2, 0, 1]))


def make_transform_from_reference_np(
    n_xyz: np.ndarray,
    ca_xyz: np.ndarray,
    c_xyz: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
  """Returns rotation and translation matrices to convert from reference.
  Note that this method does not take care of symmetries. If you provide the
  atom positions in the non-standard way, the N atom will end up not at
  [-0.527250, 1.359329, 0.0] but instead at [-0.527250, -1.359329, 0.0]. You
  need to take care of such cases in your code.
  Args:
    n_xyz: An array of shape [batch, 3] of nitrogen xyz coordinates.
    ca_xyz: An array of shape [batch, 3] of carbon alpha xyz coordinates.
    c_xyz: An array of shape [batch, 3] of carbon xyz coordinates.
  Returns:
    A tuple (rotation, translation) where:
      rotation is an array of shape [batch, 3, 3] defining the rotation.
      translation is an array of shape [batch, 3] defining the translation.
    After applying the translation and rotation to the reference backbone,
    the coordinates will approximately equal to the input coordinates.
    The order of translation and rotation differs from make_canonical_transform
    because the rotation from this function should be applied before the
    translation, unlike make_canonical_transform.
  """
  translation, rotation = make_canonical_transform(n_xyz, ca_xyz, c_xyz)
  return np.transpose(rotation, (0, 2, 1)), -translation


# hack, including this function here so we don't have to import alphafold.relax
#  since it has some extra dependencies we don't have/need

def make_atom14_positions(prot):
  """Constructs denser atom positions (14 dimensions instead of 37)."""
  restype_atom14_to_atom37 = []  # mapping (restype, atom14) --> atom37
  restype_atom37_to_atom14 = []  # mapping (restype, atom37) --> atom14
  restype_atom14_mask = []

  for rt in residue_constants.restypes:
    atom_names = residue_constants.restype_name_to_atom14_names[
        residue_constants.restype_1to3[rt]]

    restype_atom14_to_atom37.append([
        (residue_constants.atom_order[name] if name else 0)
        for name in atom_names
    ])

    atom_name_to_idx14 = {name: i for i, name in enumerate(atom_names)}
    restype_atom37_to_atom14.append([
        (atom_name_to_idx14[name] if name in atom_name_to_idx14 else 0)
        for name in residue_constants.atom_types
    ])

    restype_atom14_mask.append([(1. if name else 0.) for name in atom_names])

  # Add dummy mapping for restype 'UNK'.
  restype_atom14_to_atom37.append([0] * 14)
  restype_atom37_to_atom14.append([0] * 37)
  restype_atom14_mask.append([0.] * 14)

  restype_atom14_to_atom37 = np.array(restype_atom14_to_atom37, dtype=np.int32)
  restype_atom37_to_atom14 = np.array(restype_atom37_to_atom14, dtype=np.int32)
  restype_atom14_mask = np.array(restype_atom14_mask, dtype=np.float32)

  # Create the mapping for (residx, atom14) --> atom37, i.e. an array
  # with shape (num_res, 14) containing the atom37 indices for this protein.
  residx_atom14_to_atom37 = restype_atom14_to_atom37[prot["aatype"]]
  residx_atom14_mask = restype_atom14_mask[prot["aatype"]]

  # Create a mask for known ground truth positions.
  residx_atom14_gt_mask = residx_atom14_mask * np.take_along_axis(
      prot["all_atom_mask"], residx_atom14_to_atom37, axis=1).astype(np.float32)

  # Gather the ground truth positions.
  residx_atom14_gt_positions = residx_atom14_gt_mask[:, :, None] * (
      np.take_along_axis(prot["all_atom_positions"],
                         residx_atom14_to_atom37[..., None],
                         axis=1))

  prot["atom14_atom_exists"] = residx_atom14_mask
  prot["atom14_gt_exists"] = residx_atom14_gt_mask
  prot["atom14_gt_positions"] = residx_atom14_gt_positions

  prot["residx_atom14_to_atom37"] = residx_atom14_to_atom37

  # Create the gather indices for mapping back.
  residx_atom37_to_atom14 = restype_atom37_to_atom14[prot["aatype"]]
  prot["residx_atom37_to_atom14"] = residx_atom37_to_atom14

  # Create the corresponding mask.
  restype_atom37_mask = np.zeros([21, 37], dtype=np.float32)
  for restype, restype_letter in enumerate(residue_constants.restypes):
    restype_name = residue_constants.restype_1to3[restype_letter]
    atom_names = residue_constants.residue_atoms[restype_name]
    for atom_name in atom_names:
      atom_type = residue_constants.atom_order[atom_name]
      restype_atom37_mask[restype, atom_type] = 1

  residx_atom37_mask = restype_atom37_mask[prot["aatype"]]
  prot["atom37_atom_exists"] = residx_atom37_mask

  # As the atom naming is ambiguous for 7 of the 20 amino acids, provide
  # alternative ground truth coordinates where the naming is swapped
  restype_3 = [
      residue_constants.restype_1to3[res] for res in residue_constants.restypes
  ]
  restype_3 += ["UNK"]

  # Matrices for renaming ambiguous atoms.
  all_matrices = {res: np.eye(14, dtype=np.float32) for res in restype_3}
  for resname, swap in residue_constants.residue_atom_renaming_swaps.items():
    correspondences = np.arange(14)
    for source_atom_swap, target_atom_swap in swap.items():
      source_index = residue_constants.restype_name_to_atom14_names[
          resname].index(source_atom_swap)
      target_index = residue_constants.restype_name_to_atom14_names[
          resname].index(target_atom_swap)
      correspondences[source_index] = target_index
      correspondences[target_index] = source_index
      renaming_matrix = np.zeros((14, 14), dtype=np.float32)
      for index, correspondence in enumerate(correspondences):
        renaming_matrix[index, correspondence] = 1.
    all_matrices[resname] = renaming_matrix.astype(np.float32)
  renaming_matrices = np.stack([all_matrices[restype] for restype in restype_3])

  # Pick the transformation matrices for the given residue sequence
  # shape (num_res, 14, 14).
  renaming_transform = renaming_matrices[prot["aatype"]]

  # Apply it to the ground truth positions. shape (num_res, 14, 3).
  alternative_gt_positions = np.einsum("rac,rab->rbc",
                                       residx_atom14_gt_positions,
                                       renaming_transform)
  prot["atom14_alt_gt_positions"] = alternative_gt_positions

  # Create the mask for the alternative ground truth (differs from the
  # ground truth mask, if only one of the atoms in an ambiguous pair has a
  # ground truth position).
  alternative_gt_mask = np.einsum("ra,rab->rb",
                                  residx_atom14_gt_mask,
                                  renaming_transform)

  prot["atom14_alt_gt_exists"] = alternative_gt_mask

  # Create an ambiguous atoms mask.  shape: (21, 14).
  restype_atom14_is_ambiguous = np.zeros((21, 14), dtype=np.float32)
  for resname, swap in residue_constants.residue_atom_renaming_swaps.items():
    for atom_name1, atom_name2 in swap.items():
      restype = residue_constants.restype_order[
          residue_constants.restype_3to1[resname]]
      atom_idx1 = residue_constants.restype_name_to_atom14_names[resname].index(
          atom_name1)
      atom_idx2 = residue_constants.restype_name_to_atom14_names[resname].index(
          atom_name2)
      restype_atom14_is_ambiguous[restype, atom_idx1] = 1
      restype_atom14_is_ambiguous[restype, atom_idx2] = 1

  # From this create an ambiguous_mask for the given sequence.
  prot["atom14_atom_is_ambiguous"] = (
      restype_atom14_is_ambiguous[prot["aatype"]])

  # for k,v in prot.items():
  #   print('make_atom14_positions:', type(v), k)

  return prot


## these are jax versions of the same-named numpy functions from alphafold
## we need them since we are computing plddt/pae within the binder model

def compute_plddt_jax(logits): #: np.ndarray) -> np.ndarray:
  """Computes per-residue pLDDT from logits.

  Args:
    logits: [num_res, num_bins] output from the PredictedLDDTHead.

  Returns:
    plddt: [num_res] per-residue pLDDT.
  """
  num_bins = logits.shape[-1]
  bin_width = 1.0 / num_bins
  bin_centers = jnp.arange(start=0.5 * bin_width, stop=1.0, step=bin_width)
  probs = jax.nn.softmax(logits, axis=-1)
  predicted_lddt_ca = jnp.sum(probs * bin_centers[None, :], axis=-1)
  return predicted_lddt_ca * 100


def _calculate_bin_centers_jax(breaks): #: np.ndarray):
  """Gets the bin centers from the bin edges.

  Args:
    breaks: [num_bins - 1] the error bin edges.

  Returns:
    bin_centers: [num_bins] the error bin centers.
  """
  step = (breaks[1] - breaks[0])

  # Add half-step to get the center
  bin_centers = breaks + step / 2
  # Add a catch-all bin at the end.
  bin_centers = jnp.concatenate([bin_centers, jnp.array([bin_centers[-1] + step])],
                                axis=0)
  return bin_centers


def _calculate_expected_aligned_error_jax(
    alignment_confidence_breaks, #: np.ndarray,
    aligned_distance_error_probs, #: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
):
  """Calculates expected aligned distance errors for every pair of residues.

  Args:
    alignment_confidence_breaks: [num_bins - 1] the error bin edges.
    aligned_distance_error_probs: [num_res, num_res, num_bins] the predicted
      probs for each error bin, for each pair of residues.

  Returns:
    predicted_aligned_error: [num_res, num_res] the expected aligned distance
      error for each pair of residues.
    max_predicted_aligned_error: The maximum predicted error possible.
  """
  bin_centers = _calculate_bin_centers_jax(alignment_confidence_breaks)

  # Tuple of expected aligned distance error and max possible error.
  return (jnp.sum(aligned_distance_error_probs * bin_centers, axis=-1),
          jnp.asarray(bin_centers[-1]))


def compute_predicted_aligned_error_jax(
    logits, #: np.ndarray,
    breaks, #: np.ndarray) -> Dict[str, np.ndarray]:
):
  """Computes aligned confidence metrics from logits.

  Args:
    logits: [num_res, num_res, num_bins] the logits output from
      PredictedAlignedErrorHead.
    breaks: [num_bins - 1] the error bin edges.

  Returns:
    aligned_confidence_probs: [num_res, num_res, num_bins] the predicted
      aligned error probabilities over bins for each residue pair.
    predicted_aligned_error: [num_res, num_res] the expected aligned distance
      error for each pair of residues.
    max_predicted_aligned_error: The maximum predicted error possible.
  """
  # aligned_confidence_probs = scipy.special.softmax(
  #     logits,
  #     axis=-1)
  aligned_confidence_probs = jax.nn.softmax(logits, axis=-1)
  predicted_aligned_error, max_predicted_aligned_error = (
      _calculate_expected_aligned_error_jax(
          alignment_confidence_breaks=breaks,
          aligned_distance_error_probs=aligned_confidence_probs))
  return {
      'aligned_confidence_probs': aligned_confidence_probs,
      'predicted_aligned_error': predicted_aligned_error,
      'max_predicted_aligned_error': max_predicted_aligned_error,
  }

