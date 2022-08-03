# alphafold_finetune
Python code for fine-tuning AlphaFold to perform protein-peptide binding predictions.
This repository is a collaborative effort: Justas Dauparas implemented the
AlphaFold changes necessary for fine-tuning and wrote a template of the
fine-tuning script. Amir Motmaen and Phil Bradley further developed and
extensively tested the fine-tuning and inference scripts in the context of
protein-peptide binding.

This repository is still under development. We plan to have everything squared away
prior to publication. In the meantime, feel free to reach out with questions,
comments, or other feedback. You can open a github issue or email
`pbradley at fredhutch.org`.

UPDATE: We are having trouble with Zenodo file uploads, so in the meantime,
I uploaded a preliminary dataset with the fine-tuned parameters and the training and
testing datasets to dropbox:

https://www.dropbox.com/s/k4gay3hwyq3k0rb/datasets_alphafold_finetune_v1_2022-08-02.tgz?dl=0

Once you download the `.tgz` file, copy it into the `alphafold_finetune/` folder and
uncompress it, something like

`tar -xzvf datasets_alphafold_finetune_v1_2022-08-02.tgz`

That should create a new folder called `datasets_alphafold_finetune/` and hopefully
the relevant examples will work.

# Examples

## Fine-tuning for peptide-MHC on a tiny dataset

```
python run_finetuning.py \
    --binder_intercepts 0.80367635 --binder_intercepts 0.43373787  \
    --freeze_binder  \
    --train_dataset examples/tiny_pmhc_finetune/tiny_example_train.tsv \
    --valid_dataset examples/tiny_pmhc_finetune/tiny_example_valid.tsv
```

## Fine-tuning peptide-MHC (full model)

For this, please download the companion dataset on Zenodo (*Phil to insert info*)

```
python run_finetuning.py \
    --binder_intercepts 0.80367635 --binder_intercepts 0.43373787 \
    --freeze_binder  \
    --train_dataset datasets_alphafold_finetune/pmhc_finetune/combo_1and2_train.tsv \
    --valid_dataset datasets_alphafold_finetune/pmhc_finetune/combo_1and2_valid.tsv

```

## Running predictions of peptide binding

HLA-A*02:01 10mer scan, with default alphafold params. Here $ALPHAFOLD_DATA_DIR should point to a directory that contains the `params/` subdirectory.
```
python run_prediction.py --targets examples/pmhc_hcv_polg_10mers/targets.tsv \
    --data_dir $ALPHAFOLD_DATA_DIR --outfile_prefix polg_test1 \
    --model_names model_2_ptm --ignore_identities
```

HLA-A*02:01 10mer scan with fine-tuned params

```
python run_prediction.py --targets examples/pmhc_hcv_polg_10mers/targets.tsv \
    --outfile_prefix polg_test2 --model_names model_2_ptm_ft \
    --model_params_files datasets_alphafold_finetune/params/mixed_mhc_pae_run6_af_mhc_params_20640.pkl \
    --ignore_identities
```

Model 10 random peptides per target for 17 PDZ domains, with default params

```
python run_prediction.py --targets examples/pdz/pdz_10_random_peptides.tsv \
    --data_dir $ALPHAFOLD_DATA_DIR --outfile_prefix pdz_test1 \
    --model_names model_2_ptm --ignore_identities
```

Model 10 random peptides per target/class for 23 SH3 domains, with default model_2_ptm
AND fine-tuned params. Here we pass multiple values for `--model_names` and
`--model_params_files`, and the values should correspond 1:1 in order.
We give the string `'classic'` in place of the parameter filename for the
non-fine-tuned parameters, which is the signal to load default parameters from the
`params/` folder in `$ALPHAFOLD_DATA_DIR`.

```
python run_prediction.py --targets examples/sh3/sh3_10_random_peptides.tsv \
    --ignore_identities --outfile_prefix sh3_test1 \
    --data_dir $ALPHAFOLD_DATA_DIR \
    --model_names model_2_ptm model_2_ptm_ft \
    --model_params_files classic datasets_alphafold_finetune/params/mixed_mhc_pae_run6_af_mhc_params_20640.pkl
```


# File formats

## Inputs

### targets files

Files with lists of modeling targets (for `run_prediction.py`) or training examples
(for `run_finetuning.py`) should be formatted as tab-separated values files.
See examples in `examples/*/*tsv`. The required fields are

* `target_chainseq`: the full amino acid sequence to be modelled as a single string with '/' characters separating the individual chains. See examples in `examples/*/*tsv`.

* `templates_alignfile`: Filename of the 'alignment file' that contains information on
alphafold modeling templates (the format for alignment files is given below).

For fine-tuning, these additional fields are required:

* `native_pdbfile`: PDB file with native (ground truth) coordinates (could be a
model if following the self-distillation procedure.

* `native_alignstring`: 0-indexed sequence mapping between the target
amino acid sequence and the native amino acid sequence. The two sequences are numbered
starting at 0 with the first amino acid of the first chain, proceeding up to `nres-1`
for the last amino acid of the last chain, where `nres` is the total number of amino
acids in the concatenated sequence (the sum of the chain lengths). Formatted as a
';'-separated string like '1:0;2:1;3:2;4:3' . Same format as the
`target_to_template_alignstring` column of the alignment files.

* `binder`: 0 or 1 to indicate if the training/testing example represents a binder (1)
or a non-binder (0)

* `peptide_positions`: The positions in the target sequence that belong to the peptide.
Formatter as a ';'-separate list of 0-indexed integers, e.g. '201;202;203;204;205'

* `native_exists`: Whether or not the native_pdbfile is a real PDB structure
(True) or a modelled structure (False).
Used to set the weight on the structure loss in the combined
structure+binder loss function; related to the `--fake_native_weight` command line
argument of `run_finetuning.py` (default is 0.25). This allows us to put less weight
on recovering the modelled structures than the experimental structures.

Optional arguments:

* `binder_class`: An integer (ranging from 0 up to `num_binder_classes-1`) that
tells which class of binder this example belongs to. Used when fitting a binder
model with different binder/non-binder PAE or pLDDT switchpoints depending on the
binder class.

* `native_identities`: Count of sequence identities in native_alignstring, used
for error-checking the I/O

* `native_len`: Total number of amino acids in the native_pdbfile, used for
error-checking.

### alignment files

These tab-separated values files provide information on the alphafold modeling
template structures and their alignments to the target sequence.
See `examples/*/alignments/*` for examples. The required fields are

* `template_pdbfile`: PDB format file with template coordinates.
It's safest if this has been "cleaned" to remove water, small molecules, and
problematic amino acids. The very simple PDB reader included in `predict_utils.py`
is not robust.

* `target_to_template_alignstring`: 0-indexed sequence mapping between the target
amino acid sequence and the template amino acid sequence. The two sequences are numbered
starting at 0 with the first amino acid of the first chain, proceeding up to `nres-1`
for the last amino acid of the last chain, where `nres` is the total number of amino
acids in the concatenated sequence (the sum of the chain lengths). Formatted as a
';'-separated string like '1:0;2:1;3:2;4:3' . See `examples/*/alignments/*` for
examples.

* `target_len`: Length of the target sequence (for error checking).

* `template_len`: Length of the template sequence (for error checking).

* `identities`: Count of identical amino acids in the sequence alignment
(for error checking). When modeling lots of different sequences
(e.g., many random peptides) using the same
templates file, it can be helpful to ignore the identities column since it won't
match up for all the different targets. This can be done by passing
`--ignore_identities` to `run_prediction.py`.


