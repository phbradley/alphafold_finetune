# alphafold_finetune
Python code for fine-tuning AlphaFold to perform protein-peptide binding predictions.
This repository is a collaborative project: Justas Dauparas implemented the
AlphaFold changes necessary for fine-tuning and wrote a template of the
`run_finetuning.py` script. Phil Bradley and Amir Motmaen further developed and
extensively tested that template for protein:peptide binding.


# Examples

## Fine-tuning for peptide-MHC on a tiny dataset

```
python run_finetuning.py --binder_intercepts 0.80367635 --binder_intercepts 0.43373787  --freeze_binder  --train_dataset examples/tiny_pmhc_finetune/tiny_example_train.tsv --valid_dataset examples/tiny_pmhc_finetune/tiny_example_valid.tsv
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
python3 run_prediction.py --targets examples/pmhc_hcv_polg_10mers/targets.tsv --data_dir $ALPHAFOLD_DATA_DIR --outfile_prefix polg_test1 --model_names model_2_ptm --ignore_identities
```

HLA-A*02:01 10mer scan with fine-tuned params

```
python3 run_prediction.py --targets examples/pmhc_hcv_polg_10mers/targets.tsv --outfile_prefix polg_test2 --model_names model_2_ptm_ft --model_params_files datasets_alphafold_finetune/params/mixed_mhc_pae_run6_af_mhc_params_20640.pkl --ignore_identities
```

Model 10 random peptides per target for 17 PDZ domains, with default params

```
python3 run_prediction.py --targets examples/pdz/pdz_10_random_peptides.tsv \
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
python3 run_prediction.py --targets examples/sh3/sh3_10_random_peptides.tsv \
    --ignore_identities --outfile_prefix sh3_test1 \
    --data_dir $ALPHAFOLD_DATA_DIR \
    --model_names model_2_ptm model_2_ptm_ft \
    --model_params_files classic datasets_alphafold_finetune/params/mixed_mhc_pae_run6_af_mhc_params_20640.pkl
```
