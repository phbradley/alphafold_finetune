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
python run_finetuning.py --binder_intercepts 0.80367635 --binder_intercepts 0.43373787  --freeze_binder  --train_dataset datasets_alphafold_finetune/pmhc_finetune/combo_1and2_train.tsv --valid_dataset datasets_alphafold_finetune/pmhc_finetune/combo_1and2_valid.tsv

```

## Running predictions of peptide binding

with default alphafold params. Here $ALPHAFOLD_DATA_DIR should point to a directory that contains the `params/` subdirectory.
```
python3 run_prediction.py --targets examples/pmhc_hcv_polg_10mers/targets.tsv --data_dir $ALPHAFOLD_DATA_DIR --outfile_prefix polg_test1 --model_names model_2_ptm
```

or with fine-tuned params

```
python3 run_prediction.py --targets examples/pmhc_hcv_polg_10mers/targets.tsv --outfile_prefix polg_test2 --model_names model_2_ptm_ft --model_params_files datasets_alphafold_finetune/params/mixed_mhc_pae_run6_af_mhc_params_20640.pkl
```



