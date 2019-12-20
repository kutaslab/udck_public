# udck19 single trial analysis reproduction recipe

As of 10/25/2019 kutaslab/mkpy and kutaslab/fitgrid are bundled on Anaconda Cloud

cmudict 0.4.3 is packaged for conda from PyPI on kutaslab.

Other dependencies include spacy=2.1.8 spacy-model-en_core_web_lg=2.1

## One-time setup 

A conda environment for udck19 may be built like so

* Download and install `miniconda` (https://conda.io/docs/user-guide/install/download.html)

* Build and activate the udck19 environment from the requirements like so

  ```
  conda create --name udck19_pnas_MMDDYY --file env/conda_env -c kutaslab -c conda-forge -c defaults
  conda activate udck19_pnas_MMDDYY
  ```

## Run the analysis pipeline

Ensure all the output directories and files are writeable and
`PRERUN=False` in `scripts/udck19_pipeline_*.ipynb` then 
activate the environment and run the make file

```bash
conda activate udck19_pnas_MMDDYY
make all
```

This generates and overwrites all intermediate and final data files,
logs, figures, and pipeline output without warning.

Snapshots of raw EEG recordings are copied over from ../legacy2dif/mkh5
amended with epochs data during single trial wrangling.

See the `Makefile` for details.

Progress may be checked by inspecting the tail of the log files in
`./scripts`. 

The pipeline can be tested in about an hour or two on subsets of the
data by setting `PRERUN=True` in the notebooks. Prerunning overwrites
all files except the costly LMER summary data files which
`PRERUN=True` stashes in `./data/modeling/prerun`.

Runtime for the full data set takes around 30 hours with 36 parallel
cores on a high performance server, nearly all of it spent fitting LMER
models with fitgrid.


## Review the output

* run logs are in `./scripts/udck19_*.py.log`

* norming data in `./udck19/analysis/measures/`

* rERP data files in `./udck19/analysis/modeling/*.h5`

. Data analysis notebook pipeline outputs in `./udck19/analysis/pipeline_out`

These files have human and machine readable information about the
normative stim, experimental designs and coding schemes.

Idiosyncratic item-wise processing is specified here

```
./scripts/eeg_expt_specs.yml
./scripts/norm_expt_specs.yml
```

## IRB Protected single trial data

* Single trial hdf5 EEG epochs data in `./udck19/analysis/data/epochs/*.h5`


# Notes

## Filenames and MD5 checksums

Filenames and input file MD5 checksums are checked at each phase of
the analysis pipeline, hardcoded here:

```
./anlysis/scripts/udck19_filenames.py
```
The current disk file checksums are checked against the hardcode by 
running the file as an executable script. 

## Pipeline and file locations

`udck19_single_trial_wrangling.py`

Routines to process and generate IRB protected single trial data EEG
strip charts and segmented epochs of them.  The processes and
transform parameters are found in `__main__` and reported in the
`udck19_single_trial_wrangling.py.log`

* Read HDF5 format mkh5 EEG data for arquant (eeg_1), arcadj (eeg_2),
  and yantana (eeg_3). These files are found in
	 
  ```
  eeg_1 = ./udck19/analysis/data/eeg/arcadj/arcadj.h5
  eeg_2 = ./udck19/analysis/data/eeg/yantana/yantana.h5
  eeg_3 = ./udck19/analysis/data/eeg/arquant/arquant.h5
  ```

* Extract tabular single trial epochs timelocked to the article
  (a/an) stimulus triggered event code, and export via pandas as HDF5.
	 
  The `*.recorded_epochs.h5` segments of the continuous EEG data as
  found in the strip chart. These are not analyzed in the pipeline
  but may be of interest for other purposes.

  ```
  ./udck19/analysis/data/epochs/eeg_1_recorded_epochs.h5
  ./udck19/analysis/data/epochs/eeg_3_recorded_epochs.h5
  ./udck19/analysis/data/epochs/eeg_2_recorded_epochs.h5
  ```
	 
   The `*.prepared_epochs.h5` are the recorded_epochs after the
   preparatory transformations ("preprocessing") implemented in
   `udck19_single_trial_wrangling.py` such as centering on a
   baseline, filtering

   ```
   ./udck19/analysis/data/epochs/eeg_1_prepared_epochs.h5
   ./udck19/analysis/data/epochs/eeg_3_prepared_epochs.h5
   ./udck19/analysis/data/epochs/eeg_2_prepared_epochs.h5
    ```


## Implementation

# Debugging

Portions of the pipeline may be run separately, see the Makefile.

```bash
make norming
make epochs
make pipeline_1
make pipeline_2
make pipeline_3
make pipeline_4
make pipeline_5
make pipepline
```


