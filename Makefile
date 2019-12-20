# NOTE: activate conda env udck19 *before* running make or die
# 
# activate like so:
#
# > conda activate udck19

conda_env= env/conda_env.txt
meta_yaml = env/meta.yaml

# run analysis pipeline jupyter notebooks, save output 
RUN_PIPELINE = jupyter nbconvert --to html --output-dir pipeline_out
RUN_PIPELINE += --ExecutePreprocessor.timeout=None --execute 

# enforce environment 
# env = udck19
#env = udck19_pnas
#env = udck19_pnas_110819
env = udck19_pnas_121819

env_check:
ifeq ($(CONDA_DEFAULT_ENV),$(env))
	@echo  checking CONDA_DEFAULT_ENV=$(env) ... OK
else
	$(error Bad conda env, activate first like so: conda activate $(env))
endif

snapshot_env: env_check
	echo "snapshotting env"
	echo "# " $(env) > $(conda_env)
	echo "# " `date` >> $(conda_env)
	conda list -e >> $(conda_env)

	echo "# " $(env) > $(meta_yaml)
	echo "# " `date` >> $(meta_yaml)
	conda env export >> $(meta_yaml)

file_check: 
	@echo 'checking file names and MD5s'
	./scripts/udck19_filenames.py

norming:  snapshot_env file_check
	@echo 'calculating normative stim item measures'
	./scripts/udck19_norming.py

epochs: snapshot_env file_check 
	@echo 'slicing, decorating, preprocessing, and exporting EEG single trial epochs'
	./scripts/udck19_single_trial_wrangling.py

# artifact screening
pipeline_1: snapshot_env file_check
	 $(RUN_PIPELINE) ./scripts/udck19_pipeline_1.ipynb

# linear mixed effect modelling ... full dataset ~ 30hrs
pipeline_2: snapshot_env file_check
	 $(RUN_PIPELINE) ./scripts/udck19_pipeline_2.ipynb

# model comparisons
pipeline_3: snapshot_env file_check
	$(RUN_PIPELINE) ./scripts/udck19_pipeline_3.ipynb

# influence diagnostics
pipeline_4: snapshot_env file_check
	$(RUN_PIPELINE) ./scripts/udck19_pipeline_4.ipynb

# N4 interval mean amplitude model comparisons
pipeline_5: snapshot_env file_check
	$(RUN_PIPELINE) ./scripts/udck19_pipeline_5.ipynb

pipeline: pipeline_1 pipeline_2 pipeline_3 pipeline_4 pipeline_5

all: norming epochs pipeline
