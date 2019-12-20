# udck19

These analyses were conducted in a conda virtual environment using the
Anaconda Data Science platform (www.anaconda.com)

See reproduction_recipe.md

# Data sharing

* Files in the `data` directory contain continuous and epoched single
  trial EEG data kept under project personnel control and may be
  shared only under IRB approved protocol modification and data
  sharing agreements.

* Files in the other directories contain stimuli, data processing
  routines, and deidentified summary measures that may be viewed
  without restriction according to the terms of the licences.
  
# Experimental Stimuli

The EEG stimuli are scraped from the corresponding ERPSS scenario files

# Normative data

* The single trial responses are scraped from lightly reformatted
copies of KAD's original Excel worksheets and saved as rows x columns
text table.

* The response counts, proportions, and related measures are computed
  from the tabular text and stored as a list of YAML maps, one per
  experimental item.

# EEG experiment HDF5 data files

* udck19 Experiment 1 mkdig/eeg_1_arquant are converted from
  copies of binary files in ~kadelong/Exps/arquant

* udck19 Experiment 2: mkdig/eeg_2_arcadj are converted from copies of
  files in ~kadelong/Exps/arcadj

* udck19 Experiment 3: mkdig/eeg_3_yantana are converted from copies of
  files in kadelong/Exps/Yantana

# Raw and single trial epochs

> All raw and single trial epochs are time-locked to the *ARTICLE*

This means the noun is the following word in all eeg_1 (arquant) and
eeg_3 (yantana) epochs.

In eeg_2 (arcadj) the noun is the following word for trials with where 
noun event codes end in 1, 2, 3, 4, 5 and have item_id of the form

	iNNN_2_a__NA_noun
	iNNN_2_an_NA_noun

The noun is two-words downstream for the article+adjective+noun trials
where noun event codes end in 7, 8 and
item_id is of the form  

	iNNN_2_a__adjective_noun
	iNNN_2_an_adjective_noun

	
