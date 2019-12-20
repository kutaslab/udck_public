#!/usr/bin/env python
"""set the global experiment filenames, guard input files with md5 checksum
"""

from pathlib import Path
import hashlib
import warnings

# so we can run from other directories
parent = Path(__file__).resolve().parent
grand_parent = parent.parent
UDCK19_DIR = grand_parent.parent # top level

# ------------------------------------------------------------
# stimulus norming data files
# ------------------------------------------------------------

NORM_EXPT_SPECS_F = parent / "norm_expt_specs.yml"
NORM_EXPT_SPECS_MD5 = "d96f01a778f2a011b481eac0d4f65333"

# tab-separated table of single trial norming responses
FINAL_SCREENED_NORMS_F = (
    grand_parent / "data/norms/udck19_norms_astoermann_screened.txt"
)
FINAL_SCREENED_NORMS_MD5 = "048c60123569686899e28d47ec301870"

# main YAML doc with norming counts and measures
NORM_MEASURES_YAML_F = parent / "../measures/udck19_norms_item_counts.yml"
NORM_MEASURES_YAML_MD5 = "12092b95dc033a1e6f10f2dc51f0bd74"

norm_f_md5s = [
    (NORM_EXPT_SPECS_F, NORM_EXPT_SPECS_MD5),
    (FINAL_SCREENED_NORMS_F, FINAL_SCREENED_NORMS_MD5),
    (NORM_MEASURES_YAML_F, NORM_MEASURES_YAML_MD5),
]
for file_, md5_ in norm_f_md5s:
    with open(file_) as stream:
        md5 = hashlib.md5(stream.read().encode("utf8")).hexdigest()
        if md5_ != md5:
            raise Exception(f"MD5 has changed {file_} was {md5} now {md5_}")

# ------------------------------------------------------------
# scenario files and stim
# ------------------------------------------------------------
# for .scn file scraping
STIM_PATH = grand_parent / "stim"
EEG_STIM_F = STIM_PATH / "eeg_expt_stim.txt"
EEG_STIM_MD5 = "cf7a53ec903769fa68028dad8a8af22b"
with open(EEG_STIM_F) as stream:
    assert (
        EEG_STIM_MD5 == hashlib.md5(stream.read().encode("utf8")).hexdigest()
    )

# ------------------------------------------------------------
# Participant info
# ------------------------------------------------------------

SUB_INFO_PATH = grand_parent / "data"
SUB_INFO_F = SUB_INFO_PATH / "arquant_yantana_arcadj_sub_info.csv"
SUB_INFO_MD5 = "fba1c26467e96bb6ceeabca04a24aed2"
with open(SUB_INFO_F) as stream:
    assert(
        SUB_INFO_MD5 == hashlib.md5(stream.read().encode("utf8")).hexdigest()
    )

# ------------------------------------------------------------
# EEG data HDF5 input files are pinned in udck19/legacy2dif/mkh5
# to freeze checksums. 
# ------------------------------------------------------------

# input directories
EEG_EXPT_SPECS_F = Path(parent, "eeg_expt_specs.yml")
EEG_SOURCE_H5_DIR =  UDCK19_DIR / "legacy2dif/mkh5"
EEG_H5_DIR = Path(grand_parent, "data/eeg")

# output directories
CODE_MAP_DIR = Path(grand_parent, "measures")
EEG_EPOCHS_DIR = Path(grand_parent, "data", "epochs")  # single trial epochs


EEG_MEASURES_DIR = Path(grand_parent, "measures")  # N400 measures

# if needed for debugging ...
# MASTER_CODE_MAP_F = "master_code_map.txt"
# MASTER_CODE_MAP_PATH = Path(CODE_MAP_DIR, MASTER_CODE_MAP_F)

# define hdf5 EEG filenames to read and output filenames to write.
# Note: the pinned source EEG files are copied to h5_f during
# single trial wrangling for reproducibility.

EEG_EXPT_FILES = {
    "eeg_1": {
        "source_md5": "1e1c872cf1110f140f4e6e4ad9740284",
        "source_h5_f": EEG_SOURCE_H5_DIR / "arquant.h5",
        "h5_f": EEG_H5_DIR / "arquant.h5",
    },
    "eeg_2": {
        "source_md5": "93af6012c1b323e6694d9b0a2ad4367f",
        "source_h5_f": EEG_SOURCE_H5_DIR / "arcadj.h5",
        "h5_f": EEG_H5_DIR / "arcadj.h5",
    },
    "eeg_3": {
        "source_md5": "c90d069b47e137d72f21e98ade4362a7",
        "source_h5_f": EEG_SOURCE_H5_DIR / "yantana.h5",
        "h5_f": EEG_H5_DIR / "yantana.h5",
    },
}


for expt, files in EEG_EXPT_FILES.items():
    with open(files['source_h5_f'], 'rb') as stream:
        md5 = hashlib.md5(stream.read()).hexdigest()
        if not md5 == files["source_md5"]:
            msg = (
                f"{files['source_h5_f']} md5 {md5}"
                f" does not match record in {__file__}"
            )
            print(f"{expt} {md5}")
            raise ValueError(msg)

# ------------------------------------------------------------
# Single trial epochs
# ------------------------------------------------------------

# recorded epochs: segmented EEG data snipped as is from .h5
for expt in ["eeg_1", "eeg_2", "eeg_3"]:
    expt_files = {
        "code_map_f": CODE_MAP_DIR / f"{expt}_code_map.txt",
        "recorded_epochs_f": EEG_EPOCHS_DIR / f"{expt}_recorded_epochs.h5",
    }
    EEG_EXPT_FILES[expt].update(expt_files)


# ------------------------------------------------------------
# Analysis pipeline
# ------------------------------------------------------------

# Single-trial epochs

# prepochs: prepared by single trial wrangling for modeling all 3 datasets
PREPOCHS_ALL_F = EEG_EPOCHS_DIR / "prepochs_all.h5"

# after excluding EEG artifacts in pipeline_1
PREPOCHS_TRMD_EEG_F = EEG_EPOCHS_DIR / "prepochs_trimd_eeg.h5"

# after further excluding possible influential datapoints in pipeline_4
PREPOCHS_TRMD_EEG_COOKSD_F = EEG_EPOCHS_DIR / "prepochs_trimd_eeg_cooksD.h5"


# ------------------------------------------------------------
# single trial modeling: rERPs
# ------------------------------------------------------------

EEG_MODELING_DIR = EEG_MEASURES_DIR / "modeling"  # rERPs

# ------------------------------------------------------------
# figures
# ------------------------------------------------------------
FIG_DIR = grand_parent / "figs"
