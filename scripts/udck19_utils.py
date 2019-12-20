#!/usr/bin/env python
''' helper functions for udck19 data processing '''

import os
import io
from pathlib import Path
import warnings
import re
import hashlib
import datetime
import pprint as pp
import logging
from scipy import stats
import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt


import yaml

# ------------------------------------------------------------
# globals
# ------------------------------------------------------------
import udck19_filenames as fnames

LOGGER = logging.getLogger(__file__)
LOGGER.setLevel(logging.INFO)

# checked against active conda environment's  CONDA_DEFAULT_ENV 
# UDCK19_CONDA_ENV = 'udck19_pnas'
# UDCK19_CONDA_ENV = "udck19_pnas_110819"
UDCK19_CONDA_ENV = "udck19_pnas_121819"

# ------------------------------------------------------------
# Epochs: dimensions (samples x N) before, after EEG screening
# 
# N_EPOCH_SAMPS is after downsampling to 125Hz in udck19_singletrial_wrangling.py
#   1000 ms / 125 == 8 ms sampling period
#   375 samples * 8 ms == 3000 ms
# N_PROEPOCHS_ALL is number of recorded epochs
# N_TRMD_EEG_EPOCHS  is number of epochs after screening EEG in udck19_pipline_1.ipynb

N_EPOCH_SAMPS, N_PREPOCHS_ALL, N_TRMD_EEG_EPOCHS = 375, 13258, 12043

# column label and  HDF5 dataset key for pd.read_hdf(, key=)
EEG_SCREEN_COL = "eeg_screen"

# ------------------------------------------------------------
# index and EEG column names

ALL_EEG_STREAMS = [
    'lle',
    'lhz',
    'MiPf',
    'LLPf',
    'RLPf',
    'LMPf',
    'RMPf',
    'LDFr',
    'RDFr',
    'LLFr',
    'RLFr',
    'LMFr',
    'RMFr',
    'LMCe',
    'RMCe',
    'MiCe',
    'MiPa',
    'LDCe',
    'RDCe',
    'LDPa',
    'RDPa',
    'LMOc',
    'RMOc',
    'LLTe',
    'RLTe',
    'LLOc',
    'RLOc',
    'MiOc',
    'A2',
    'HEOG',
    'rle',
    'rhz',
]


EEG_26_STREAMS = [
    'RLPf', 'RMPf',
    'RDFr', 'RLFr', 'RMFr',
    'RMCe', 'RDCe', 'RDPa',
    'RLTe', 'RLOc',
    'RMOc',
    'MiPf', 'MiCe', 'MiPa', 'MiOc',
    'LLPf', 'LMPf',
    'LDFr', 'LLFr', 'LMFr',
    'LMCe', 'LDCe', 'LDPa',
    'LLTe', 'LLOc',
    'LMOc',
    ]

EEG_MIDLINE = ['MiPf', 'MiCe', 'MiPa', 'MiOc']

# ------------------------------------------------------------
# modeling vars
# ------------------------------------------------------------

MODELING_INDEX_NAMES = [
    'Epoch_idx', 'expt', 'sub_id', 'article_item_id', 'item_id', 'Time'
]

# predictor variables
RHS_VARS = [
    'expt',
    'sub_id',
    'item_id',
    'article_item_id',
    'article_cloze_z',
    'ART_noun_cloze_z',
    'NA_noun_cloze_z',
]


# ------------------------------------------------------------
# global expt names
# ------------------------------------------------------------
EEG_EXPTS = ['EEG_EXP1', 'EEG_EXP2', 'EEG_EXP3']


def get_eeg_expt_specs():
    # global EEG experiment specs
    with open(fnames.EEG_EXPT_SPECS_F) as stream:
        return yaml.load(stream, Loader=yaml.SafeLoader)


EEG_EXPT_SPECS = get_eeg_expt_specs()


# ------------------------------------------------------------
# regression ERP models
#
#     fitting: pipeline_2
#     model comparison: pipeline_3
#     ols influence diagnostics: pipeline_4
#
#     output filenames based on key
# ------------------------------------------------------------

def fit_lmer_formulas(
        epochs_fg, fitr, formulas, h5_f, logger
):
    """ wrapper to run a fitgrid fitter on model sets and save results


    Parameters
    ----------

    epochs_fg : Fitgrid.epochs
        the data to be modeled

    fitr : functools.partial
        fitgrid.utils.summarize wrapper

    formulas: list of strings
        each string is one well-formed lme4 model formula

    h5_f: pathlib.Path
        full path and filename to write the fit summary df

    logger : logging.logger
        where to log the running info

    """
    assert isinstance(formulas, list)
    assert all([isinstance(formula, str) for formula in formulas])
    assert isinstance(h5_f, Path)
    assert isinstance(logger, logging.Logger)

    # h5_f = file_path / f"{file_pfx}{name}.h5"

    logger.info(f"""
    random effects structures
    file: {h5_f}
    {pp.pformat(formulas)}
    """)

    if h5_f.exists():
        logger.info(f'removing previous {h5_f}')
        h5_f.unlink()  # pathlib for delete

    # compute lme4::lmer rerps and save in hdf5 group
    for formula in formulas:
        now = datetime.datetime.now()
        now_str = now.strftime("%d.%b %Y %H:%M:%S")
        logger.info(f"fitting {formula}\n{now_str}")
        fitr(
            epochs_fg, RHS=[formula]
        ).to_hdf(h5_f, key=formula_to_name(formula), mode='a')

        elapsed = datetime.datetime.now() - now
        logger.info(f"Elapsed time: {elapsed}\n")


LMER_MODELS = {

    # experiment as a random effect
    "lmer_acz_ranef": [
        # ------------------------------------------------------------
        # models for random effects selection, start maximal
        # ------------------------------------------------------------
        "article_cloze_z + (article_cloze_z | expt) + (article_cloze_z | sub_id) + (article_cloze_z | article_item_id)",

        # drop 1 slope
        "article_cloze_z + (article_cloze_z | expt) + (article_cloze_z | sub_id) + (1 | article_item_id)",
        "article_cloze_z + (article_cloze_z | expt) + (1 | sub_id) + (article_cloze_z | article_item_id)",
        "article_cloze_z + (1 | expt) + (article_cloze_z | sub_id) + (article_cloze_z | article_item_id)",

        # drop 2 slopes
        "article_cloze_z + (article_cloze_z | expt) + (1 | sub_id) + (1 | article_item_id)",
        "article_cloze_z + (1 | expt) + (article_cloze_z | sub_id) + (1 | article_item_id)",
        "article_cloze_z + (1 | expt) + (1 | sub_id) + (article_cloze_z | article_item_id)",

        # drop 3 slopes
        "article_cloze_z + (1 | expt) + (1 | sub_id) + (1 | article_item_id)",

        # drop 1 random effect
        "article_cloze_z + (1 | sub_id) + (1 | article_item_id)",
        "article_cloze_z + (1 | expt) + (1 | sub_id)",
        "article_cloze_z + (1 | expt) + (1 | article_item_id)",
        # further reduction is not warranted

        # ------------------------------------------------------------
        # for model comparisons in pipeline_3
        # ------------------------------------------------------------
        # the first of thise (KIM+1) has substantial convergence
        # issues, retained for comparison. The second still has some
        # convergence failures but is a reasonable candidate for the
        # keep it maximal (KIM) stopping rule stop

        "(article_cloze_z | expt) + (article_cloze_z | sub_id) + (1 | article_item_id)",
        "(1 | expt) + (article_cloze_z | sub_id) + (1 | article_item_id)",

        # keep it parsimonious (KIP) stopping rule stops here, almost
        # completely free of # convergence failures
        "(1 | expt) + (1 | sub_id) + (1 | article_item_id)",

    ],

    # model comparison experiment as a fixed effect
    "lmer_acz_x_expt_ranef": [

        # subject random slope
        "article_cloze_z + expt + expt:article_cloze_z + (article_cloze_z | sub_id) + (1 | article_item_id)",
        "article_cloze_z + expt + (article_cloze_z | sub_id) + (1 | article_item_id)",
        "expt + (article_cloze_z | sub_id) + (1 | article_item_id)",
        "article_cloze_z + (article_cloze_z | sub_id) + (1 | article_item_id)",
        "(article_cloze_z | sub_id) + (1 | article_item_id)",

        # subject intercept only
        "article_cloze_z + expt + expt:article_cloze_z + (1 | sub_id) + (1 | article_item_id)",
        "article_cloze_z + expt + (1 | sub_id) + (1 | article_item_id)",
        "expt + (1 | sub_id) + (1 | article_item_id)",
        "article_cloze_z + (1 | sub_id) + (1 | article_item_id)",
        "(1 | sub_id) + (1 | article_item_id)",

    ]

}

# Note: no experiment predictor, for use with eeg_1, eeg_2, eeg_3 separately
LMER_MODELS_BY_EXPT = [
        'article_cloze_z + (article_cloze_z | sub_id) + (1 | article_item_id)',
        '(article_cloze_z | sub_id) + (1 | article_item_id)',
        'article_cloze_z + (1 | sub_id) + (1 | article_item_id)',
        '(1 | sub_id) + (1 | article_item_id)',
]


# ------------------------------------------------------------
# general purpose
# ------------------------------------------------------------
def get_udck19_logger(logger_name):

    logr = logging.getLogger(logger_name)
    logr.setLevel(logging.DEBUG)

    log_sh = logging.StreamHandler()  # stream=sys.stdout)
    log_sh.setLevel(logging.DEBUG)

    log_fh = logging.FileHandler(logger_name + '.log', mode='w')
    log_fh.setLevel(logging.DEBUG)

    log_formatter = logging.Formatter(
        "{name}:{levelname}:{message}", style='{'
    )
    log_fh.setFormatter(log_formatter)
    log_sh.setFormatter(log_formatter)
    logr.addHandler(log_fh)
    logr.addHandler(log_sh)

    return(logr)


def check_ENV():
    if not (
            'CONDA_DEFAULT_ENV' in os.environ.keys()
            and os.environ['CONDA_DEFAULT_ENV'] == UDCK19_CONDA_ENV
    ):
        msg = f"activate conda env {UDCK19_CONDA_ENV} before running {__file__}"
        raise OSError(msg)


def parse_item_id(item_id, as_type='strings'):
    """ returns item number, n_iNNN_X_Y_Z_W """
    val_types = dict(
        stim=int,
        n_sent=int,
        article=str,
        adj=str,
        noun=str)

    patt = (r'i(?P<stim>\d{3})_'
            r'(?P<n_sent>(1|2))_'
            r'(?P<article>(NA|a|an))_{1,2}'
            r'(?P<adjective>(NA|\S+))_'
            r'(?P<noun>(NA|\S+))')

    match = re.match(patt, item_id)
    if as_type == 'strings':
        result = match.groupdict()
    elif as_type == 'values':
        # coerce the groupdict value strings to their data type
        result = dict()
        for key, val in match.groupdict().items():
            if val != 'NA':
                result.update({key: val_types[key](val)})
            else:
                result.update({key: None})
    else:
        msg = 'bad as_type argument: ' + as_type
        raise ValueError(msg)
    return result


def parse_item_id_test():
    """ smoke test item_id parser """
    for as_type in ['strings', 'values']:

        parse_item_id('i001_1_NA_NA_NA', as_type=as_type)
        parse_item_id('i001_1_a__NA_NA', as_type=as_type)
        parse_item_id('i001_1_an_NA_NA', as_type=as_type)
        parse_item_id('i001_2_NA_NA_NA', as_type=as_type)
        parse_item_id('i001_2_a__NA_NA', as_type=as_type)
        parse_item_id('i001_2_an_NA_NA', as_type=as_type)
        return None


parse_item_id_test()


def sanitize_string(in_str):
    """sanitize and standardize string data, e.g., norm reponses"""

    out_str = in_str

    # convert missing nans and numeric responses to strings,
    if pd.isnull(out_str):
        out_str = 'NA'  # np.nan -> NA

    try:
        out_str = re.sub(r'`', '\'', out_str)  # left-quote -> apostrophe
        out_str = re.sub(r'’', '\'', out_str)  # right-quote -> apostrophe
        out_str = re.sub(r'“', '"', out_str)  # left double quote -> "
        out_str = re.sub(r'”', '"', out_str)  # right double quote -> "
        out_str = re.sub(r'^\s+', '', out_str)  # strip leading
        out_str = re.sub(r'\s+$', '', out_str)  # strip trailing
        out_str = re.sub(r'\s+', ' ', out_str)  # multiple -> singleton
        out_str = re.sub(r'\t', ' ', out_str)  # strip tabs
        out_str = re.sub(r'\\', '', out_str)  # strip backslash
    except Exception as fail:
        LOGGER.debug(f"{fail} cannot string_sanitize", in_str, type(in_str))
        raise
    return out_str


def code_to_idx(code):
    # strip leading, trailing digit of 5 digit EEG log evcode, returns stim_idx
    return int(code/10) % 1000


def getmd5(in_str):
    ''' returns md5 hexdigest of in_str '''

    md5 = hashlib.md5()
    md5.update(bytes(in_str, 'utf8'))
    return md5.hexdigest()


# ------------------------------------------------------------
# norming utilities
# ------------------------------------------------------------

def load_yaml_norms(norm_measures_yaml_f):
    """convert YAML map to pd.DataFrame"""
    with open(norm_measures_yaml_f) as stream:
        norms_df = pd.DataFrame.from_dict(
            yaml.load(stream, Loader=yaml.SafeLoader), orient='columns'
        )
    return norms_df


# raw response data utilities
def get_initial_character_class(string):
    """ identify first phonemen of in_str; return vowel, consonent, other, NA

    Parameters
    ----------
    string : str

    Returns
    -------
    first_letter_class : str
      values are 'vowel', 'consonant', 'numeral'

    Raises
    ------
    ValueError
       if string == 'NA', 'na', or first character isn't matched

    """

    # NOTE: initial character class based on orthographic character,
    # not phonetic sound
    assert isinstance(string, str) and len(string) > 0
    if string in ['NA', 'na', 'None']:
        msg = f'cannot get character class of missing data {string}'
        raise ValueError(msg)

    vowels = r'^[aeiouAEIOU]'
    consonants = r'^[bcdfghjklmnpqrstvxzwyBCDFGHJKLMNPQRSTVXZWY]'
    numerals = r'^[0-9]'
    initial_character = None
    if re.match(vowels, string) is not None:
        initial_character = 'vowel'
    elif re.match(consonants, string) is not None:
        initial_character = 'consonant'
    elif re.match(numerals, string) is not None:
        initial_character = 'numeral'

        msg = 'cannot determine initial character class of {0}'.format(string)
        raise ValueError(msg)
    return initial_character


def get_token_attr(tok, attr):
    """wrapper to scrape spacy token attributes with a bit of handling for missing
    values and token.is_oov to simplify response counting later.

    Parameters
    ----------
    tok : spacy.token
        item in a spacy doc, e.g., as returned by the nlp tokenizer

    attr : str
       named attribute, e.g., orth_, lemma_, pos_ or 'XYZ_is_OOV' or 'None'

    Returns
    -------
    result : str
       one of these
       - 'None' if tok is None
       - tok.orth + '_IS_OOV' if tok.is_oov is True
       - 'NA' if value of the tok.attr is 'NA' or 'na'
       - else, lower case value of the tok.attr, e.g., 'apple', 'Oreo'->'oreo'
    """

    na_vals = ['NA', 'na']  # , 'nan', 'other']

    # handle data fails first
    if tok is None:
        result = str(None)
    elif tok.is_oov:
        result = str(tok)+'_IS_OOV'
    else:
        # handle attributes
        attr_val = getattr(tok, attr)
        if attr_val is None:
            result = str(None)
        elif attr_val in na_vals:
            # responses coded 'NA' are orth_ 'NA' but lemma_ 'na',
            # make both consistent uppercase 'NA'here for consistent
            # missing response counting later
            result = 'NA'
        else:
            result = str(attr_val)

            # apparently at some point between 2.0 and 2.2 spacy
            # decided to stop normalizing lemmas to lower case,
            # but leave -PRON- and the like ...
            if attr == "lemma_" and not re.match(r"^-[A-Z]+-$", result):
                result = result.lower()

    return result


# ------------------------------------------------------------
# Artifact tagging in epochs data frames
# ------------------------------------------------------------
def encode_eeg_fail(fail_mask):
    # boolean row vector of len == number of channel columns -> int
    return np.sum(
        [(2 ** idx) for idx, bit in enumerate(fail_mask) if bit == 1]
    ).astype(int)


def decode_eeg_fail(reject_code):
    # int -> list of channel column indexes
    byte_one = np.uint8(1)
    return [
        i
        for i in range(8 * np.min_scalar_type(reject_code).itemsize)
        if (reject_code >> i) & byte_one
    ]


def test_encode_decode():
    # 8 "epochs", 1 in column j indicates channel j failed the test
    fail_masks = [
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [1, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 1],
    ]

    # gold standard for decoding bad channel lists
    standard = [
        [],
        [0],
        [1],
        [0, 1],
        [2],
        [0, 2],
        [1, 2],
        [0, 1, 2],
        [3],
        [0, 3],
        [1, 3],
        [0, 1, 3],
        [2, 3],
        [0, 2, 3],
        [1, 2, 3]
    ]

    assert all(
        [
           encode_eeg_fail(fail_mask) == i
           for i, fail_mask in enumerate(fail_masks)
        ]
    )
    assert standard == [decode_eeg_fail(c) for c in range(15)]


test_encode_decode()  # execute file to run test


# ------------------------------------------------------------
# single-trial epoch helpers
# ------------------------------------------------------------
def check_epochs_shape(epochs_df):
    # total number of rows in epochs dataframe == index epochs x timestamps
    n_times = len(epochs_df.index.unique('Time'))
    n_eps = len(epochs_df.index.unique('Epoch_idx'))
    assert (n_times * n_eps) == len(epochs_df)
    return n_times, n_eps


def standardize(ep_df, col_names):
    # add a column of centered, scaled data from col_name as col_name_z
    #
    # using pandas b.c. scipy.stats.zscore doesn't handle the cloze NaNs
    # for adj stim in eeg_2 arcadj
    #
    # col_names = [
    #     'article_cloze', 'ART_noun_cloze', 'NA_noun_cloze'
    # ]
    # adds
    #   'article_cloze_z', 'ART_noun_cloze_z', 'NA_noun_cloze_z'
    #
    # raises RuntimeError if col_z exists

    assert isinstance(ep_df, pd.DataFrame)
    assert all((col in ep_df.columns for col in col_names))
    means_sds = dict()

    for col in col_names:
        col_z = col + "_z"
        if col_z in ep_df.columns:
            raise RuntimeError(f"{col_z} already exists, delete it first")

        col_mean = ep_df[col].mean(skipna=True)

        # ddof for divisor, scipy default is N-0, pandas is N-1
        col_std = ep_df[col].std(skipna=True, ddof=0)

        ep_df[col_z] = (ep_df[col] - col_mean) / col_std

        if not ep_df[col].hasnans:
            assert np.allclose(stats.zscore(ep_df[col]), ep_df[col_z])

        # track for reverting to original scale
        means_sds.update(
            {
                col: {
                    'mean': col_mean,
                    'sd': col_std,
                }
            }
        )
    return ep_df, means_sds


# ------------------------------------------------------------
# udck19 specific fitgrid rERP helpers
# ------------------------------------------------------------

def formula_to_name(formula):
    """convert special chars in formulas to legal column/file names"""
    char_map = {
        r"\+": "_P_",  # Plus
        r"\:": "_C_",  # Colon
        r"\*": "_S_",  # Star
        r"\|": "_B_",  # Bar
        r"\(": "L_",   # Left paren
        r"\)": "_R",   # Right paren
    }

    name = re.sub(r"\s+", "", formula).strip()
    for char, string in char_map.items():
        name = re.sub(char, string, name)
    return name


# read a summaries file to dataframe in models order
def read_fg_summaries_hdf(h5_f, models):
    return pd.concat(
        [pd.read_hdf(h5_f, key=formula_to_name(model)) for model in models]
    )


# ------------------------------------------------------------
# jupyter noteboook figure helpers
# ------------------------------------------------------------

# default upper left
FIG_TAG_SPECS = {
    'x': 0.0,
    'y': 1.0,
    'fontsize': 'x-large',
    'fontweight': 'bold',
    'horizontalalignment': 'left',
    'verticalalignment': 'bottom'
}


def panel_from_idx(idx):
    # 0, 1, 2 ,... 26, 27, ... map to letters A, B, ... AA, BB, ..."""

    letter = ''.join([chr((idx % 26) + 65)] * (1 + int(idx / 26.0)))
    return letter


# note auto-increment figure counter
def udck19_figsave(f, fname, fig_n, formats=['pdf'], kws={}):
    for fmt in formats:
        f.savefig(
            fnames.FIG_DIR / (re.sub(r"\s+", "_", fname) + f".{fmt}"),
            format=fmt,
            facecolor='white',
            bbox_inches='tight',
            **kws)
    return fig_n + 1


# ------------------------------------------------------------
# Multichannel matplotlib layout
# ------------------------------------------------------------
# gridded for traces
MPL_32_CHAN = {
    'w': .125,
    'h': .075,
    'chanlocs': {
        'cal': (0.0625, 0.2),  # LLTe, LMOc
        'lle': (0.25, 0.85),
        'rle': (0.625, 0.85),
        'lhz': (0.0625, 0.85),
        'rhz': (0.8125, 0.85),
        'MiPf': (0.4375, 0.725),
        'MiCe': (0.4375, 0.425),
        'MiPa': (0.4375, 0.275),
        'MiOc': (0.4375, 0.125),
        'LLPf': (0.1875, 0.725),
        'RLPf': (0.6875, 0.725),
        'LMPf': (0.3125, 0.65),
        'RMPf': (0.5625, 0.65),
        'LLFr': (0.0625, 0.5),
        'RLFr': (0.8125, 0.5),
        'LMFr': (0.3125, 0.5),
        'RMFr': (0.5625, 0.5),
        'LDFr': (0.1875, 0.575),
        'RDFr': (0.6875, 0.575),
        'LDCe': (0.1875, 0.425),
        'RDCe': (0.6875, 0.425),
        'LLTe': (0.0625, 0.35),
        'RLTe': (0.8125, 0.35),
        'LMCe': (0.3125, 0.35),
        'RMCe': (0.5625, 0.35),
        'LMOc': (0.3125, 0.2),
        'RMOc': (0.5625, 0.2),
        'LDPa': (0.1875, 0.275),
        'RDPa': (0.6875, 0.275),
        'LLOc': (0.1875, 0.125),
        'RLOc': (0.6875, 0.125),
        'A2': (0.8125, 0.2)
    }
}

# semi-topographic
MPL_32_HEAD = {
    'w': .125,
    'h': .075,
    'chanlocs': {
        'cal': (0.0625, 0.2),  # LLTe, LMOc
        'lle': (0.25, 0.85),
        'rle': (0.625, 0.85),
        'lhz': (0.0625, 0.85),
        'rhz': (0.8125, 0.85),
        'MiPf': (0.4375, 0.725),
        'MiCe': (0.4375, 0.425),
        'MiPa': (0.4375, 0.275),
        'MiOc': (0.4375, 0.125),
        'LLPf': (0.1875, 0.725),
        'RLPf': (0.6875, 0.725),
        'LMPf': (0.3125, 0.65),
        'RMPf': (0.5625, 0.65),
        'LLFr': (0.0625, 0.5),
        'RLFr': (0.8125, 0.5),
        'LMFr': (0.3125, 0.5),
        'RMFr': (0.5625, 0.5),
        'LDFr': (0.1875, 0.575),
        'RDFr': (0.6875, 0.575),
        'LDCe': (0.1875, 0.425),
        'RDCe': (0.6875, 0.425),
        'LLTe': (0.0625, 0.35),
        'RLTe': (0.8125, 0.35),
        'LMCe': (0.3125, 0.35),
        'RMCe': (0.5625, 0.35),
        'LMOc': (0.3125, 0.2),
        'RMOc': (0.5625, 0.2),
        'LDPa': (0.1875, 0.275),
        'RDPa': (0.6875, 0.275),
        'LLOc': (0.1875, 0.125),
        'RLOc': (0.6875, 0.125),
        'A2': (0.8125, 0.2)
    }
}

MPL_MIDLINE = {
    'w': .75,
    'h': .2,
    'chanlocs': {
        'MiPf': (0.1, 0.7),
        'MiCe': (0.1, 0.5),
        'MiPa': (0.1, 0.3),
        'MiOc': (0.1, 0.1),
        'cal': (0.1, 0.1),
    }
}

# ------------------------------------------------------------
# spherical coordinate head layout
# ------------------------------------------------------------
# usual electrode locations plus reference, EOG, mastoids, ground
sph26_txt = io.StringIO("""
channel  phi   theta  ch_type
MiPf  90.0   90.0   eeg
LLPf  90.0  126.0   eeg
LLFr  90.0  162.0   eeg
LLTe  90.0  198.0   eeg
LLOc  90.0  234.0   eeg
MiOc  90.0  270.0   eeg
RLOc  90.0  306.0   eeg
RLTe  90.0  342.0   eeg
RLFr  90.0   18.0   eeg
RLPf  90.0   54.0   eeg
LMPf  59.0  108.0   eeg
LDFr  59.0  144.0   eeg
LDCe  59.0  180.0   eeg
LDPa  59.0  216.0   eeg
LMOc  59.0  252.0   eeg
RMOc  59.0  288.0   eeg
RDPa  59.0  324.0   eeg
RDCe  59.0    0.0   eeg
RDFr  59.0   36.0   eeg
RMPf  59.0   72.0   eeg
LMFr  26.0  126.0   eeg
LMCe  26.0  198.0   eeg
MiPa  26.0  270.0   eeg
RMCe  26.0  342.0   eeg
RMFr  26.0   54.0   eeg
MiCe   0.0    0.0   eeg

A1    130.0  205.0  ref
A2    130.0  335.0  ref

lle   140.0  120.0  eog
rle   140.0   60.0  eog

lhz   108.0  130.0  eog
rhz   108.0   50.0  eog

nasion 108.0   90.0  fid
lpa    108.0  180.0  fid
rpa    108.0    0.0  fid

gnd     72.0    90.0 gnd

""")


def sph2cart(row):
    label, phi, theta, r, ch_type = [*row]

    deg2rad = 2 * np.pi / 360
    phi *= deg2rad
    theta *= deg2rad

    x = np.cos(theta) * np.sin(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(phi)

    # normalized to circle of radius 1
    lambert_x = x * np.sqrt(1 / (1 + z))
    lambert_y = y * np.sqrt(1 / (1 + z))

    row['x'], row['y'], row['z'] = x, y, z
    row['x_lambert'], row['y_lambert'] = lambert_x, lambert_y

    return row


cap26 = pd.read_csv(sph26_txt, sep=r"\s+", engine='python')
sph26_txt.close()
cap26['r'] = 5  # radius
CAP26_DF = cap26.apply(lambda x: sph2cart(x), 1).set_index('channel')


# ------------------------------------------------------------
# multichannel time series plotter
# ------------------------------------------------------------
def plotchans(
    rerps,
    beta_kws,
    layout,
    se_ci=None,
    style='default',
    plotchans_style=None,
):

    # some default axis specs
    if plotchans_style is None:

        plotchans_style = {

            'axes.facecolor': 'none',
            'axes.edgecolor': 'gray',
            'axes.spines.top': False,
            'axes.spines.right': False,

            'xtick.direction': 'inout',
            'xtick.major.size': 10,
            'xtick.color': 'gray',

            'ytick.direction': 'in',
            'ytick.major.size': 5,
            'ytick.color': 'gray',

        }

    # channel label locations
    # x0, y0, verticalalignment, horizontalalignment
    chlab_locs = {
        'west': (-0.05, 0.5, 'center', 'right'),
        'east': (1.05, 0.5, 'center', 'left'),
        'north': (0.5, 1.05, 'bottom', 'center'),
        'south': (0.5, -0.05, 'top', 'center'),
        'northwest': (-0.05, 1.05, 'bottom', 'right'),
    }

    figs = []
    for beta, beta_data in rerps.groupby('beta'):

        with plt.style.context([style, plotchans_style]):

            try:
                fc = mpl.style.library[style]['axes.facecolor']
            except Exception:
                fc = None

            f = plt.figure(figsize=(12, 10), facecolor=fc)

            fig_legend_handles = []  # collect lines and model labels
            fig_legend_labels = []

            # plot the beta trace this channel each model
            for axi, chan in enumerate(rerps.columns):

                # ------------------------------------------------------------
                # pin all axes to the same size and and scales as the first
                # channel labels
                # ------------------------------------------------------------
                bbox = [*layout['chanlocs'][chan], layout['w'], layout['h']]
                if axi == 0:
                    ax = f.add_axes(bbox, label=chan)
                    ax0 = ax

                else:
                    ax = f.add_axes(
                        bbox,
                        label=chan,
                        sharex=ax0,
                        sharey=ax0,
                    )

                # ------------------------------------------------------------
                # channel labels
                # ------------------------------------------------------------
                if "chan_label" in beta_kws[beta].keys():
                    try:
                        clx0, cly0, clva, clha, = chlab_locs[
                            beta_kws[beta]['chan_label']
                        ]
                    except Exception:
                        warnings.warn(
                            f"Bad channel label location in {beta_kws[beta]}"
                        )
                        chx0, cly0, clva, clha = chlab_locs['northwest']

                    ax.text(
                        clx0,
                        cly0,
                        chan,
                        verticalalignment=clva,
                        horizontalalignment=clha,
                        transform=ax.transAxes,
                        fontsize='large'
                    )

                # ------------------------------------------------------------
                # plot the traces
                # ------------------------------------------------------------
                for model, model_data in beta_data.groupby('model'):

                    chan_data = (
                        model_data.query('key == "Estimate"')[chan].to_frame()
                    )

                    line = ax.plot(
                        chan_data.reset_index('Time')['Time'],
                        chan_data[chan],
                        clip_on=False,
                        markevery=1
                    )

                    # hack to build the figure legend from axis 0
                    if axi == 0:
                        fig_legend_handles.append(line[0])
                        fig_legend_labels.append(model)

                    # ----------------------------------------
                    # uncertainty bands
                    # ----------------------------------------
                    band_label = ""
                    if se_ci in ["SE", "CI"]:
                        # fill between y0, y1
                        if se_ci == "SE":
                            chan_data['SE'] = model_data.query(
                                'key == "SE"'
                            )[chan].to_numpy()
                            chan_data['y1'] = chan_data[chan] - chan_data['SE']
                            chan_data['y2'] = chan_data[chan] + chan_data['SE']
                            band_label = f"(+/- {se_ci})"

                        if se_ci == "CI":
                            ci_lo, ci_hi = sorted(
                                [
                                    key
                                    for key in model_data.index.unique('key')
                                    if "_ci" in key
                                ]
                            )
                            chan_data['y1'] = model_data.query(
                                'key == @ci_lo'
                            )[chan].to_numpy()
                            chan_data['y2'] = model_data.query(
                                'key == @ci_hi'
                            )[chan].to_numpy()
                            band_label = (
                                f"(CI {ci_lo.replace('_ci','')}"
                                f" - {ci_hi.replace('_ci','')})"
                            )

                        ax.fill_between(
                            x=chan_data.reset_index('Time')['Time'],
                            y1=chan_data['y1'],
                            y2=chan_data['y2'],
                            alpha=0.25,
                            clip_on=False,
                        )

            # -----------------------
            # figure tuning ...
            # -----------------------
            # add and format calibration axis)

            f.suptitle(
                f"{beta} {band_label}",
                x=0,
                y=1.0,
                fontsize='x-large',
                fontweight='bold'
            )

            # add calibration axis
            ax_cal = f.add_axes(
                [*layout['chanlocs']['cal'], layout['w'], layout['h']],
                label='_cal',
                sharex=ax0,
                sharey=ax0,
                zorder=0.1,  # under plot
            )
            ax_cal.dataLim = ax0.dataLim

            # tune axis decorations
            axs = f.get_axes()
            for ax in axs:
                ax.grid(False)

                # adjust channel axes for bottom margin
                axpos = list(ax.get_position().bounds)
                axpos[1] += beta_kws[beta]['margins']['bottom']
                ax.set_position(axpos)

                # update the beta specific params
                ax.set(**beta_kws[beta]['axes'])

                # x-axis timeline
                ax.spines['bottom'].set_position('zero')
                ax.spines['bottom'].set_bounds(
                    ax0.dataLim.x0, ax0.dataLim.x1
                )

                # y-axis cal bar
                ax.set_yticks(beta_kws[beta]['cal']['yticks'])
                ax.spines['left'].set_bounds(
                    *beta_kws[beta]['cal']['yticks']
                )
                ax.spines['left'].set_position(('data', ax.dataLim.x0))

                # axis-specific formatting
                if ax is not ax_cal:
                    ax.tick_params(labelcolor='none')

                if ax is ax_cal:
                    for spine in ['bottom', 'left']:
                        ax.spines[spine].set_color('black')
                    ax.tick_params(labelcolor='black')

                    ax_cal.set_ylabel(
                        beta_kws[beta]['cal']['ylabel'],
                        rotation='horizontal',
                        horizontalalignment='right',
                        verticalalignment='center',
                    )

            f.legend(
                handles=fig_legend_handles,
                labels=fig_legend_labels,
                loc='upper center',
                bbox_to_anchor=(0.5, beta_kws[beta]['margins']['bottom'])
            )
            figs.append({'beta': beta, 'fig': f})
    return figs


# pairwise fit_meas Stoermann plot
def plot_pairwise_delta(
        summary_df, fit_measure, t0, t1, order=1, ax_kws={}, fig_kws={}
):
    try:
        model0, model1 = summary_df.index.unique('model')
    except Exception:
        raise ValueError('summary_df must have exactly two models')

    # model-wise measure is same at all betas, first beta is as good as any
    beta0 = summary_df.index.unique('beta')[0]
    qstr = (
        f"beta=='{beta0}' and key=='{fit_measure}'"
        f" and Time >= {t0} and Time <= {t1}"
    )
    lm_fit_meas_dfs = [
        summary_df.query(
            qstr + " and model==@model"
        ).reset_index('model', drop=True)
        for model in [model0, model1]
    ]

    # optionally flip order of subtraction
    if order == 1:
        delta_fit_meas = lm_fit_meas_dfs[1] - lm_fit_meas_dfs[0]
        title_str = f"{model1} -\n{model0}"
    elif order == -1:
        delta_fit_meas = lm_fit_meas_dfs[0] - lm_fit_meas_dfs[1]
        title_str = f"{model0} -\n{model1}"
    else:
        raise ValueError('order must be 1 or -1')

    f, ax = plt.subplots(1, 1, **fig_kws)

    ax = delta_fit_meas.reset_index('Time').plot(x='Time', ax=ax)
    ax.set(**ax_kws)
    ax.set_title(title_str)

    ax.axhline(y=0, lw=1, color='black')
    for delta in [2, 4, 7, 10]:
        (lw, ls) = (2, 'dashed') if delta == 2 else (1, 'dotted')
        ax.axhline(y=delta, color='gray', lw=lw, ls=ls)
        ax.axhline(y=-delta, color='gray', lw=lw, ls=ls)
    ax.legend(
        loc='center right',
        bbox_to_anchor=(-0.1, 0.5)
    )
    return f, ax



def get_symdiv_cmap_norm(cmap, lower, upper, n_shades):

    # n_shades in each half of the symmetric color bar
    if lower != -upper:
        msg = f"upper and lower bounds must be symmetric around zero"
        raise ValueError(msg)

    # odd ncolors always odd so we have a white band at 0
    n_colors = (2 * n_shades) + 1
    bounds = np.linspace(lower, upper, n_colors + 1)
    norm = mpl.colors.BoundaryNorm(bounds, n_colors)

    # get blue-white-red divergent colormap
    cmap = mpl.cm.get_cmap(cmap, n_colors)
    return cmap, norm


