#!/usr/bin/env python
"""udck19 single trial data wrangling

Starts with calibrated EEG HDF5 data and norm measures and produces
single trial epochs tagged with norm measures, baslined and filtered
for downsampling.

See if __name__=='__main__' below for the sequence and the individual
function docstrings for the details.

"""

import re
from pathlib import Path
import shutil
import hashlib
import pprint as pp

import numpy as np
import pandas as pd
import bottleneck as bn
from scipy.signal import kaiserord, lfilter, firwin, freqz

import spacy

import h5py
from mkpy import mkh5

# import file names and ensure data are unsullied
from udck19_filenames import (
    STIM_PATH, EEG_STIM_F, NORM_MEASURES_YAML_F, EEG_EXPT_FILES, PREPOCHS_ALL_F
)

# import udck19_utils
from udck19_utils import (
    get_udck19_logger,
    check_ENV,
    sanitize_string,
    code_to_idx,
    parse_item_id,
    load_yaml_norms,
    get_eeg_expt_specs,
    ALL_EEG_STREAMS,
    check_epochs_shape,
    encode_eeg_fail
)

from udck19_norming import count_dict_to_df

# ------------------------------------------------------------
# set up
# ------------------------------------------------------------
# check conda env is active
check_ENV()
LOGGER = get_udck19_logger(__file__)

# set up spacy
NLP = spacy.load('en_core_web_lg')
if NLP.meta is None or NLP.meta['name'] != 'core_web_lg':
    msg = "spacy.load('en_core_web_lg') failed to load core_web_lg\n"
    msg += "make sure to: source activate udck19"
    raise RuntimeError(msg)
LOGGER.info(NLP.meta)

# ------------------------------------------------------------
# udck19 SINGLE TRIAL PARAMS
# ------------------------------------------------------------

# expand the epoch interval so filter edge artifacts can be trimmed,
# by halfwidth = (ntaps - 1) / 2
# for 2x decimation. E.g., 
# if filter half width = 35 samples * 4 ms / sample = 140 ms each edge
# RECORDED_EPOCH_START, RECORDED_EPOCH_STOP = -1640, 1644
# if filter half width = 38 samples * 4 ms / sample = 152 ms each edge
RECORDED_EPOCH_START, RECORDED_EPOCH_STOP = -1652, 1656

# integer multiples of downsampled time-points, e.g., 4ms by 2 == 8ms
DOWNSAMPLED_EPOCH_START, DOWNSAMPLED_EPOCH_STOP = -1496, 1496

# center on entire prestimulus interval ... don't include 0
BASELINE_START, BASELINE_STOP = DOWNSAMPLED_EPOCH_START, -1

EEG_EXPT_SPECS = get_eeg_expt_specs()

# epoch params ... recorded = "raw" epochs

INDEX_NAMES = ['Epoch_idx', 'expt', 'sub_id', 'item_id', 'Time']
LOGGER.info(
    f"epochs: {RECORDED_EPOCH_START} article=0 {RECORDED_EPOCH_STOP}: "
    f"index.names: {INDEX_NAMES}"
)

LOGGER.info('ALL_EEG_STREAMS: {0}'.format(ALL_EEG_STREAMS))

# optional: average spatial ROIs as
# key = new data column name, value = channels to average
# ROIS = {
#      'cp_roi': ['LMCe', 'RMCe', 'MiCe', 'MiPa', 'LDCe', 'RDCe'],
# }
ROIS = None
LOGGER.info('ROIS: {0}'.format(ROIS))

# ------------------------------------------------------------
#  debugging ... available columns
#
# 'is_anchor', 'item_id_pfx', 'lemma_ART_noun_anywhere_cloze',
# 'lemma_ART_noun_initial_cloze', 'lemma_NA_noun_anywhere_cloze',
#  'lemma_NA_noun_initial_cloze', 'lemma_context_anywhere_info',
#  'lemma_context_initial_info', 'lemma_modal_anywhere',
#  'lemma_modal_anywhere_cloze', 'lemma_modal_initial',
#  'lemma_modal_initial_character_classes', 'lemma_modal_initial_cloze'
#  'lemma_n_NAs', 'lemma_n_responses', 'lemma_n_strings', 'list_id',
#  'noun', 'noun_code', 'noun_pos', 'orth_ART_noun_anywhere_cloze',
#  'orth_ART_noun_initial_cloze', 'orth_NA_noun_anywhere_cloze',
#  'orth_NA_noun_initial_cloze', 'orth_article_initial_cloze',
#  'orth_context_anywhere_info', 'orth_context_initial_info',
#  'orth_modal_anywhere', 'orth_modal_anywhere_cloze',
#  'orth_modal_initial', 'orth_modal_initial_character_classes',
#  'orth_modal_initial_cloze', 'orth_n_NAs', 'orth_n_responses',
#  'orth_n_strings', 'raw_evcodes', 'regexp', 's1_code', 'stim_idx',
#  'topic_n_NA', 'topic_n_consonants', 'topic_n_vowels', 'anchor_time',
#  'crw_ticks', 'pygarv',
# ------------------------------------------------------------

# re-label obscure system data column of interest
# see epoch export below for usage
COI_LABELS = {
    # from mkpy code tagger
    "expt": "expt",
    "data_group": "sub_id",
    "dblock_path": "h5_dataset",
    "dblock_ticks": "dataset_index",
    "log_evcodes": "event_code",
    "match_code": "regex_match",
    "anchor_code": "regex_anchor",
    "log_flags": "garv_reject",
    # from mkpy code_map
    "item_id": "item_id",
    "article": "article",
    "adjective": "adjective",
    "noun": "noun",
    # manual merged into the code map from the norming
    "orth_article_initial_cloze": "article_cloze",
    "lemma_ART_noun_anywhere_cloze": "ART_noun_cloze",
    "lemma_NA_noun_anywhere_cloze": "NA_noun_cloze",
}
LOGGER.info(COI_LABELS)


# ------------------------------------------------------------
# scenario file functions
# ------------------------------------------------------------
def load_scns(
    expt_id='',
    scn_dir='',
    scn_patt=r'',
    stim_ids=None,
    noun_codes=None,
    lists=None,
    list_ids=None,
    **kwargs,
):
    """parse actual scenario file stim into a single data frame

    Parameters
    ----------
        passed in as **dict, see check_scn for types and usage

    Returns
    -------
    scns_data : pd.DataFrame
        complete list of A/An stim across all .scn files for the expt glob

    Notes
    -----
    """

    # helper
    def item_id_from_row(row):
        """build eeg scenario file item_id with same format as norming stim item_id
        """

        stim_id = 'i{0:03d}'.format(row['stim_idx'])
        n_sentences = '1' if row['s1'] is None else '2'

        article = (
            '#ERROR'
            if row['article'] is None
            else row['article'].ljust(2, '_')
        )

        adjective = 'NA' if row['adjective'] is None else row['adjective']
        noun = '#ERROR' if row['noun'] is None else row['noun']
        item_id = '_'.join([stim_id, n_sentences, article, adjective, noun])
        return item_id

    scn_path = Path(STIM_PATH, scn_dir)

    stim_id_range = range(stim_ids['start'], stim_ids['stop'] + 1)
    list_id_range = range(list_ids['start'], list_ids['stop'] + 1)
    noun_code_range = range(noun_codes['start'], noun_codes['stop'] + 1)

    all_scns = []
    for lst in lists:
        for lst_id in list_id_range:
            scn_glob = scn_path.glob(scn_patt.format(lst, lst_id))
            scn_files = [scnf for scnf in scn_glob]
            assert all([scnf.is_file for scnf in scn_files])

            scn_data = pd.DataFrame()
            data_shape = None

            for scn_file in sorted(scn_files):
                LOGGER.info(f"*** SCANNING SCENARIO FILE {scn_file}")
                with open(scn_file, 'r') as scnf:
                    scn_str = scnf.read()
                    stim_info = scrape_scn(
                        scn_str, stim_id_range, noun_code_range
                    )
                    stim_info.insert(0, 'scn_f', Path(scn_file).name)
                    scn_data = scn_data.append(stim_info)
            scn_data.insert(0, 'expt', expt_id)
            scn_data.insert(1, 'scn_dir', scn_dir)
            scn_data.insert(2, 'list', '{0}{1}'.format(lst, lst_id))
            scn_data.insert(3, 'list_id', f"{expt_id}_{lst}".lower())

            scn_data = scn_data.sort_values('stim_idx')

            # check shape each iteration
            if data_shape is not None:
                assert data_shape == scn_data.shape
            data_shape = scn_data.shape

            all_scns.append(scn_data)

    scns_data = pd.concat(all_scns)

    scns_data['item_id'] = scns_data.apply(
        lambda row: item_id_from_row(row), 1
    )

    scns_data.sort_values(['item_id', 'list'])
    scns_data.set_index('item_id', inplace=True)

    return scns_data


def scrape_scn(scn_str, stim_idx_range, noun_code_range):
    '''extract expt specific arquant, arcadj, yantana a/an stim codes and
       text from scenario files
    '''

    # pattern for soa, dur, ev_code, and between 1 and 3 text= lines
    scraper = re.compile(
        (
            r'\s*(?P<soa>\d+)\s+'
            r'(?P<dur>\d+)\s+'
            r'(?P<ev_code>\d+)'
            r'(?P<stim>(?:\s+?text=.*\n{0,1}){1,3})'
        )
    )

    # these order the DataFrame columns
    stim_cols = [
        'stim_idx',
        'trial',
        's1_code',
        'noun_code',
        'noun_pos',
        'article',
        'adjective',
        'noun',
        's1',
        'rsvp',
    ]

    def init_stim(stim_cols):
        """returns a new empty dict to hold trial info"""
        empty_trial = dict()
        for stim_col in stim_cols:
            empty_trial.update({stim_col: None})
        return empty_trial

    # hard-coded EEG log event code for sentence begin/end delimiter
    # ... brittle
    start_code = 1
    stop_code = 1

    # first pass grab codes & text and + text + text if any
    matches = scraper.findall(scn_str)

    # second pass cleans up multi-line text= ... +
    text_regx = re.compile(r'text=("[^+]*"|[^"]\S+[^"])')
    stims = []
    for match in matches:
        texts = text_regx.findall(match[3])
        texts = [re.sub(r'(^"|"$)', '', line) for line in texts]
        if len(texts) > 0:
            text = sanitize_string(' '.join(texts))
            stim = list(match)
            stim[3] = text
            stims.append(stim)

    # split the stims into *trials* ... complicated a bit by the fact
    # that an arquant trial is a single RSVP sentence while arcadj,
    # yantana trials are S1=context. S2=RSVP trials.

    stim_list = []
    state = 0
    this_stim = init_stim(stim_cols)
    for stim in stims:

        soa = int(stim[0])
        # dur = int(stim[1])
        ev_code = int(stim[2])
        stim_str = stim[3]

        # process start/stop code delimiters, these vary across
        # the expts ... this is hardcoded for these expts, brittle
        if ev_code in [start_code or stop_code] and soa >= 1000:

            # re-init for next stim
            if state == 0:
                this_stim = init_stim(stim_cols)
                state = 1  # == word count

            elif state > 1:

                # else dump current stim and init for next
                if this_stim['noun_code'] in noun_code_range:
                    # print('appending ', this_stim)
                    this_stim['trial'] = len(stim_list) + 1
                    stim_list.append(this_stim)

                # reset
                this_stim = init_stim(stim_cols)
                state = 0  # reset
            else:
                pass

        # set S1 context if any
        elif ev_code > 30000:
            this_stim['s1'] = stim_str
            this_stim['s1_code'] = ev_code

        # load sentence, state counts words
        elif stim_str != '' and (ev_code in stim_idx_range):

            if this_stim['rsvp'] is None:
                this_stim['rsvp'] = stim_str
            else:
                this_stim['rsvp'] += ' ' + stim_str

            # set stim idx on first word
            if this_stim['stim_idx'] is None:
                this_stim['stim_idx'] = ev_code
            state += 1

        elif ev_code in noun_code_range:
            # handle critical stimuli
            if state > 1:
                assert this_stim['noun_code'] is None  # best not be set yet

            stim_words = this_stim['rsvp'].split(' ')  # pre-noun fragment
            this_stim['noun'] = re.sub(r',', '', stim_str)  # strip , "noun,"
            this_stim['noun_pos'] = len(stim_words)
            this_stim['noun_code'] = ev_code

            one_back = stim_words[-1]
            two_back = stim_words[-2]

            if one_back in ['a', 'an']:
                this_stim['article'] = one_back
            elif two_back in ['a', 'an']:
                this_stim['adjective'] = one_back
                this_stim['article'] = two_back
            else:
                msg = (
                    " bad article-(adjective)-noun sequence while scraping "
                    "scenario stream\n"
                    "leaving article, adjective as default None"
                )
                msg += '\n{0}'.format(this_stim)
                LOGGER.error(msg)
            this_stim['rsvp'] += ' ' + stim_str  # append noun and continue

        else:
            pass

    trial_data = pd.DataFrame(stim_list)[stim_cols]

    # missing values here coerce dtype to float so 'int' ensures no
    # missing values
    assert trial_data['stim_idx'].dtype == 'int'
    assert trial_data['noun_code'].dtype == 'int'
    return trial_data


def check_scn(expts):
    ''' load A/An stim from expt scenarios and scan for errors '''

    # helper
    def get_noun_context(row):
        rsvp_words = row['rsvp'].split(' ')[: row['noun_pos']]
        noun_context = '{0} {1}'.format(row['s1'], ' '.join(rsvp_words))
        noun_context = sanitize_string(noun_context)
        return noun_context

    # 1. check stim within an experiment
    for expt, specs in expts.items():
        eeg_stim_check(specs)

    # 2. check stim ids *across* all three expts
    stim_df = pd.DataFrame()
    for exp, specs in expts.items():
        stim_df = stim_df.append(load_scns(**specs))

    stim_df['noun_context'] = stim_df.apply(
        lambda row: get_noun_context(row), 1
    )

    # check item_id == noun_context up to the article *ACROSS* expts
    for item_id in stim_df.index.unique():
        stim_items = stim_df.loc[item_id].copy()
        if isinstance(stim_items, pd.Series):
            stim_items = pd.DataFrame(stim_items).T
        stim_items.drop_duplicates(inplace=True)
        if len(stim_items['noun_context'].unique()) != 1:
            variants = stim_items[
                [
                    'expt',
                    'stim_idx',
                    's1_code',
                    'noun_code',
                    'noun_pos',
                    'article',
                    'adjective',
                    'noun',
                    'noun_context',
                ]
            ].drop_duplicates()
            msg = 'noun context varies\n{0}'.format(
                '\n'.join(
                    variants.apply(
                        lambda row: ' '.join([str(d) for d in row.values]), 1
                    )
                )
            )
            LOGGER.error(msg)


def eeg_stim_check(exp_dict):
    """check stim within an eeg expt, across expt scenario files for a/an
    stim, ignore fillers.

    Problems are logged as ERROR
    """

    exp_descr = '{0}_{1}'.format(exp_dict['expt_id'], exp_dict['expt_name'])

    # precompile for joining expt 2, 3 S1+RSVP stim columns
    s1s2_regex = re.compile(r'(?P<s1>.*?\.)\s*(?P<s2>.+\.){0,1}.*')

    # scrape scenario files and compare with expt design specs
    scns_data = load_scns(**exp_dict)
    stim_idxs = scns_data.index.unique()
    n_actual_scn_ids = len(scns_data.index.unique())
    n_nominal_scn_ids = exp_dict['n_stim'] * exp_dict['stim_tuple']
    if n_actual_scn_ids != n_nominal_scn_ids:
        msg = (
            'Expt {0} number of scenario file actual item ids: {1} '
            '!= {2} nominal ids: n_stim {3} * stim_tuple {4}'
        )

        msg = msg.format(
            exp_descr,
            n_actual_scn_ids,
            n_nominal_scn_ids,
            exp_dict['n_stim'],
            exp_dict['stim_tuple'],
        )

        LOGGER.error(msg)

    # run checks per each stim_idx
    for stim_idx in stim_idxs:

        # slice this stim_idx across all scns
        stim = scns_data.loc[stim_idx]

        # monkey patch Pandas ... singleton row slices come back
        # as pd.Series column. WTF

        if isinstance(stim, pd.Series):
            stim = pd.DataFrame(stim).transpose()

        # check item_id actually appears in the right
        # number of scenario files
        if stim.shape[0] != exp_dict['n_scn']:
            msg = (
                "inconsistent scenario files {0} "
                "stim.shape[0] != exp_dict['n_scn']"
            )
            msg.format(exp_dict['expt_id'], stim.shape, stim_idx)
            LOGGER.error(msg)

        # check expt id
        exp = stim['expt'].unique()
        assert len(exp) == 1 and exp[0] == exp_dict['expt_id']

        # check noun codes ...

        # are noun_code and stim_idx is unique
        noun_codes = stim['noun_code'].unique()
        if len(noun_codes) != 1:
            LOGGER.error('multiple noun codes stim {0}'.format(stim))

        # are noun code stim idx values consistent?
        noun_code_idxs = set(
            [code_to_idx(noun_code) for noun_code in stim['noun_code']]
        )
        if len(noun_code_idxs) != 1:
            LOGGER.error('noun_code error {0}'.format(stim))

        # 2. check stim strings, expt specific info pulled from w/
        # EEG_EXP_SPECS dict ...

        # join s1 and rsvp for arcadj, yantana
        stim_str = stim.apply(
            lambda x: (x['s1'] + ' ' + x['rsvp'])
            if x['s1'] is not None
            else x['rsvp'],
            1,
        ).unique()
        assert len(stim_str) > 0

        # are stim consistent across scenario files?
        if len(stim_str) != 1:  # exp_dict['stim_tuple']:
            msg = (
                'Scenario file error: {0} {1} stim mismatch across files'
                'across .scn files:\n{2}'
            )
            msg = msg.format(exp, stim_idx, stim_str)
            LOGGER.error(msg)

        # fails if stim_str isn't of the form: s1. s2
        for stim_s in stim_str:

            stim_parts = s1s2_regex.match(stim_s)
            assert stim_parts is not None

            if stim_parts['s1'] is None:
                msg = 'Stim error: {0} {1} in S1. format\n{2}\n{3}'
                msg = msg.format(exp, stim_idx, stim_s, stim_str)
                LOGGER.error(msg)

            # two part: S1. R S V P.
            # long form output scns_data.loc[stim_idx].values[0]
            if (
                exp in ['eeg_x2_arcadj', 'eeg_x3_yantana']
                and stim_parts['s2'] is None
            ):
                msg = 'Stim error: {0} {1} in S2 RSVP format:\n{2}\n{3}'
                msg = msg.format(exp, stim_idx, stim_s, stim_str)
                LOGGER.error(msg)

    # 3. check critical a/an, adj, noun stimuli
    for crit_stim in ['article', 'noun']:
        bad_idxs = [
            idx for idx, stm in enumerate(scns_data[crit_stim]) if stm is None
        ]
        if len(bad_idxs) > 0:
            msg = 'Stim error: {0} {1} critical {2} missing:'
            msg = msg.format(exp, stim_idx, crit_stim)
            LOGGER.error(msg)
            LOGGER.error(pp.pformat(scns_data.iloc[bad_idxs]))


def merge_norms_eeg_stim(norms_df, eeg_stim_df):
    """merge normed stim with eeg stim

    Returns
    -------
    norms_eeg_a_an_df : pd.DataFrame
       merge of norm and eeg stim on matching item id iNNN_?_(a_|an)

    norm_eeg_mismatches_df : pd.DataFrame
       eeg item_id where norm stim does not match initial substring EEG stim

    """
    msg = "*** Merging norm measures with eeg scenario stim ***"
    LOGGER.info(msg)

    # helper
    def join_stim(row):
        """collapse context and rsvp for 2-sentence trials in eeg expts 2, 3"""
        stim = ''
        if row['s1'] == 'NA':
            stim = row['rsvp']
        else:
            stim = ' '.join([row['s1'], row['rsvp']])
        return stim

    # filter for the a/an norming items only
    norms_a_an = [
        not re.match(r'(?P<id>i\d{3}_.?_(a|an)).*', item_id) is None
        for item_id in norms_df.index
    ]

    norms_a_an_df = norms_df.loc[norms_a_an][['expt_id', 'stim']].copy()

    # construct an item id prefix through the article
    norms_a_an_df['item_id_pfx'] = [
        item_id[0:9] for item_id in norms_a_an_df.index
    ]

    norms_a_an_df['stim'] = norms_a_an_df['stim'].apply(sanitize_string)

    # join s1, rsvp
    eeg_a_an_stim = eeg_stim_df.apply(lambda row: join_stim(row), 1)

    eeg_stim_df['eeg_stim'] = eeg_a_an_stim
    eeg_stim_df['eeg_stim'] = eeg_stim_df['eeg_stim'].apply(sanitize_string)

    # construct the same item id prefix as for norming items
    # allow for #ERROR in eeg item_ids
    eeg_a_an_rows = [
        not re.match(r'i\d{3}_.?_(a_|an|#ERROR)_\w+_.*', item_id) is None
        for item_id in eeg_stim_df.index
    ]

    # drop scenario file specific columns
    eeg_a_an_stim_df = eeg_stim_df[eeg_a_an_rows].drop(
        ['scn_dir', 'scn_f', 'list', 'trial', 's1', 'rsvp'], axis='columns'
    )

    eeg_a_an_stim_df['item_id_pfx'] = [
        item_id[0:9] for item_id in eeg_a_an_stim_df.index
    ]

    # eeg stim left merge <- norm stim ... this ignores
    # normed stim in norm_1, norm_2 that weren't used in eeg_1
    eeg_norms_a_an_df = eeg_a_an_stim_df.merge(
        norms_a_an_df, on='item_id_pfx', how='left'
    )

    eeg_norms_a_an_df.drop_duplicates(inplace=True)

    # scan for mismatches
    mismatches = list()
    # track mismatching item indexes and stim
    for idx, row in eeg_norms_a_an_df.iterrows():
        if pd.isnull(row['item_id']):
            row['mismatch'] = 'null_item_id'
            mismatches.append(row)
        elif pd.isnull(row['item_id_pfx']):
            row['mismatch'] = 'null_item_id_pfx'
            mismatches.append(row)
        elif pd.isnull(row['eeg_stim']):
            row['mismatch'] = 'null_eeg_stim'
            mismatches.append(row)
        elif pd.isnull(row['stim']):
            row['mismatch'] = 'null_norm_stim'
            mismatches.append(row)
        else:
            if not (re.match(row['stim'], row['eeg_stim'])):
                row['mismatch'] = 'eeg_norm_stim'
                mismatches.append(row)

    eeg_norms_mismatches_df = pd.concat(mismatches, axis=1).transpose()
    eeg_norms_mismatches_df.drop_duplicates(inplace=True)
    eeg_norms_mismatches_df.sort_values(['expt', 'item_id_pfx'])

    # set indexes
    eeg_norms_a_an_df.set_index('item_id', inplace=True, drop=False)
    eeg_norms_mismatches_df.set_index('item_id', inplace=True, drop=False)
    return eeg_norms_a_an_df, eeg_norms_mismatches_df


# ------------------------------------------------------------
# EEG stim norm data extraction
# ------------------------------------------------------------
def get_probe_measures(probe, item_id, norms_df):
    """fetch and format normative measures for probe and item_id

    Parameters:

    """
    probe_doc = NLP(probe)
    assert len(probe_doc) == 1
    probe_token = probe_doc[0]
    item = norms_df.loc[item_id]
    item_measures = item['context_measures']
    n_responses = item_measures['orth']['n_responses']
    assert n_responses == item_measures['lemma']['n_responses']

    token_types = ['orth', 'lemma']
    measures = dict(item_id=item['item_id'], stim=item['stim'])
    for token_type in token_types:
        counts_df = count_dict_to_df(item[token_type])
        attr_ = token_type + "_"  # spacy specific
        measures[token_type] = calc_probe_measures(
            counts_df, getattr(probe_token, attr_)
        )
    return measures


def calc_probe_measures(counts_df, probe):
    """Back end function does the math for probe in counts_df, nothing more.

    Parameters
    ----------
    counts_df : pd.DataFrame
      string x position occurrence (= counts) matrix, e.g., from
      token.orth_ or token.lemma_
    probe: str
      string to calculate measures for

    Notes
    -----

    Use token.orth_ probes with token.orth_ occurrence matrices and
    token.lemma_ probes with token.lemma matrices, this is not checked.

    """

    # drop the NAs if any, subsequent calcs are on actual response counts
    if 'NA' in counts_df.index:
        n_NAs = int(counts_df.loc['NA'][0])
        counts_df = counts_df.drop('NA')
    else:
        n_NAs = int(0)

    # first column tallys all initial responses
    n_responses = int(counts_df[0].sum())

    if probe in counts_df.index:
        initial_count = int(counts_df.loc[probe, 0])
        # sum across positions
        anywhere_count = int(counts_df.loc[probe].sum())
    else:
        anywhere_count, initial_count, = int(0), int(0)

    initial_cloze = np.round(initial_count / n_responses, 3)
    anywhere_cloze = np.round(anywhere_count / n_responses, 3)

    probe_measures = dict(
        probe=str(probe),
        n_responses=int(n_responses),
        n_NAs=int(n_NAs),
        initial_count=int(initial_count),
        initial_cloze=float(initial_cloze),
        anywhere_count=int(anywhere_count),
        anywhere_cloze=float(anywhere_cloze),
    )

    return probe_measures


def scrub_code_map_stim(eeg_expt_specs, code_map_df, mismatches_df):
    """exclude dropped eeg items, log item_exceptions, and patch event table

    The corrections are specified in

       eeg_expt_specs[expt]['item_exclusions']
       eeg_expt_specs[expt]['item_exceptions']
       eeg_expt_specs[expt]['item_patches']

    Parameters
    ----------
    eeg_expt_specs : dict
       as loaded from YAML FILE EEG_EXPT_SPECS_F

    code_map_df : pd.DataFrame
       as returned by merge_norms_eeg_stim()

    mismatches_df : pd.DataFrame
       as returned by merge_norms_eeg_stim()

    Returns
    -------
    code_map_df : pd.DataFrame
       same as input with problematic rows dropped

    """
    msg = 'dropping excluded items according to yaml specs'
    LOGGER.info(msg)
    for expt, specs in eeg_expt_specs.items():
        if len(specs['item_exclusions']) == 0:
            msg = 'Experiment {0} no item_exclusions'.format(expt)
            LOGGER.info(msg)
        else:
            for item_id, reason in specs['item_exclusions'].items():
                msg = 'Experiment {0} excluding {1}: {2}'
                msg = msg.format(expt, item_id, reason)
                LOGGER.info(msg)
                # knock out bad item idx
                code_map_df.drop(item_id, inplace=True)

    # log troublesome items ... these must must be reviewed manually and added
    # to item_exclusions, item_exceptions maps in the expt specs YAML file
    for item_id, row in mismatches_df.iterrows():
        mismatch = row.to_csv(sep='\t')
        msg = 'Norm stim vs eeg stim mismatch: {0} {1}\n  {2}'
        msg = msg.format(item_id, row['mismatch'], mismatch)
        LOGGER.warning(msg)

    # horrible, log and run item_patche, details in eeg_expt_specs.yml
    for expt, specs in eeg_expt_specs.items():
        if 'item_patches' in specs.keys():
            for item_id, patch in specs['item_patches'].items():
                msg = 'Patching {0} {1} {2}: {3}'.format(
                    expt, item_id, patch['comment'], patch['code']
                )
                LOGGER.info(msg)
                # have to pass exec the local in a dict
                ldict = {'code_map_df': code_map_df}
                exec(patch['code'], globals(), ldict)
                code_map_df = ldict['code_map_df']

    # check number of stim ids computed vs. listed manually
    # in eeg_expt_specs.yml
    for expt, specs in eeg_expt_specs.items():
        expt_df = code_map_df[code_map_df['expt'] == expt]
        n_item_ids = len(expt_df.index)
        n_unique_item_ids = len(expt_df.index.unique())
        n_good_item_ids = specs['n_good_item_ids']
        if not n_item_ids == n_unique_item_ids == n_good_item_ids:
            msg = (
                "code_map file generation failed, "
                "code_map_df['{0}'].shape == {1}"
            )
            msg = msg.format(expt, expt_df.shape)
            LOGGER.critical(msg)
            raise ValueError(msg)

    code_map_df.sort_values(['expt', 'stim_idx'], inplace=True)
    return code_map_df


def merge_norm_measures_with_code_map(code_map_df, norms_df):
    """Merge norm measures with the EEG stim using item_id as the merge key

    This looks up the presented article and noun cloze in the norming
    responses and attaches them together with the previously computed
    normative measures to the EEG stimulus according to the item id

    Note:
      The code_map_df must be scrubbed clean so there are no
      duplicate item_id indexes and all bad rows have been dropped.

    Parameters
    ----------
    code_map_df : pd.DataFrame
      as returned by scrub_code_map_stim()
    norms_df : pd.DataFrame
      as returned by udck19_utils::load_yaml_norms()

    Returns
    -------
    code_map_df : pd.DataFrame
      the input dataframe extended with additional columns of norm measures

    """
    LOGGER.info('merge_norm_measures_with_code_map')

    # init and lookup the stim item measures
    # code_map_df['article_cloze'] = np.nan
    # code_map_df['noun_lemma_anywhere'] = np.nan
    # code_map_df['noun_cloze'] = np.nan

    for idx, row in code_map_df.iterrows():
        eeg_item_id, item_info = None, None

        eeg_item_id = row['item_id']
        item_info = parse_item_id(eeg_item_id)
        assert int(item_info['stim']) == row['stim_idx']
        assert item_info['noun'] == row['noun']
        assert item_info['article'] == row['article']

        article, noun = None, None
        article = row['article']
        noun = row['noun']

        # ------------------------------------------------------------
        # 1. Article norming: contexts without articles
        #    norm_1, norm_3, norm_6,
        # ------------------------------------------------------------

        # 1. fetch norming measures for the bare context, no article

        # THE CRITICAL STEP ... requires norming v. eeg item code match
        # Ex. for eeg item_id i001_1_a__NA_NA use norm item_id i001_NA_NA_NA
        # to look up the 'a' article measures

        article_norm_item_id = None
        article_norm_item_id = re.sub(
            r'_(an|a_)_[a-zA-Z0-9]+_[a-zA-Z0-9]+$', '_NA_NA_NA', eeg_item_id
        )

        # 1.1 presented article
        article_meas = None
        article_meas = get_probe_measures(
            article, article_norm_item_id, norms_df
        )

        code_map_df.loc[idx, 'orth_article_initial_cloze'] = article_meas[
            'orth'
        ]['initial_cloze']

        # 1.2 presented noun. Also query the noun lemma *without* the
        # article, i.e., contextual constraint for probe NOUN, any form without
        # the influence of the article according to the norms.
        NA_noun_meas = None
        NA_noun_meas = get_probe_measures(
            noun, article_norm_item_id, norms_df
        )

        # code_map_df.loc[idx, 'NA_noun_lemma_anywhere_cloze'] =
        # NA_noun_meas['lemma']['anywhere_cloze']

        for ttype in ['orth', 'lemma']:
            for key, val in NA_noun_meas[ttype].items():
                if 'cloze' in key:
                    col_name = '{0}_NA_noun_{1}'.format(ttype, key)
                    code_map_df.loc[idx, col_name] = NA_noun_meas[ttype][key]

        # ------------------------------------------------------------
        # we don't have norming data for arcadj article + adj EEG stim
        # so skip items with non-NA in adjective position
        # ------------------------------------------------------------
        adj_search = re.match(
            r"^i\d{3}_\d_(?:a_|an)(?!_NA_)\w+$", eeg_item_id
        )
        if adj_search is not None:
            LOGGER.info(
                f"article + adjectives not normed, skipping {eeg_item_id}"
            )
            continue

        # ------------------------------------------------------------
        # 2. Noun norming: context *including* the presented article.
        #    norm_2, norm_4, norm_5
        # ------------------------------------------------------------

        # again, CRITICAL STEP ... requires norming v. eeg item code match
        # e.g., eeg item_id i001_1_an_NA_apple looks up values for
        # from norming item_id i001_1_an_NA_NA

        noun_norm_item_id = None
        # replace noun string with NA for the norm_df index with articles
        noun_norm_item_id = re.sub(r'_[a-zA-Z0-9]+$', '_NA', eeg_item_id)
        assert noun_norm_item_id in norms_df.index

        # for article + adj EEG stim we don't have norming data,
        # move along to next item
        # if noun_norm_item_id not in norms_df.index:
        #    print(f"no noun or context measures for {eeg_item_id}")
        #    continue

        # 2.1 query the noun following the article,
        LOGGER.info(f"noun OK {eeg_item_id}")
        noun_meas = None
        noun_meas = get_probe_measures(noun, noun_norm_item_id, norms_df)

        # scrape the return values
        for ttype in ['orth', 'lemma']:
            for key, val in noun_meas[ttype].items():
                if 'cloze' in key:
                    col_name = f'{ttype}_ART_noun_{key}'
                    code_map_df.loc[idx, col_name] = noun_meas[ttype][key]

        # 3. query the precomputed context measures regardless of
        # presented probe ... constraint-like highest cloze,
        # info/entropy etc.  Note: inventory of measures returned is
        # set in get_context_measures()
        for context_id in [article_norm_item_id, noun_norm_item_id]:
            context_measures = None
            context_measures = norms_df.loc[context_id]['context_measures']

            # e.g., orth, lemma
            for key, val in sorted(context_measures.items()):
                # e.g., initial cloze, etc..
                for measure, value in val.items():
                    col_name = '{0}_{1}'.format(key, measure)
                    if hasattr(value, '__getitem__'):
                        # format lists [a,b,c] as JSON "[a, b, c]"
                        code_map_df.loc[idx, col_name] = '[{0}]'.format(
                            ', '.join(['{}'.format(v) for v in value])
                        )
                    else:
                        code_map_df.loc[idx, col_name] = value

    return code_map_df


def make_eeg_code_map(eeg_expt_specs, eeg_stim_f, norm_measures_yaml_f):
    """build master tag table for all three experiments and dump by expt"""

    LOGGER.info(f"make_eeg_code_map {eeg_expt_specs} {eeg_stim_f}")

    # 1. slurp item counts and precomputed norm context measures ... slow
    norms_df = load_yaml_norms(norm_measures_yaml_f)
    norms_df.set_index('item_id', drop=False, inplace=True)

    # 2. slurp EEG stim item text and scrub stim string
    eeg_stim_df = pd.read_csv(
        EEG_STIM_F, sep='\t', keep_default_na=False, quoting=3
    )
    eeg_stim_df.set_index('item_id', drop=False, inplace=True)

    # 3. sync, merge stim from norms and eeg stim
    code_map_df, mismatches_df = merge_norms_eeg_stim(norms_df, eeg_stim_df)

    # 4. drop excluded items and patch, see EEG_EXPT_SPECS_F for
    # specifics
    code_map_df = scrub_code_map_stim(
        eeg_expt_specs, code_map_df, mismatches_df
    )

    # ------------------------------------------------------------
    # 5. add the ARTICLE anchor regexp event code pattern
    # in place of cdbl .{*}{2}{10011-10904} like so ...
    #
    # - arquant eeg_1:
    #     #d{1,3} 2 noun_code
    #
    # - arcadj eeg_2: as eeg_1 plus art+adj+noun sequences
    #
    #     art+noun noun_codes end in [1234]:
    #         (#d{1,3}) 2 noun_code
    #
    #     art+adj+noun noun_codes end in [78]:
    #          (#d{1,3}) 1 #d{1,3} 2 noun_code
    #
    #   - yantana eeg_3: as eeg_1
    #         #d{1,3} 2 noun_code
    # ------------------------------------------------------------

    def noun_code_to_article_anchor(noun_code):
        """ map noun code to  1 vs. 2 back article anchors"""

        if noun_code % 10 in [1, 2, 3, 4, 5, 6]:
            # article + noun trials in eeg_1, eeg_2, eeg_3, so
            # here a 1-back pattern
            article_anchor = r'(#\d{{1,3}}) 2 {0}'
        elif noun_code % 10 in [7, 8]:
            # article + adjective + noun trials in eeg_2 only, so
            # here a 2-back pattern
            article_anchor = r'(#\d{{1,3}}) 1 \d{{1,3}} 2 {0}'
        else:
            msg = 'uh oh ... unknown noun event code : {0}'
            msg = msg.format(noun_code)
            raise ValueError(msg)

        return article_anchor.format(noun_code)

    # sweep function down noun code column, all expts
    code_patterns = code_map_df['noun_code'].apply(
        lambda x: noun_code_to_article_anchor(x)
    )
    code_map_df.insert(1, 'regexp', code_patterns)

    # 6. add the single trial norm measures
    code_map_df = merge_norm_measures_with_code_map(code_map_df, norms_df)
    return code_map_df


def write_code_map(master_code_map_df, eeg_expt_files):
    """split up the master event table into separate event tables for each
    expt eeg/stim dataset.

    """
    #  dump master like so if needed for debugging ...
    #  master_code_map_df.to_csv(MASTER_CODE_MAP_F,
    #                                sep='\t', na_rep='NA',
    #                                float_format='%3.3f',
    #                                index=False)

    for eeg_expt in sorted(master_code_map_df['expt'].unique()):
        code_map_f = eeg_expt_files[eeg_expt]['code_map_f']
        LOGGER.info(
            'Writing event tags for {0}: {1}'.format(eeg_expt, code_map_f)
        )

        expt_tags = None
        expt_tags = master_code_map_df[master_code_map_df['expt'] == eeg_expt]
        expt_tags.index.rename('Index', inplace=True)
        expt_tags.reset_index(inplace=True)
        expt_tags.to_csv(
            code_map_f,
            sep='\t',
            na_rep='NA',
            float_format='%3.3f',
            index=False,  # do not write row names
        )


def eeg_expt_stim_to_csv(exps, csv_f):
    """Save A/An stim from expt scenarios as tab-separated text

    Parameters
    ----------
    csv_f : str
        path to output file ... overwritten without mercy

    exps : list of EEG_EXPN dicts

    """
    stim_df = pd.DataFrame()
    for exp, specs in exps.items():
        stim_df = stim_df.append(load_scns(**specs))
    stim_df.to_csv(csv_f, sep='\t', na_rep='NA', quoting=3)


# --------------------------------------------------
# Epoch, single trial processing utils
# --------------------------------------------------
def eeg_to_recorded_epochs(eeg_expt_files):
    """scan the event tags file for event codes of interest, slice and
    save the epochs.

    Note: global ALL_EEG_STREAMS

    """

    LOGGER.info('eeg_to_recorded_epochs')

    for eeg_expt, files in eeg_expt_files.items():

        # readability
        source_h5_f, h5_f = None, None
        code_map_f, recorded_epochs_f = None, None

        source_h5_f = files['source_h5_f']
        h5_f = files['h5_f']
        code_map_f = files['code_map_f']
        recorded_epochs_f = files['recorded_epochs_f']

        # check we have source/input files and output is writeable
        for f in [source_h5_f, code_map_f]:
            if not Path(f).exists():
                raise FileExistsError(f)

        Path(files['recorded_epochs_f']).touch(exist_ok=True)

        # make a working copy of the pinned HDF5 file from legacy2df/mkh5
        LOGGER.info(f"copying source EEG file {source_h5_f} to {h5_f}")
        _ = shutil.copy(source_h5_f, h5_f)
        h5_f.chmod(0o644)  # rw-r-r

        with open(h5_f, 'rb') as stream:
            md5 = hashlib.md5(stream.read()).hexdigest()
        if not md5 == files["source_md5"]:
            msg = f"shutil.copy({source_h5_f}, {h5_f} bad md5 checksum: {h5_f}"
            raise ValueError(msg)

        msg = 'Expt {0}: slicing eeg {1} into epochs {2}'
        msg = msg.format(eeg_expt, h5_f, recorded_epochs_f)
        LOGGER.info(msg)

        # snapshot the complete event table into the epochs lookup table
        # in the hdf5 data file. Left, right interval padding is
        # added here for trimming filter edge artifacts later
        h5 = mkh5.mkh5(str(h5_f))
        event_table = h5.get_event_table(str(code_map_f))
        h5.set_epochs(
            'articles',
            event_table,
            RECORDED_EPOCH_START,
            RECORDED_EPOCH_STOP
        )

        # fetch just Epoch_idx, Time, and COIs to save space
        mkpy_cois = list(COI_LABELS.keys())
        assert all([key in event_table.columns for key in mkpy_cois])
        epochs_df, _ = h5.get_epochs(
            'articles',
            'pandas',
            ['Epoch_idx', 'Time'] + mkpy_cois + ALL_EEG_STREAMS)

        # the general purpose mkpy column names are impenetrable,
        # rename for human consumption in these experiments
        epochs_df.rename(columns=COI_LABELS, inplace=True)

        # save with pandas
        epochs_df.to_hdf(files['recorded_epochs_f'], 'articles', mode='w')


def load_epochs(epochs_f):
    """read tabular hdf5 epochs file, return as pd.DataFrame

    Parameter
    ---------
    epochs_f : str
        name of the recorded epochs file to load

    Return
    ------
    df : pd.DataFrame
        columns in INDEX_NAMES are pd.MultiIndex axis 0
    """

    epochs_df = pd.read_hdf(epochs_f)

    # patch in sub_id
    if 'sub_id' not in epochs_df.columns:
        epochs_df['sub_id'] = epochs_df['data_group']
    assert all(epochs_df['sub_id'] == epochs_df['data_group'])

    epochs_df.set_index(INDEX_NAMES, inplace=True)
    epochs_df.sort_index(inplace=True)
    validate_epochs_df(epochs_df)

    LOGGER.info(f'{epochs_df.shape}')
    LOGGER.info(f'{epochs_df.head()}')
    LOGGER.info(f'{epochs_df.tail()}')

    return epochs_df


def validate_epochs_df(epochs_df):
    """check form and index of the epochs_df is as expected

    Parameters
    ----------
    epochs_df : pd.DataFrame

    """

    LOGGER.info('validating epochs pd.DataFrame')
    assert epochs_df.index.names == INDEX_NAMES

    # TO DO: ? add single epoch time index check


def DEPRECATED_drop_garv_artifacts(epochs_df):
    """drop epochs where a positive log flag indicates garv rejection

    returns epochs data frame without the bad trials and a dataframe
    of the bad *events*, Time == 0 (only) for reporting.

    """

    LOGGER.info('entering drop_garv_artifacts()')
    validate_epochs_df(epochs_df)

    # slice the Time index == 0 and confirm there is a critical event there.
    slice_time_0 = pd.IndexSlice[:, :, :, :, 0]
    match_events_df = epochs_df.loc[slice_time_0, :]

    assert (match_events_df[['log_evcodes', 'match_code']] > 0).values.all()

    # bool True at flagged artifacts
    bad_event_mask = match_events_df['log_flags'] > 0
    garv_bad_epoch_idxs = match_events_df[
        bad_event_mask
    ].index.get_level_values('Epoch_idx')

    # toss the bad epochs
    good_epochs_df = epochs_df.drop(garv_bad_epoch_idxs)

    # capture garv positive event codes
    bad_events_slice = pd.IndexSlice[garv_bad_epoch_idxs.values, :, :, :, 0]
    bad_events_df = epochs_df.loc[bad_events_slice, :]

    # show the goods
    with pd.option_context(
        'display.max_rows', None, 'display.max_columns', None
    ):
        msg = (
            f'dropping {garv_bad_epoch_idxs.shape[0]} of {epochs_df.shape[0]} '
            f'epochs:\n{garv_bad_epoch_idxs}'
        )
        LOGGER.info(msg)

    # count the bads
    artifact_counts = pd.crosstab(
        bad_events_df.reset_index()['sub_id'],
        columns=bad_events_df.reset_index()['item_id'],
        margins=True,
    )

    subject_artifacts = artifact_counts['All'].sort_values()
    item_artifacts = artifact_counts.loc['All'].sort_values()

    with pd.option_context(
        'display.max_rows', None, 'display.max_columns', None
    ):
        LOGGER.info(f'subject_artifacts:\n{subject_artifacts}')
        LOGGER.info(f'item_artifacts:\n{item_artifacts}')

    return good_epochs_df, bad_events_df


def rereference_A1A2(epochs_df, eeg_streams):
    """rereference eeg data to math linked mastoids by subtracting half of A2
    """
    LOGGER.info(f'rereference_A1A2 {eeg_streams}')
    validate_epochs_df(epochs_df)

    # pd.DataFrame is kind of slow on the rows, falling back to numpy
    def reref(eeg_data_row, a2_jdx):
        return eeg_data_row - (0.5 * eeg_data_row[a2_jdx])

    # keep dataframe columns for next jdx index lookup
    eeg_df = epochs_df[eeg_streams]

    A2_jdx = eeg_df.columns.get_loc('A2')
    assert eeg_df.columns[A2_jdx] == 'A2'

    # 2-D array  of the eeg stream data ... presumably a view
    eeg_data = eeg_df.values
    reref_eeg_data = np.full_like(eeg_data, np.nan)
    assert reref_eeg_data.shape == eeg_df.values.shape

    # do the math
    n_rows = np.size(reref_eeg_data, 0)
    for idx in range(n_rows):
        reref_eeg_data[idx, :] = eeg_data[idx, :] - (
            0.5 * eeg_data[idx, A2_jdx]
        )

    # return the result as a data frame
    reref_df = epochs_df.copy()
    reref_df[eeg_streams] = reref_eeg_data
    assert all(epochs_df['A2'] == 2.0 * reref_df['A2'])

    validate_epochs_df(reref_df)
    return reref_df


def add_roi_columns(df, rois):
    """add a new "region of interest" column equal to the mean of existing
eeg columns

    Parameters
    ----------

    df : pd.DataFrame samples in rows, channels in column

    rois : dict
        each key is an new column roi label, each value is a list of
        column labels to pool

    Example
    -------

    rois = {
    'cp_roi': ['LMCe', 'RMCe', 'MiCe', 'MiPa', 'LDCe', 'RDCe']

    }

    """
    LOGGER.info(f'add_roi_columns {rois}')
    for roi, chans in rois.items():
        df[roi] = df[chans].mean(axis=1)  # average across chans columns
        LOGGER.info('\nNew ROI head\n{0}'.format(df[chans + [roi]].head()))
        LOGGER.info('\nNew ROI tail\n{0}'.format(df[chans + [roi]].tail()))
    return df


# ------------------------------------------------------------
# Process recorded epochs into ready-for-measurement single trial epochs
# ------------------------------------------------------------
def recorded_epochs_to_prepared_epochs(eeg_expt_files, prep_specs):
    """wrapper around the expts"""
    LOGGER.info('processing recorded epochs to prepared epochs')
    pd.concat(
        (
            _prepare_epochs(expt, files, prep_specs)
            .reset_index().set_index(INDEX_NAMES)
            for expt, files in eeg_expt_files.items()
        ),
        axis=0).to_hdf(PREPOCHS_ALL_F, key="prepochs_all", mode='w')


def _prepare_epochs(expt, files, prep_steps):
    """backend prepare epochs for modelling. functions log themselves

    Parameters
    ----------
    expt : str {eeg_1, eeg_2, eeg_3}

    files: dict
        filenames given by EEG_EXPT_FILES[expt]

    prep_steps : list of dict
        each key is a function, value is args list and kwargs dict

    Notes
    -----
    . load recorded single trial epochs

    . run data transforms options, order drawn from prep_steps

    . return prepped data for file dump

    """

    # helper to make Epoch_idx unique across the experiments
    def eeg_n_to_epoch_idx(df):
        """returns unique epoch index across experiments eeg_1, eeg_2, eeg_3

        first integer of index codes the eeg expt number, e.g.,

        eeg_1, epoch 37   -> 10037
        eeg_1, epoch 2123 -> 22123
        eeg_3, epoch 5111 -> 35111

        """
        expt_names = df['expt'].unique()
        assert len(expt_names) == 1
        expt_int = int(expt_names[0][-1:])  # trailing eeg experiment number

        return (expt_int * 10000) + df['Epoch_idx']

    LOGGER.info(f'preparing epochs {expt}')

    # 1. Load expt epochs and undo the index ... ugh
    epochs_f = files['recorded_epochs_f']  # from the h5 epochs dump
    pr_epochs_df = pd.read_hdf(epochs_f, 'articles')

    eeg_epoch_idx = eeg_n_to_epoch_idx(pr_epochs_df)
    pr_epochs_df['Epoch_idx'] = eeg_epoch_idx

    # replace the index
    pr_epochs_df.set_index(INDEX_NAMES, inplace=True)

    # 2. Run epoch processing sequence, cf. functools.partial
    for fnc, fargs, fkwargs in prep_steps:
        pr_epochs_df = fnc(pr_epochs_df, *fargs, **fkwargs)

    return pr_epochs_df


# ------------------------------------------------------------
# epochs processing function: all return modified copy
# ------------------------------------------------------------
def tag_peak_to_peak_excursions(epochs_df, eeg_streams, crit_ptp):

    # CRIT_PTP = 150.0
    ptp_codes = []

    LOGGER.info(f"""
    Tagging epochs with EEG peak-to-peak amplitude > {crit_ptp} microvolts
    """)

    # scan each epoch for peak-to-peak excursions
    for idx, epoch in epochs_df[eeg_streams].groupby('Epoch_idx'):
        ptps = epoch.to_numpy().ptp(axis=0).T  # pandas 24 T to keep it wide
        mask = (ptps > crit_ptp).astype('uint8')  # 0's and 1's each channel
        ptp_codes.append((idx, encode_eeg_fail(mask)))

    # before
    n_samps, n_epochs = check_epochs_shape(epochs_df)

    # propagate to all time points in the epoch
    ptp_excursion = np.repeat(ptp_codes, n_samps, axis=0)
    assert all(
        epochs_df.index.get_level_values('Epoch_idx') == ptp_excursion[:, 0]
    )
    epochs_df['ptp_excursion'] = ptp_excursion[:, 1]

    # after
    assert n_samps, n_epochs == check_epochs_shape(epochs_df)

    ptp_excursion_epoch_ids = [
        epoch_id for epoch_id, code in ptp_codes if code > 0
    ]
    msg = f"""
    ptp_excursions: {len(ptp_excursion_epoch_ids)}
    {ptp_excursion_epoch_ids}
    """
    LOGGER.info(msg)
    return epochs_df


def tag_flat_eeg(epochs_df, eeg_streams, blocking_min, blocking_n_samps):
    # streams: eeg data columns
    # blocking criteria, e.g., less than this microvolts in
    # in an interval of this many consecutive data points
    #   blocking_min = 0.01
    #   blocking_n_samps = 5

    LOGGER.info(f"""
    Tagging flat EEG epochs where peak-to-peak in any consecutive
    {blocking_n_samps} samples < {blocking_min} microvolts ...
    """)

    blocking_codes = []
    for idx, epoch in epochs_df[eeg_streams].groupby('Epoch_idx'):

        n_samps, n_epochs = check_epochs_shape(epochs_df)

        # numpy peak-to-peak
        epoch_arry = epoch.to_numpy()  # pandas 24
        win_mins = bn.move_min(epoch_arry, window=blocking_n_samps, axis=0)
        win_maxs = bn.move_max(epoch_arry, window=blocking_n_samps, axis=0)

        # minimum peak-to-peak of any window in the epoch
        win_ptp = np.nanmin(win_maxs - win_mins, axis=0)
        blck_mask = (win_ptp < blocking_min).astype('uint8')
        blocking_codes.append((idx, encode_eeg_fail(blck_mask)))

    blocked = np.repeat(blocking_codes, n_samps, axis=0)
    assert all(
        epochs_df.index.get_level_values('Epoch_idx') == blocked[:, 0]
    )

    epochs_df['blocked'] = blocked[:, 1]
    assert n_samps, n_epochs == check_epochs_shape(epochs_df)

    blocked_epoch_ids = epochs_df.query(
        "blocked != 0"
    ).index.unique('Epoch_idx')

    LOGGER.info(
        f"{len(blocked_epoch_ids)} blocked epochs: {blocked_epoch_ids}"
    )

    return epochs_df


def tag_garv_artifacts(epochs_df):

    # unpack and propagate garv codes into columns
    n_samps, n_epochs = check_epochs_shape(epochs_df)
    epochs_df['garv_blink'] = np.repeat(
        epochs_df.query("Time==0")['garv_reject'].apply(
            lambda x: 1 if x >= 48 else 0).to_numpy(),  # pandas 24
        n_samps
    )

    # propagate and rename garv rejects
    epochs_df['garv_screen'] = np.repeat(
        epochs_df.query("Time==0")['garv_reject'].apply(
            lambda x: "accept" if x == 0 else "reject").to_numpy(),
        n_samps
    )

    assert n_samps, n_epochs == check_epochs_shape(epochs_df)
    LOGGER.info(
        "Experimenter tagged artifacts in garv_reject,"
        " blinks in column 'garv_blink'"
    )

    return epochs_df


def consolidate_artifacts(epochs_df, eeg_screen_cols, eeg_screen_col):
    # combine eeg_screen_cols > 0 into one artifact indicator column
    n_samps, n_epochs = check_epochs_shape(epochs_df)
    epochs_df[eeg_screen_col] = np.repeat(
        epochs_df.query("Time==0")[eeg_screen_cols].apply(
            lambda x: "accept"
            if x.sum() == 0 else "reject",
            1  # axis 1 to iterate by row and sum across artifact columns
        ).to_numpy(),
        n_samps
    )
    assert n_samps, n_epochs == check_epochs_shape(epochs_df)
    msg = f"""
    Artifact(s) in any of {eeg_screen_cols} are tagged as a reject
    in indicator column '{eeg_screen_col}'
    """
    LOGGER.info(msg)
    return epochs_df


def make_article_item_id(epochs_df):
    """tag article item ids the same way across the three experiments

    **NOTE:** Because of counterbalancing schemes, the article_item_id
    is only correct for *ARTICLES* and *MUST NOT BE USED FOR NOUN ITEM
    ANALYSIS*

    Within and across the three data sets, stimulus contexts that are
    are the same up to but not including the article have the same
    `article_item_id`

    Procedure: strip the noun from the long form `item_id` to create
    `article_item_id`

    """

    LOGGER.info("Converting item_id to article_item_id ...")
    item_id_idx = epochs_df.index.names.index('item_id')
    art_item_id = [
        re.match(r"(?P<id>i.+)_\w+$", iid)['id']
        for iid in epochs_df.index.get_level_values(item_id_idx)
    ]

    assert len(art_item_id) == len(epochs_df)
    epochs_df['article_item_id'] = art_item_id
    return epochs_df


def downsample_epochs(epochs_df, t0, t1, by):
    LOGGER.info(f"Downsampling ... decimating from {t0} to {t1} by {by}")

    # careful with index slice (start, stop, step) in pandas
    # start, stop are ms ROW LABELS, step is a ROW INDEX *COUNTER* not ms
    assert epochs_df.index.names == [
        'Epoch_idx', 'expt', 'sub_id', 'item_id', 'Time'
    ]

    time_slicer = pd.IndexSlice[:, :, :, :, slice(t0, t1, by)]
    epochs_df = epochs_df.loc[time_slicer, :].copy()
    return epochs_df


# deprecated:  moved to notebook udck_supp_1.ipynb
def drop_data_rejects(epochs_df, reject_column):
    """returns a copy of epochs_df where reject column is non-zero"""

    LOGGER.info(f"dropping rejects {reject_column}")

    # enforce all data in each epoch is marked all 0 good or all
    # (some kind of, possibly various) non-zero bad
    goods = []
    for epoch_idx, epoch_data in epochs_df.groupby("Epoch_idx"):
        if max(epoch_data[reject_column]) == 0:
            goods.append(epoch_idx)

    good_epochs_df = epochs_df.query("Epoch_idx in @goods").copy()
    good_epochs_df.sort_index(inplace=True)

    # sanity check the result
    for epoch_idx, epoch_data in good_epochs_df.groupby('Epoch_idx'):
        n_goods = len(np.where(epoch_data[reject_column] == 0)[0])
        if n_goods != len(epoch_data):
            raise Exception('uncaught exception')

    return good_epochs_df


def bimastoid_reference(epochs_df, eeg_streams, a2):
    """math-linked mastoid reference = subtract half of A2, all channels"""

    LOGGER.info(f"bimastoid_reference {a2}")

    half_A2 = epochs_df[a2].values / 2.0
    br_epochs_df = epochs_df.copy()
    for col in eeg_streams:
        br_epochs_df[col] = br_epochs_df[col] - half_A2

    return br_epochs_df


def center_on_interval(epochs_df, eeg_streams, start, stop):
    """eeg_stream subtract the mean of Time index slice(start:stop)

    Parameters
    ----------
    epochs_df : pd.DataFrame
        must have Epoch_idx and Time row index names

    eeg_streams: list of str
        column names to apply the transform

    start, stop : int,  start < stop
        basline interval Time values, stop is inclusive

    """

    msg = f"centering on interval {start} {stop}: {eeg_streams}"
    LOGGER.info(msg)

    validate_epochs_df(epochs_df)
    times = epochs_df.index.unique('Time')
    assert start >= times[0]
    assert stop <= times[-1]

    # baseline subtraction ... compact expression, numpy is faster
    qstr = f"{start} <= Time and Time < {stop}"
    epochs_df[eeg_streams] = epochs_df.groupby(["Epoch_idx"]).apply(
        lambda x:  x[eeg_streams] - x.query(qstr)[eeg_streams].mean(axis=0)
    )
    validate_epochs_df(epochs_df)
    return epochs_df


def lowpass_filter_epochs(
        epochs_df,
        eeg_streams,
        cutoff_hz,
        width_hz,
        ripple_db,
        sfreq,
        trim_edges,
):
    """FIRLS wrapper"""

    # ------------------------------------------------------------
    # encapsulate filter helpers
    # ------------------------------------------------------------
    def _get_firls_lp(cutoff_hz, width_hz, ripple_db, sfreq):
        """
        FIRLS at https://scipy-cookbook.readthedocs.io/items/FIRFilter.html

        Parameters
        ----------

        cutoff_hz : float
            1/2 amplitude attenuation (-6dB) frequency in Hz, e.g., 5.0, 30.0

        width_hz : float
            width of transition band in Hz, symmetric around cutoff_hz

        ripple_db : float
            attenuation in the stop band, in dB, e.g., 24.0, 60.0

        sfreq : float
            sampling frequency, e.g., 250.0, 500.0

        """

        LOGGER.info(f"""
        Buildiing firls filter: cutoff_hz={cutoff_hz}, width_hz={width_hz}, ripple_db={ripple_db}, sfreq={sfreq}
        """)

        # Nyquist frequency
        nyq_rate = sfreq / 2.0

        # transition band width in normalizied frequency
        width = width_hz / nyq_rate

        # order and Kaiser parameter for the FIR filter.
        N, beta = kaiserord(ripple_db, width)

        # firwin with a Kaiser window to create a lowpass FIR filter.
        taps = firwin(N, cutoff_hz / nyq_rate, window=('kaiser', beta))

        # frequency response ... useful for reporting
        w, h = freqz(taps)

        return taps, N, beta, w, h

    # ------------------------------------------------------------
    def _apply_firls_lp(df, columns, taps, N, beta):
        """apply the FIRLS filtering

        filtfilt() mangles data coming and going, doubles the order so
        instead we forward pass with lfilter() and compensate for the delay

        """

        assert len(taps) % 2 == 1  # enforce odd number of taps

        delay = int((len(taps) - 1) / 2)
        a = 1.0

        msg = f"""
        applying linear phase delay compensated filter.
        a: {a}, N: {N}, delay: {delay}
        taps:
        {taps}
        """
        LOGGER.info(msg)

        filt_df = df.copy()
        for column in columns:

            # forward pass
            filtered_column = lfilter(taps, a, df[column])

            # roll the phase shift by delay back to 0
            filt_df[column] = np.roll(filtered_column, -delay)

        return filt_df

    # build and apply the filter
    taps, N, beta, w, h = _get_firls_lp(
        cutoff_hz=cutoff_hz,
        width_hz=width_hz,
        ripple_db=ripple_db,
        sfreq=sfreq,
    )

    filt_epochs_df = _apply_firls_lp(epochs_df, eeg_streams, taps, N, beta)

    # optionally drop corrupted data
    if trim_edges:
        half_width = int(np.floor(N / 2))
        times = filt_epochs_df.index.unique('Time')
        start_good = times[half_width]  # == first good sample b.c. 0-base index
        stop_good = times[-(half_width+1)]  # last good sample, 0-base index
        return filt_epochs_df.query(
            "Time >= @start_good and Time <= @stop_good"
        )
    else:
        return filt_epochs_df


if __name__ == '__main__':

    LOGGER.info('udck19 EEG single trial data wrangling')

    # ------------------------------------------------------------
    # 1. scrape EEG experimental items scenario files into CSV text
    # file and scan experimental items from scenario files, log
    # errors/warnings
    # ------------------------------------------------------------
    eeg_expt_stim_to_csv(EEG_EXPT_SPECS, EEG_STIM_F)
    check_scn(EEG_EXPT_SPECS)

    # ------------------------------------------------------------
    # 2. slurp EEG stim, norm measures, merge them into a mkpy code
    #    map. Discard problematic items, e.g., stim and scenario
    #    coding errors, and export the master code map with item_ids,
    #    stim, code tags, stim, and norm measures as a text file.
    # ------------------------------------------------------------
    master_code_map_df = make_eeg_code_map(
        EEG_EXPT_SPECS,
        EEG_STIM_F,
        NORM_MEASURES_YAML_F,
    )
    write_code_map(master_code_map_df, EEG_EXPT_FILES)

    # ------------------------------------------------------------
    # 3. slice epochs of interest from the three eeg experiments and
    #    dump untouched with event table decorations
    # ------------------------------------------------------------
    eeg_to_recorded_epochs(EEG_EXPT_FILES)

    # ------------------------------------------------------------
    # 4. prepare recorded_epochs for modeling and measurement
    # 
    #    Processing funtions and arguments for the epochs preparation, each a
    #    tuple of (fnc, args, kwargs) is run in list order as
    #
    #    pr_epochs_df = fnc(pr_epochs_df, *args, **kwargs)
    # ------------------------------------------------------------

    PREP_STEPS = [

        # lowpass antialising filter: 25 Hz for 5x oversampling after
        # decimating by 2 to downsample from 250 to 125 samples/second
        (
            lowpass_filter_epochs,
            (),
            {
                "eeg_streams": ALL_EEG_STREAMS,
                "cutoff_hz": 25.0, # half amplitude 
                "width_hz": 12.0,  # transition band 24 - 31 Hz
                "ripple_db": 60.0,  # max pass band ripple = +1.001
                "sfreq": 250.0,  # samples/second
                "trim_edges": True,  # discard 1st and last 1/2 filter width
            }
        ),

        # downsample by 2 from 250 to 125 samples/second
        (
            downsample_epochs,
            (),
            {
                "t0": DOWNSAMPLED_EPOCH_START,
                "t1": DOWNSAMPLED_EPOCH_STOP,
                "by": 2
            }
        ),

        # re-reference
        (
            bimastoid_reference,
            (),
            {
                "eeg_streams": ALL_EEG_STREAMS,
                "a2": "A2"
            }
        ),

        # center each epoch on this interval
        (
            center_on_interval,
            (),
            {
                "eeg_streams": ALL_EEG_STREAMS,
                "start": BASELINE_START,
                "stop": BASELINE_STOP,
            },
        ),


        # make the article item id for single trial analysis across
        # expts *NOTE* Because of counterbalancing schemes, the
        # article_item_id is only correct for *ARTICLES* and *MUST NOT
        # BE USED FOR NOUN ITEM ANALYSIS*
        (make_article_item_id, (), {}),

        # tag peak-to-peak amplitude excursions
        (
            tag_peak_to_peak_excursions,
            (),
            {"eeg_streams": ALL_EEG_STREAMS, "crit_ptp": 150.0}
        ),

        # tag flat eeg
        (
            tag_flat_eeg,
            (),
            {
                "eeg_streams": ALL_EEG_STREAMS,
                "blocking_min": 0.01,
                "blocking_n_samps": 5,
            }
        ),

        # unpack garv codes into epochs columns
        (tag_garv_artifacts, (), {}),

        # consolidate artifacts
        (
            consolidate_artifacts,
            (),
            {
                "eeg_screen_cols": ['garv_blink', 'ptp_excursion', 'blocked'],
                "eeg_screen_col": 'eeg_screen',
            }
        ),

    ]

    LOGGER.info(f"preparing  epochs {PREP_STEPS}")
    recorded_epochs_to_prepared_epochs(EEG_EXPT_FILES, PREP_STEPS)
