#!/usr/bin/env python
"""udck19 norming data management and analysis script"""

import re
import logging
from pathlib import Path
import yaml
import numpy as np
import pandas as pd

# nlp tools
import spacy
from cmutools import cmuTools

# udck19 helpers
import udck19_filenames as fnames
import udck19_utils as utils

# check conda env is active
utils.check_ENV()

# set up logger
LOGGER = utils.get_udck19_logger(__file__)

# set up nlp tools
NLP = spacy.load('en_core_web_lg')
CMUT = cmuTools()  # cmu words, entries, phones loaded one time on init
if (NLP.meta is None or NLP.meta['name'] != 'core_web_lg'):
    msg = "spacy.load('en_core_web_lg') failed to load core_web_lg\n"
    msg += "make sure to: source activate udck19"
    raise RuntimeError(msg)
LOGGER.info(NLP.meta)


# helper functions
def merge_corrections(data):
    """create a column of screened single trial responses in table of norming data

    Parameters
    ----------
    data : pd.DataFrame
       table of single trial responses

    Returns
    -------
    data : pd.DataFrame
       original data with a new column data['screened']

    Notes:
    ------
    data['response'] and data['corrections'] must exist in the input

    data['screened'] must not exist in the input

    """

    assert 'response' in data.columns
    assert 'corrections' in data.columns
    assert 'screened' not in data.columns

    screened = data['response'].copy()
    for idx, row in data.iterrows():
        if (row['corrections']) != '':
            screened.iloc[idx] = row['corrections']
    data['screened'] = screened
    return data


def count_responses(responses):
    """calcluate spacy orthographic and lemma string counts by word position

    Parameters
    ----------
    responses : pd.Series of str
      screened response strings (length > 0), 'NA' designates missing response

    Returns
    -------
    results : dict
      keys are 'orth' and 'lemma'
      vals are nested dicts
          keys are response word position 0, 1, 2, ... up to length
            of longest response string.
          vals are key:val, where key is the string, val == integer count

    Notes
    -----
    Example output

    results = {
      orth:
        0: {a: 1, an: 24, apples: 3, healthy: 1, vitamins: 1}
        1: {apple: 24, tomato: 1}

      lemma:
        0: {a: 1, an: 24, apple: 3, healthy: 1, vitamin: 1}
        1: {apple: 24, tomato: 1}
    }
    """

    # run spacy on each response string
    docs = []
    for response in responses:
        docs.append(NLP(response))
    response_data = pd.DataFrame(docs)

    orth_counts = dict()
    lemma_counts = dict()
    for col in response_data.columns:
        # collect orthographic counts
        orth_counts[col] = (
            pd.Series(
                [
                    utils.get_token_attr(resp, 'orth_')
                    for resp in response_data[col]
                ]
            ).value_counts().to_dict()
        )

        # collect lemma counts
        lemma_counts[col] = (
            pd.Series(
                [
                    utils.get_token_attr(resp, 'lemma_')
                    for resp in response_data[col]
                ]
            ).value_counts().to_dict()
        )

    results = dict(
        orth=orth_counts,
        lemma=lemma_counts,
    )

    return results


def count_dict_to_df(count_dict):
    """convert dict of token counts by position to pd.DataFrame occurrence matrix

    Includes missing data counts, number of non-missing responses.

    Parameters
    ----------
    count_dict : dict
       where each `key` is a string and each `val` is an integer count

    Returns
    -------
    count_data : pd.DataFrame
      an occurence matrix, row (string counts)  x columns (position) and
      count_data.index contains all the count_dict.keys()

    Example
    -------

    Orthographic string counts

    count_dict =

    {0: {'NA': 1, 'apple': 30, 'aspirin': 1},
     1: {'None': 13, 'a': 15, 'every': 1, 'everyday': 1, 'per': 1, 'tree': 1},
     2: {'None': 15, 'day': 17}}


    count_data =

               0   1   2
    NA         1   0   0
    a          0  15   0
    apple     30   0   0
    aspirin    1   0   0
    day        0   0  17
    every      0   1   0
    everyday   0   1   0
    per        0   1   0
    tree       0   1   0

    """

    # strings with no occurrences in a column are nan in pandas,
    # override -> position get 0 counts
    count_data = pd.DataFrame(count_dict).fillna(value=0).astype(int)

    # check marginal count sums across columns
    n_responses = count_data[0].sum()
    for jdx in count_data.columns:
        if not count_data[jdx].sum() == n_responses:
            msg = (
                'first word response counts {0} do not match '
                'counts word position {1}: {2}'
            )
            msg = msg.format(jdx, n_responses, count_data[jdx].sum())
            LOGGER.error(msg)
            raise ValueError(msg)

    # validate and count NAs ... retain these to track missing
    # response data counts
    if 'NA' in count_data.index:
        # NAs, if any should only occur at 1st position in response
        if len(count_data.columns) > 1:
            if not all(count_data.loc['NA'][-1:] == 0):
                msg = 'NA vals beyond position 0: {0}'.format(count_data)
                LOGGER.error(msg)
                raise ValueError(msg)

    # validate then drop None counts
    if 'None' in count_data.index:
        # as token position increases
        if not count_data.loc['None'].is_monotonic_increasing:
            msg = (
                '"None" token counts not monotonic increasing '
                f'with position {count_data}'
            )
            LOGGER.error(msg)
            raise ValueError(msg)

        # all good so drop the None
        count_data.drop('None', inplace=True)

    return count_data


def count_topics(data):
    """look up attributes of first NOUN or VERB "topic" in item_id reponses.

    This is only meant to run on norming items *without* a supplied article.
    Results are tidied up from pandas to pure python for clean yamilzation


    Parameter
    ---------
    data : pd.Dataframe
       row-slice of norming responses data frame for a single item_id, e.g.

    Returns
    -------
    results: dict
        keys are topics from token.orth_, values are attributes


    Example
    -------

    df.loc['item_id i001_2_NA_NA_NA'] has 30 rows of responses

    results =

      {'NA': {'cmu_symbol': 'NA',
              'count': 1,
              'cv_initial': 'NA',
              'position_count': {0: 1}},
       'apple': {'cmu_symbol': 'AE',
                 'count': 23,
                 'cv_initial': 'v',
                 'position_count': {1: 23}},
       'apples': {'cmu_symbol': 'AE',
                  'count': 5,
                  'cv_initial': 'v',
                  'position_count': {0: 5}},
       'noodle': {'cmu_symbol': 'N',
                  'count': 1,
                  'cv_initial': 'c',
                  'position_count': {1: 1}},
       'vegetables': {'cmu_symbol': 'V',
                      'count': 2,
                      'cv_initial': 'c',
                      'position_count': {1: 1, 2: 1}}}

    """

    # this is only run on article norming items
    item_id = data.index.unique()
    assert len(item_id) == 1
    item_id = item_id[0]
    assert re.match(r'i\d{3}_[12]_NA_NA_NA', item_id) is not None

    topics = []
    data_column = 'screened'
    for idx, resp in data.iterrows():

        # find the topic as NOUN, VERB, fallback is first tok
        stim_doc, resp_doc, topic = None, None, None
        stim_doc = NLP(resp['stim'])

        # try with the sentence context
        resp_doc = NLP(resp['stim'] + ' ' + resp[data_column])
        n_words = len(stim_doc)  # whatever spaCy parse comes up with for stim
        topic, topic_idx = None, None

        for tok in resp_doc[n_words + 1:]:
            if tok.pos_ in ['NOUN', 'VERB']:
                topic = tok
                topic_idx = tok.i - n_words  # position in response of topic
                break

        # first fallback is parse the response alone
        if topic is None:
            resp_doc = NLP(resp[data_column])
            for tok in resp_doc:
                if tok.pos_ in ['NOUN', 'VERB']:
                    topic = tok
                    topic_idx = tok.i  # since just parsing the response
                    break

        # if still no luck final fallback is with first token not
        # including a, an ... often a single word response
        if topic is None:
            if resp_doc[0].orth_.lower() not in ['a', 'an']:
                topic = resp_doc[0]
                topic_idx = 0
            else:
                topic = resp_doc[1]
                topic_idx = 1

        assert topic is not None

        # override ... spaCy.pos_ and cmu for 'NA'
        pos = topic.pos_ if topic.orth_ != 'NA' else 'NA'

        # lookup pronounciation, skip NAs
        cv_initial, cmu_symbol = None, None
        if topic.orth_ != 'NA':
            # homegrown wrapper for cmu pronounciations
            cv_initial, cmu_symbol = CMUT.get_initial_phone(topic.orth_)
        else:
            cv_initial, cmu_symbol = 'NA', 'NA'

        topics.append((topic.orth_, pos, topic_idx, cv_initial, cmu_symbol))

    # convert to data frame for easy counting
    topics_df = pd.DataFrame(
        topics,
        columns=['topic', 'pos', 'idx', 'cv_initial', 'cmu_symbol']
    )

    topics_df.set_index('topic', inplace=True)
    topics_dict = {}
    for topic in topics_df.index.unique():
        topic_dict = {}
        # count each found topic
        resps = topics_df.loc[topic]

        # single rows come back Series column, force to wide DataFrame
        if isinstance(resps, pd.Series):
            resps = pd.DataFrame(resps).T

        # these should be unique
        for col in ['cv_initial', 'cmu_symbol']:
            this_val = resps[col].unique()
            assert len(this_val) == 1
            topic_dict[col] = str(this_val[0])

        # these may not be ...
        counts = resps['idx'].value_counts()
        topic_dict['position_count'] = dict(
            [(key, int(val)) for key, val in counts.items()]
        )

        topic_dict['count'] = int(sum(counts))

        topics_dict[topic] = topic_dict

    # return a dict for yaml dump
    return(topics_dict)


def get_topic_initial_phones(topics_dict):
    # convert norm dict to data frame and drop any NA rows from the counts

    if 'NA' in topics_dict.keys():
        n_NA = topics_dict['NA']['count']
    else:
        n_NA = 0
    topics_df = pd.DataFrame.from_dict(topics_dict, orient='index')
    topics_df.drop('NA', inplace=True, errors='ignore')
    topics_df.set_index('cv_initial', inplace=True)
    # print(topics_df)

    n_consonants = (
        int(topics_df.loc['c', 'count'].sum())
        if 'c' in topics_df.index
        else 0
    )

    n_vowels = (
        int(topics_df.loc['v', 'count'].sum())
        if 'v' in topics_df.index
        else 0
    )

    result = dict(
        n_consonants=n_consonants,
        n_vowels=n_vowels,
        n_NA=n_NA
    )

    return result


def get_normalized_entropy(counts_vec):
    """vaguely normalized entropy-like quantity for an item_id response set

    normalized to range [0.0 ... 1.0] like so:

      denom = np.log(np.sum(counts)) # total number of responses [3,1] != [6,2]

    """

    # drop nan's and zeros
    # dropping zero counts is equivalent to convention 0 log(0) == 0
    counts = [
        int(count)
        for count in counts_vec
        if ~np.isnan(count) and count > 0
    ]

    n_counts = len(counts)
    assert n_counts >= 1

    denom = np.log(len(counts))
    probs = [count/np.sum(counts) for count in counts]
    assert np.isclose(sum(probs), 1.0)
    if denom == 0:
        entropy = 0.0
    else:
        entropy = (
            -1.0 * sum([((prob * np.log(prob)) / denom) for prob in probs])
        )
    return entropy


def calc_context_measures(counts_df):
    """workhorse computations of normative measures for one context item_id

    Parameters
    ----------
    counts_df : pd.DataFrame
      string occurrence matrix:
         * string counts (row) x position (column)
         * row indexed by the strings.

    Returns
    -------
    results : dict {key:val, key:val, ...}
       where key is the measure name, and val is the value.

    Notes
    -----
    The measure values are Python scalars or lists of scalars

    """

    # 1. drop missing responses if any, subsequent calcs on actual responses
    if 'NA' in counts_df.index:
        n_NAs = int(counts_df.loc['NA', 0])
        counts_df = counts_df.drop('NA')
    else:
        n_NAs = int(0)

    #  exclude place holder 'None' counts for empty cells in occurence matrix
    if 'None' in counts_df.index:
        counts_df = counts_df.drop('None')

    # number of non-missing responses in response set
    n_responses = int(counts_df[0].sum())

    # 2. calculate modal measures ...

    # first token in the responses, i.e. first column of the df
    initial_counts = counts_df[0]
    initial_count_max = initial_counts.max()
    modal_initial = sorted(counts_df.index[counts_df[0] == initial_count_max].tolist())
    modal_initial_cloze = np.round(initial_count_max / n_responses, 3)
    modal_initial_character_classes = [
        utils.get_initial_character_class(string)
        for string in modal_initial
    ]

    # token in any position, i.e., row sum across columns in the df
    anywhere_counts = counts_df.sum(1)  # axis=1 -> row sum

    # most common word, any position
    anywhere_count_max = anywhere_counts.max()

    modal_anywhere = (
        sorted(
            counts_df
            .index[anywhere_counts == anywhere_count_max]
            .tolist()
        )
    )

    # all response strings
    modal_anywhere_cloze = (
        np.round(anywhere_count_max / counts_df.sum(0).sum(), 3)
    )

    # 3. calculate normalized entropy-like info measures ... the sign of
    # info tracks w/ constraint

    # information-based constraint for next word
    initial_entropy = get_normalized_entropy(initial_counts)
    context_initial_info = np.round(1.0 - initial_entropy, 3)

    # entire set of response strings, all all positions
    all_counts = [count for count in counts_df.values.flatten()]
    n_strings = int(sum(all_counts))  # add up all non-nan string counts
    all_entropy = get_normalized_entropy(all_counts)
    context_anywhere_info = np.round(1.0 - all_entropy, 3)

    # 4. strip numpy wrappers for clean YAML
    measures = dict(
        n_NAs=int(n_NAs),

        n_responses=int(n_responses),
        modal_initial=[str(s) for s in modal_initial],
        modal_initial_cloze=float(modal_initial_cloze),
        modal_initial_character_classes=[
            str(s) for s in modal_initial_character_classes
        ],
        n_strings=int(n_strings),
        modal_anywhere=[str(s) for s in modal_anywhere],
        modal_anywhere_cloze=float(modal_anywhere_cloze),

        context_initial_info=float(context_initial_info),
        context_anywhere_info=float(context_anywhere_info)
    )

    return measures


def response_txt_to_norm_measures(
        norm_expt_specs_f, responses_txt_f, norm_measures_yaml_f
):
    """compute response counts from corrected single trial TSV response file

    Parameters
    ----------
    responses_text_f : string
       path to tabular text file with single trial stim, response rows

    norm_measures_yaml_f : string
       file path to yaml file with the calculated counts and normative measures

    """

    screened = pd.read_csv(responses_txt_f, sep='\t', keep_default_na=False)

    with open(norm_expt_specs_f, 'r') as stream:
        norm_expt_specs = yaml.load(stream, Loader=yaml.SafeLoader)

    # 0. lookup and drop excluded Ss
    bad_sub_ids = []
    bad_item_ids = []
    for norm, specs in norm_expt_specs.items():
        for sub_id, reason in specs['subject_exclusions'].items():
            bad_sub_ids += [sub_id]
            msg = 'dropping {0} sub_id {1}: {2}'
            msg = msg.format(norm, sub_id, reason)
            LOGGER.info(msg)

        for item_id, reason in specs['item_exclusions'].items():
            bad_item_ids += [item_id]
            msg = 'dropping {0} item_id {1}: {2}'
            msg = msg.format(norm, item_id, reason)
            LOGGER.info(msg)

    # drop subjects and items
    screened = screened.set_index('subject_id').drop(bad_sub_ids)
    screened = screened.set_index('item_id').drop(bad_item_ids)
    screened.reset_index(inplace=True)

    # 1. merge corrected item columns
    corrected = merge_corrections(screened)
    corrected.set_index('item_id', inplace=True)
    corrected.sort_index(inplace=True)

    item_counts = []
    item_ids = sorted(corrected.index.unique())
    n_items = len(item_ids)

    for ith, item_id in enumerate(item_ids):
        LOGGER.info('Counting {0} {1}/{2}'.format(item_id, ith+1, n_items))
        item_id_vals = utils.parse_item_id(item_id, as_type='values')

        # slice this item_id responses out of main dataframe
        data = corrected.loc[item_id]
        expt_id = data['expt_id'].unique()
        assert len(expt_id) == 1

        list_id = data['stim_list_id'].unique()
        assert len(list_id) == 1

        # hack the the trailing article back in if there is one ...
        suffix = (
            ' ' + item_id_vals['article']
            if not item_id_vals['article'] is None
            else ''
        )

        stim = [stem + suffix for stem in data['stim'].unique()]
        assert len(stim) == 1
        stim = stim[0]

        # data = corrected[corrected['item_id'] == item_id]
        this_item = {
            'expt_id': expt_id[0],
            'item_id': item_id,
            'stim': stim,
            'stim_list_id': list_id[0],
         }

        # merge the response count dicts
        for key, val in count_responses(data['screened']).items():
            this_item[key] = val

        # compute and merge the context measures
        orth_count_df = count_dict_to_df(this_item['orth'])
        lemma_count_df = count_dict_to_df(this_item['lemma'])

        orth_measures = calc_context_measures(orth_count_df)
        lemma_measures = calc_context_measures(lemma_count_df)

        # count topics for norming without supplied article only
        if re.match(r'i\d{3}_[12]_NA_NA_NA', item_id) is not None:
            topic_counts = count_topics(data)
            topic_measures = get_topic_initial_phones(topic_counts)
        else:
            topic_counts = {}
            topic_measures = {}
        this_item['topic'] = topic_counts

        # add the measures
        this_item.update(
            {
                'context_measures': {
                    'orth': {key: val for key, val in orth_measures.items()},
                    'lemma': {key: val for key, val in lemma_measures.items()},
                    'topic': {key: val for key, val in topic_measures.items()}
                }
            }
        )

        # capture the bundle
        item_counts.append(this_item)

    # save the norming measures as YAML and verify it round trips in python
    with open(norm_measures_yaml_f, 'w') as counts:
        hdr_str = f"# Generated by {__file__} do not edit\n"
        yaml_s = yaml.dump(
            item_counts,
            default_flow_style=False,
            explicit_start=True
        )
        counts.write(hdr_str + yaml_s)

    # check list of dict -> YAML -> list of dict round trip
    # load counts from yaml file
    with open(norm_measures_yaml_f, 'r') as counts:
        item_counts_2 = yaml.load(counts.read(), yaml.SafeLoader)

    yaml_round_trip = (item_counts == item_counts_2)
    msg = 'list of count dicts -> YAML -> list of count dicts round trip: {0}'
    msg = msg.format(yaml_round_trip)
    LOGGER.info(msg)
    assert item_counts == item_counts_2
    del item_counts_2

    return None


if __name__ == '__main__':

    # read raw data and write measures defined above to YAML file
    response_txt_to_norm_measures(
        fnames.NORM_EXPT_SPECS_F,
        fnames.FINAL_SCREENED_NORMS_F,
        fnames.NORM_MEASURES_YAML_F
    )
