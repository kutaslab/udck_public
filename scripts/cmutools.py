# class and methods to fetch initial phoneme from a word using CMU Pronouncing dict
#
# usage: 
# >>> from cmutools import cmuTools
# >>> cmut = cmuTools()
# >>> first_phone_type, first_phon =  cmut.get_initial_phoneme('apple')
# 
# >>> first_phone_type, ('v', 'AE')
#
# 
# CMUdict
# -------

# CMUdict (the Carnegie Mellon Pronouncing Dictionary) is a free
# pronouncing dictionary of English, suitable for uses in speech
# technology and is maintained by the Speech Group in the School of
# Computer Science at Carnegie Mellon University.

# The Carnegie Mellon Speech Group does not guarantee the accuracy of
# this dictionary, nor its suitability for any specific purpose. In
# fact, we expect a number of errors, omissions and inconsistencies to
# remain in the dictionary. We intend to continually update the
# dictionary by correction existing entries and by adding new ones. From
# time to time a new major version will be released.

# We welcome input from users: Please send email to Alex Rudnicky
# (air+cmudict@cs.cmu.edu).

# The Carnegie Mellon Pronouncing Dictionary, in its current and
# previous versions is Copyright (C) 1993-2014 by Carnegie Mellon
# University.  Use of this dictionary for any research or commercial
# purpose is completely unrestricted.  If you make use of or
# redistribute this material we request that you acknowledge its
# origin in your descriptions.

# If you add words to or correct words in your version of this
# dictionary, we would appreciate it if you could send these additions
# and corrections to us (air+cmudict@cs.cmu.edu) for consideration in a
# subsequent version. All submissions will be reviewed and approved by
# the current maintainer, Alex Rudnicky at Carnegie Mellon.

# https://github.com/prosegrinder/python-cmudict  09/09/18

import re
import cmudict
class cmuTools(object):
    
    def __init__(self):
        """install cmudict on init"""
        self.cmu_phones = cmudict.phones()
        self.cmu_entries = cmudict.entries()
        self.cmu_words = cmudict.words()
        
        self._set_vowel_types()
        
    def _set_vowel_types(self):
    
        # tag cmu phones into vowels and consonants for A/An alternation
        self.vowel_types = ['vowel']
        self.consonant_types = ['stop', 'affricate', 'fricative', 'aspirate', 'nasal', 'liquid', 'semivowel']
        
        self.cmu_vowels, self.cmu_consonants = [], []
        for symbol, phone_type in self.cmu_phones:
            if phone_type[0] in self.vowel_types:
                self.cmu_vowels += [symbol]
            elif phone_type[0] in self.consonant_types:
                self.cmu_consonants += [symbol]
            else:
                msg = '{0} {1} not in CMU phones'.format(symbol, phone_type)
                raise ValueError(msg)

    def get_initial_phone(self, word):
        """"""
        first_phone = None
        first_phone_type = None
        
        if word == 'NA' or word is None:
            first_phone_type, first_phone
        
        # or look it up it     
        try:
            idx = self.cmu_words.index(word)
            first_phone = self.cmu_entries[idx][1][0]
            first_phone = re.sub(r'\d+', '', first_phone) # strip stress indicator
            if first_phone in self.cmu_vowels:
                first_phone_type = 'v'
            else:
                first_phone_type = 'c'
            
        except ValueError:
        
            # word not found in cmdict
            msg = '{0} not found in cmu_words, falling back to orthography'
            msg = msg.format(word)
            first_phone = 'cmu_oov'
            if word[0] in ['a', 'e', 'i', 'o', 'u']:
                first_phone_type = 'v'
            else:
                first_phone_type = 'c'
            
        return first_phone_type, first_phone

