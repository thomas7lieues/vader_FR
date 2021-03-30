# coding: utf-8
# Author: C.J. Hutto
# Thanks to George Berry for reducing the time complexity from something like O(N^4) to O(N).
# Thanks to Ewan Klein and Pierpaolo Pantone for bringing VADER into NLTK. Those modifications were awesome.
# For license information, see LICENSE.TXT

"""
If you use the VADER sentiment analysis tools, please cite:
Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based Model for
Sentiment Analysis of Social Media Text. Eighth International Conference on
Weblogs and Social Media (ICWSM-14). Ann Arbor, MI, June 2014.
"""
import os
import re
import math
import string
import codecs
import json
from itertools import product
from inspect import getsourcefile
from io import open
from fuzzywuzzy import fuzz
import logging
import unidecode
import sys
sys.path.append('/')
import vaderSentiment_fr.tree as tree

logger = logging.getLogger(__name__)

# ##Constants##
# nombre minimum de lettres à vérifier pour fuzzywuzzy
nb_lettres_min=3

# Pourcentage minimal pour considérer que les mots se ressemblent
seuil_ratio = 85

# (empirically derived mean sentiment intensity rating increase for booster words)
B_INCR = 0.293
B_DECR = -0.293

# (empirically derived mean sentiment intensity rating increase for using ALLCAPs to emphasize a word)
C_INCR = 0.733
N_SCALAR = -0.74

NEGATE = \
    ["pas", "jamais", "plus", "rien", "aucunement",
     "nullement", "sans", "ni"]

# booster/dampener 'intensifiers' or 'degree adverbs'
# http://en.wiktionary.org/wiki/Category:English_degree_adverbs

BOOSTER_DICT = \
    {"absolument": B_INCR, "ainsi": B_INCR, "archi": B_INCR, "beaucoup": B_INCR,
     "bigrement": B_INCR, "bougrement": B_INCR, "carrément": B_INCR, "complètement": B_INCR,
     "considérablement": B_INCR, "davantage": B_INCR, "diablement": B_INCR, "diantrement": B_INCR,
     "divinement": B_INCR, "drôlement": B_INCR, "délicieusement": B_INCR, "entièrement": B_INCR,
     "exceptionnel": B_INCR, "exceptionnelle": B_INCR, "exceptionnellement": B_INCR, "exceptionnelles": B_INCR,
     "exceptionnels": B_INCR, "excessivement": B_INCR, "extra": B_INCR, "extrême": B_INCR,
     "extrêmement": B_INCR, "fabuleusement": B_INCR, "fichtrement": B_INCR, "fort": B_INCR,
     "forte": B_INCR, "fortes": B_INCR, "forts": B_INCR, "grandement": B_INCR,
     "hyper": B_INCR, "impeccablement": B_INCR, "incroyablement": B_INCR, "infiniment": B_INCR,
     "joliment": B_INCR, "merveilleusement": B_INCR, "prodigieusement": B_INCR, "profondément": B_INCR,
     "putain de": B_INCR, "rudement": B_INCR, "sacrément": B_INCR, "spécialement": B_INCR,
     "sublimement": B_INCR, "super": B_INCR, "superbement": B_INCR, "tant": B_INCR,
     "tellement": B_INCR, "terriblement": B_INCR, "totalement": B_INCR, "trop": B_INCR,
     "très": B_INCR, "ultra": B_INCR, "vachement": B_INCR, "vraiment": B_INCR,
     "énormément": B_INCR, "approximativement": B_DECR, "assez": B_DECR, "difficilement": B_DECR,
     "environ": B_DECR, "guère": B_DECR, "insuffisament": B_DECR, "insuffisamment": B_DECR,
     "juste": B_DECR, "léger": B_DECR, "légers": B_DECR, "légère": B_DECR,
     "légèrement": B_DECR, "légères": B_DECR, "moins": B_DECR, "peu": B_DECR,
     "plutôt": B_DECR, "presque": B_DECR, "quasi": B_DECR, "quasiment": B_DECR,
     "quelque": B_DECR, "quelque peu": B_DECR, "rare": B_DECR, "rarement": B_DECR,
     "à peine": B_DECR, "à peu près": B_DECR}

# check for sentiment laden idioms that do not contain lexicon words (future work, not yet implemented)
SENTIMENT_LADEN_IDIOMS = {"cut the mustard": 2, "hand to mouth": -2,
                          "back handed": -2, "blow smoke": -2, "blowing smoke": -2,
                          "upper hand": 1, "break a leg": 2,
                          "cooking with gas": 2, "in the black": 2, "in the red": -2,
                          "on the ball": 2, "under the weather": -2}

# check for special case idioms and phrases containing lexicon words
SPECIAL_CASES = {"the shit": 3, "the bomb": 3, "bad ass": 1.5, "badass": 1.5, "bus stop": 0.0,
                 "yeah right": -2, "kiss of death": -1.5, "to die for": 3, "beating heart": 3.5}



# Mots à ne pas nier
_this_module_file_path_ = os.path.abspath(getsourcefile(lambda: 0))
no_negate_full_filepath = os.path.join(
    os.path.dirname(_this_module_file_path_), "no_negate.txt")

with codecs.open(no_negate_full_filepath,'r') as f :
    NO_NEGATE = f.read().strip().split("\n")


# #Static methods# #

def negated(input_words, include_nt=True):
    """
    Determine if input contains negation words
    """
    input_words = [str(w).lower() for w in input_words]
    neg_words = []
    neg_words.extend(NEGATE)
    for word in neg_words:
        if word in input_words:
            return True
    if include_nt:
        for word in input_words:
            if "n't" in word:
                return True
    '''if "least" in input_words:
        i = input_words.index("least")
        if i > 0 and input_words[i - 1] != "at":
            return True'''
    return False

def normalize(score, alpha=15):
    """
    Normalize the score to be between -1 and 1 using an alpha that
    approximates the max expected value
    """
    norm_score = score / math.sqrt((score * score) + alpha)
    if norm_score < -1.0:
        return -1.0
    elif norm_score > 1.0:
        return 1.0
    else:
        return norm_score



def allcap_differential(words):
    """
    Check whether just some words in the input are ALL CAPS
    :param list words: The words to inspect
    :returns: `True` if some but not all items in `words` are ALL CAPS
    """
    is_different = False
    allcap_words = 0
    for word in words:
        if word.isupper():
            allcap_words += 1
    cap_differential = len(words) - allcap_words
    if 0 < cap_differential < len(words):
        is_different = True
    return is_different


def scalar_inc_dec(word, valence, is_cap_diff):
    """
    Check if the preceding words increase, decrease, or negate/nullify the
    valence
    """
    scalar = 0.0
    word_lower = word.lower()
    if word_lower in BOOSTER_DICT:
        scalar = BOOSTER_DICT[word_lower]
        if valence < 0:
            scalar *= -1
        # check if booster/dampener word is in ALLCAPS (while others aren't)
        if word.isupper() and is_cap_diff:
            if valence > 0:
                scalar += C_INCR
            else:
                scalar -= C_INCR
    return scalar


def word_is_uppercase(word):
    characters = list(word)

    for character in characters:
        if character.islower():
            return False
    return True


class SentiText(object):
    """
    Identify sentiment-relevant string-level properties of input text.
    """

    def __init__(self, text):
        if not isinstance(text, str):
            text = str(text).encode('utf-8')
        self.text = text
        self.words_and_emoticons = self._words_and_emoticons()
        # doesn't separate words from\
        # adjacent punctuation (keeps emoticons & contractions)
        self.is_cap_diff = allcap_differential(self.words_and_emoticons)

    @staticmethod
    def _strip_punc_if_word(token):
        """
        Removes all trailing and leading punctuation
        If the resulting string has two or fewer characters,
        then it was likely an emoticon, so return original string
        (ie ":)" stripped would be "", so just return ":)"
        """
        stripped = token.strip(string.punctuation)
        if len(stripped) <= 2:
            return token
        return stripped

    def _words_and_emoticons(self):
        """
        Removes leading and trailing puncutation
        Leaves contractions and most emoticons
            Does not preserve punc-plus-letter emoticons (e.g. :D)
        """
        wes = re.split(r"[\s']", self.text)
        stripped = list(map(self._strip_punc_if_word, wes))
        return stripped


class SentimentIntensityAnalyzer(object):
    """
    Give a sentiment intensity score to sentences.
    """

    def __init__(self, lexicon_file="fr_lexicon.txt", emoji_lexicon="emojis_fr.txt"):
        _this_module_file_path_ = os.path.abspath(getsourcefile(lambda: 0))
        lexicon_full_filepath = os.path.join(
            os.path.dirname(_this_module_file_path_), lexicon_file)
        with codecs.open(lexicon_full_filepath, encoding='utf-8') as f:
            self.lexicon_full_filepath = f.read()
        self.lexicon = self.make_lex_dict()

        emoji_full_filepath = os.path.join(
            os.path.dirname(_this_module_file_path_), emoji_lexicon)
        with codecs.open(emoji_full_filepath, encoding='utf-8') as f:
            self.emoji_full_filepath = f.read()
        self.emojis = self.make_emoji_dict()
        self.treenode = self.make_tree()

        

    def make_tree(self):
        words = list(self.lexicon.keys())
        treenode= tree.TrieNode("*")
        for word in words :
            tree.add(treenode,word)

        return treenode

    def get_max_words(self, item):
        # On vérifie que le nombre de lettres minimum est atteint, sinon on retourne None
        if len(item)<=nb_lettres_min:
            return None

        # On retire les accetns
        item = unidecode.unidecode(item)
        max_word = ""
        max_ratio = 0

        # On parcourt le lexicon pour trouver le mot le plus similaire
        for word, polarity in self.lexicon.items():
            word = unidecode.unidecode(word)
            ratio = fuzz.ratio(word, item)
            if ratio > max_ratio:
                max_ratio = ratio
                max_word = word
        if max_ratio >= seuil_ratio:
            return max_word
        return None

    def make_lex_dict(self):
        """
        Convert lexicon file to a dictionary
        """
        lex_dict = {}
        for line in self.lexicon_full_filepath.rstrip('\n').split('\n'):
            if not line:
                continue
            (word, measure) = line.strip().split('->')[0:2]
            lex_dict[unidecode.unidecode(word)] = float(measure) # On utilise unidecode pour supprimer les accents
        return lex_dict

    def make_emoji_dict(self):
        """
        Convert emoji lexicon file to a dictionary
        """
        emoji_dict = {}
        for line in self.emoji_full_filepath.rstrip('\n').split('\n'):
            (emoji, description) = line.strip().split('\t')[0:2]
            emoji_dict[emoji] = description
        return emoji_dict

    def polarity_scores(self, text):
        """
        Return a float for sentiment strength based on the input text.
        Positive values are positive valence, negative value are negative
        valence.
        """
        # convert emojis to their textual descriptions
        text_no_emoji = ""
        prev_space = True
        for chr in text:
            if chr in self.emojis:
                # get the textual description
                description = self.emojis[chr]
                if not prev_space:
                    text_no_emoji += ' '
                text_no_emoji += description
                prev_space = False
            else:
                text_no_emoji += chr
                prev_space = chr == ' '
        text = text_no_emoji.strip()

        sentitext = SentiText(text)

        sentiments = []
        words_and_emoticons = sentitext.words_and_emoticons
        for i, item in enumerate(words_and_emoticons):
            valence = 0
            # check for vader_lexicon words that may be used as modifiers or negations
            if item.lower() in BOOSTER_DICT:
                sentiments.append(valence)
                continue
            if (i < len(words_and_emoticons) - 1 and item.lower() == "kind" and
                    words_and_emoticons[i + 1].lower() == "of"):
                sentiments.append(valence)
                continue

            sentiments = self.sentiment_valence(
                valence, sentitext, item, i, sentiments)

        # sentiments = self._but_check(words_and_emoticons, sentiments)

        valence_dict = self.score_valence(sentiments, text)
        # Création du dictionnaire pour obtenir à chaque mot la valeur attribuée

        return valence_dict

    def polarity_scores_max(self, text):
        """
        Return a float for sentiment strength based on the input text.
        Positive values are positive valence, negative value are negative
        valence.
        """
        # convert emojis to their textual descriptions
        text_no_emoji = ""
        prev_space = True
        for chr in text:
            if chr in self.emojis:
                # get the textual description
                description = self.emojis[chr]
                if not prev_space:
                    text_no_emoji += ' '
                text_no_emoji += description
                prev_space = False
            else:
                text_no_emoji += chr
                prev_space = chr == ' '
        text = text_no_emoji.strip()

        sentitext = SentiText(text)

        sentiments = []
        detected_words = [] # Mot détecté (utile si fuzzywuzzé)
        words_and_emoticons = sentitext.words_and_emoticons
        for i, item in enumerate(words_and_emoticons):
            valence = 0
            # check for vader_lexicon words that may be used as modifiers or negations
            if item.lower() in BOOSTER_DICT:
                sentiments.append(valence)
                detected_words.append(item)
                continue
            if (i < len(words_and_emoticons) - 1 and item.lower() == "kind" and
                    words_and_emoticons[i + 1].lower() == "of"):
                sentiments.append(valence)
                detected_words.append(item)
                continue

            sentiments, detected_words = self.sentiment_valence_max(
                valence, sentitext, item, i, sentiments, detected_words)

        sentiments = self._but_check(words_and_emoticons, sentiments)

        valence_dict = self.score_valence(sentiments, text)
        details = {}
        for i in range(len(sentiments)):
            if detected_words[i] is not None :
                details[words_and_emoticons[i]+"/"+str(detected_words[i])]=sentiments[i]
            else :
                details[words_and_emoticons[i]]=sentiments[i]
        valence_dict["details"] = details

        return valence_dict



    def sentiment_valence(self, valence, sentitext, item, i, sentiments):
        is_cap_diff = sentitext.is_cap_diff
        words_and_emoticons = sentitext.words_and_emoticons
        item_lowercase = unidecode.unidecode(item.lower())
        if item_lowercase in self.lexicon:
            # get the sentiment valence
            valence = self.lexicon[item_lowercase]

            # check for "no" as negation for an adjacent lexicon item vs "no" as its own stand-alone lexicon item
            if item_lowercase == "no" and i != len(words_and_emoticons)-1 and words_and_emoticons[i + 1].lower() in self.lexicon:
                # don't use valence of "no" as a lexicon item. Instead set it's valence to 0.0 and negate the next item
                valence = 0.0
            if (i > 0 and words_and_emoticons[i - 1].lower() == "no") \
               or (i > 1 and words_and_emoticons[i - 2].lower() == "no") \
               or (i > 2 and words_and_emoticons[i - 3].lower() == "no" and words_and_emoticons[i - 1].lower() in ["or", "nor"]):
                valence = self.lexicon[item_lowercase] * N_SCALAR

            # check if sentiment laden word is in ALL CAPS (while others aren't)
            if item.isupper() and is_cap_diff:
                if valence > 0:
                    valence += C_INCR
                else:
                    valence -= C_INCR

            for start_i in range(0, 3):
                # dampen the scalar modifier of preceding words and emoticons
                # (excluding the ones that immediately preceed the item) based
                # on their distance from the current item.
                if i > start_i and words_and_emoticons[i - (start_i + 1)].lower() not in self.lexicon:
                    s = scalar_inc_dec(
                        words_and_emoticons[i - (start_i + 1)], valence, is_cap_diff)
                    if start_i == 1 and s != 0:
                        s = s * 0.95
                    if start_i == 2 and s != 0:
                        s = s * 0.9
                    valence = valence + s
                    valence = self._negation_check(
                        valence, words_and_emoticons, start_i, i)
                    if start_i == 2:
                        valence = self._special_idioms_check(
                            valence, words_and_emoticons, i)

            valence = self._least_check(valence, words_and_emoticons, i)
        sentiments.append(valence)
        return sentiments

    def sentiment_valence_max(self, valence, sentitext, item, i, sentiments, detected_words):
        is_cap_diff = sentitext.is_cap_diff
        words_and_emoticons = sentitext.words_and_emoticons
        item_lowercase = unidecode.unidecode(item.lower())
        # tree.find_prefix(treenode, "connzz")
        if not tree.find_word(self.treenode, item_lowercase):
            item_lowercase = self.get_max_words(item_lowercase)
        # get the sentiment valence
        if item_lowercase is not None :
            valence = self.lexicon[item_lowercase]

            # check for "no" as negation for an adjacent lexicon item vs "no" as its own stand-alone lexicon item
            if item_lowercase == "pas" and i != len(words_and_emoticons)-1 and words_and_emoticons[i + 1].lower() in self.lexicon:
                    # don't use valence of "no" as a lexicon item. Instead set it's valence to 0.0 and negate the next item
                valence = 0.0
            if (i > 0 and words_and_emoticons[i - 1].lower() == "no") \
               or (i > 1 and words_and_emoticons[i - 2].lower() == "no") \
               or (i > 2 and words_and_emoticons[i - 3].lower() == "no" and words_and_emoticons[i - 1].lower() in ["pzpzp", "nor"]):
                valence = self.lexicon[item_lowercase] * N_SCALAR

            # check if sentiment laden word is in ALL CAPS (while others aren't)
            if item.isupper() and is_cap_diff:
                if valence > 0:
                    valence += C_INCR
                else:
                    valence -= C_INCR

            for start_i in range(0, 3):
                # dampen the scalar modifier of preceding words and emoticons
                # (excluding the ones that immediately preceed the item) based
                # on their distance from the current item.
                if i > start_i and words_and_emoticons[i - (start_i + 1)].lower() not in self.lexicon:
                    s = scalar_inc_dec(
                        words_and_emoticons[i - (start_i + 1)], valence, is_cap_diff)
                    if start_i == 1 and s != 0:
                        s = s * 0.95
                    if start_i == 2 and s != 0:
                        s = s * 0.9
                    valence = valence + s
                    valence = self._negation_check(
                        valence, words_and_emoticons, start_i, i)
                    if start_i == 2:
                        valence = self._special_idioms_check(
                            valence, words_and_emoticons, i)

            valence = self._least_check(valence, words_and_emoticons, i)
        sentiments.append(valence)
        detected_words.append(item_lowercase)
        return sentiments, detected_words

    def _least_check(self, valence, words_and_emoticons, i):
        # check for negation case using "least"
        if i > 1 and words_and_emoticons[i - 1].lower() not in self.lexicon \
                and words_and_emoticons[i - 1].lower() == "least":
            if words_and_emoticons[i - 2].lower() != "at" and words_and_emoticons[i - 2].lower() != "very":
                valence = valence * N_SCALAR
        elif i > 0 and words_and_emoticons[i - 1].lower() not in self.lexicon \
                and words_and_emoticons[i - 1].lower() == "least":
            valence = valence * N_SCALAR
        return valence

    @staticmethod
    def _but_check(words_and_emoticons, sentiments):
        # check for modification in sentiment due to contrastive conjunction 'but'
        words_and_emoticons_lower = [str(w).lower()
                                     for w in words_and_emoticons]
        if 'mais' in words_and_emoticons_lower:
            bi = words_and_emoticons_lower.index('mais')
            for sentiment in sentiments:
                si = sentiments.index(sentiment)
                if si < bi:
                    sentiments.pop(si)
                    sentiments.insert(si, sentiment * 0.5)
                elif si > bi:
                    sentiments.pop(si)
                    sentiments.insert(si, sentiment * 1.5)
        return sentiments

    @staticmethod
    def _special_idioms_check(valence, words_and_emoticons, i):
        words_and_emoticons_lower = [str(w).lower()
                                     for w in words_and_emoticons]
        onezero = "{0} {1}".format(
            words_and_emoticons_lower[i - 1], words_and_emoticons_lower[i])

        twoonezero = "{0} {1} {2}".format(words_and_emoticons_lower[i - 2],
                                          words_and_emoticons_lower[i - 1], words_and_emoticons_lower[i])

        twoone = "{0} {1}".format(
            words_and_emoticons_lower[i - 2], words_and_emoticons_lower[i - 1])

        threetwoone = "{0} {1} {2}".format(words_and_emoticons_lower[i - 3],
                                           words_and_emoticons_lower[i - 2], words_and_emoticons_lower[i - 1])

        threetwo = "{0} {1}".format(
            words_and_emoticons_lower[i - 3], words_and_emoticons_lower[i - 2])

        sequences = [onezero, twoonezero, twoone, threetwoone, threetwo]

        for seq in sequences:
            if seq in SPECIAL_CASES:
                valence = SPECIAL_CASES[seq]
                break

        if len(words_and_emoticons_lower) - 1 > i:
            zeroone = "{0} {1}".format(
                words_and_emoticons_lower[i], words_and_emoticons_lower[i + 1])
            if zeroone in SPECIAL_CASES:
                valence = SPECIAL_CASES[zeroone]
        if len(words_and_emoticons_lower) - 1 > i + 1:
            zeroonetwo = "{0} {1} {2}".format(words_and_emoticons_lower[i], words_and_emoticons_lower[i + 1],
                                              words_and_emoticons_lower[i + 2])
            if zeroonetwo in SPECIAL_CASES:
                valence = SPECIAL_CASES[zeroonetwo]

        # check for booster/dampener bi-grams such as 'sort of' or 'kind of'
        n_grams = [threetwoone, threetwo, twoone]
        for n_gram in n_grams:
            if n_gram in BOOSTER_DICT:
                valence = valence + BOOSTER_DICT[n_gram]
        return valence

    @staticmethod
    def _sentiment_laden_idioms_check(valence, senti_text_lower):
        # Future Work
        # check for sentiment laden idioms that don't contain a lexicon word
        idioms_valences = []
        for idiom in SENTIMENT_LADEN_IDIOMS:
            if idiom in senti_text_lower:
                print(idiom, senti_text_lower)
                valence = SENTIMENT_LADEN_IDIOMS[idiom]
                idioms_valences.append(valence)
        if len(idioms_valences) > 0:
            valence = sum(idioms_valences) / float(len(idioms_valences))
        return valence

    @staticmethod
    def _negation_check(valence, words_and_emoticons, start_i, i):
        words_and_emoticons_lower = [str(w).lower()
                                     for w in words_and_emoticons]

        no_check = [unidecode.unidecode(elem) for elem in NO_NEGATE]

        # Si le mot n'est pas à nier
        if words_and_emoticons_lower[i] in no_check:
            return valence

        # Si le mot peut être nié
        if start_i == 0:
            # 1 word preceding lexicon word (w/o stopwords)
            if negated([words_and_emoticons_lower[i - (start_i + 1)]]):
                valence = valence * N_SCALAR
        if start_i == 1:
            # 2 words preceding the lexicon word position
            if negated([words_and_emoticons_lower[i - (start_i + 1)]]):
                valence = valence * N_SCALAR
        if start_i == 2:
            if negated([words_and_emoticons_lower[i - (start_i + 1)]]):
                valence = valence * N_SCALAR
        return valence

    def _punctuation_emphasis(self, text):
        # add emphasis from exclamation points and question marks
        ep_amplifier = self._amplify_ep(text)
        qm_amplifier = self._amplify_qm(text)
        sp_amplifier = self._amplify_sp(text)

        punct_emph_amplifier = ep_amplifier + qm_amplifier + sp_amplifier

        return punct_emph_amplifier

    @staticmethod
    def _amplify_uc(text_):
        text = text_

        # On supprime les ponctuations
        ponctuations = ["?", "!", ".", ",", ";"]
        for ponctuation in ponctuations:
            text = text.replace(ponctuation, "")

        # Découpage par mots
        liste = text.split(" ")
        words = liste.copy()

        # On supprime les mots que d'une lettre
        words = [word for word in words if len(word) > 1]

        # On supprime les premières lettres
        for i in range(len(words)):
            words[i] = words[i][1:len(words[i])]

        uc_count = 0
        for word in words:
            if word_is_uppercase(word):
                if uc_count > 4:
                    break
                uc_count += 1
        uc_amplifier = uc_count * 0.292
        return uc_amplifier

    @staticmethod
    def _amplify_sp(text):
        sp_coeff = 0.21

        # Va recenser tous les points de suspensions
        # ["...","..","....", ".""] par exemple
        reg_ = re.compile("\.{2,}")
        points = reg_.findall(text)

        # un string qui regroupe tous les points de suspension
        # ["...", ".."] -> "....."
        s = "".join(points)
        sp_count = len(s)
        if sp_count in [1, 0]:
            return 0.0
        if sp_count in [2, 3]:
            return 0.21
        if sp_count <= 6:
            return (sp_count-2)*sp_coeff
        sp_count = 6
        return (sp_count-2)*sp_coeff

    @staticmethod
    def _amplify_ep(text):
        # check for added emphasis resulting from exclamation points (up to 4 of them)
        ep_count = text.count("!")
        if ep_count > 4:
            ep_count = 4
        # (empirically derived mean sentiment intensity rating increase for
        # exclamation points)
            # Intensité du point d'exclamation
        ep_amplifier = ep_count * 0.292
        return ep_amplifier

    @staticmethod
    def _amplify_qm(text):
        # check for added emphasis resulting from question marks (2 or 3+)
        qm_count = text.count("?")
        qm_amplifier = 0
        if qm_count > 1:
            if qm_count <= 3:
                # (empirically derived mean sentiment intensity rating increase for
                # question marks)
                                # Intensité du point d'interrogation
                qm_amplifier = qm_count * 0.18
            else:
                qm_amplifier = 0.96
        return qm_amplifier

    @staticmethod
    def _sift_sentiment_scores(sentiments):
        # want separate positive versus negative sentiment scores
        pos_sum = 0.0
        neg_sum = 0.0
        neu_count = 0
        for sentiment_score in sentiments:
            if sentiment_score > 0:
                # compensates for neutral words that are counted as 1
                pos_sum += (float(sentiment_score) + 1)
            if sentiment_score < 0:
                # when used with math.fabs(), compensates for neutrals
                neg_sum += (float(sentiment_score) - 1)
            if sentiment_score == 0:
                neu_count += 1
        return pos_sum, neg_sum, neu_count

    def score_valence(self, sentiments, text):
        if sentiments:
            sum_s = float(sum(sentiments))
            # compute and add emphasis from punctuation in text
            punct_emph_amplifier = self._punctuation_emphasis(text)
            uc_amplifier = self._amplify_uc(text)

            if sum_s > 0:
                sum_s += punct_emph_amplifier
                if sum_s >= 0.05:
                    sum_s += uc_amplifier
            elif sum_s < 0:
                sum_s -= punct_emph_amplifier
                if sum_s <= -0.05:
                    sum_s -= uc_amplifier
            else:
                sum_s -= 0.5 * punct_emph_amplifier

            compound = normalize(sum_s)
            # discriminate between positive, negative and neutral sentiment scores
            pos_sum, neg_sum, neu_count = self._sift_sentiment_scores(
                sentiments)

            if pos_sum > math.fabs(neg_sum):
                pos_sum += punct_emph_amplifier
            elif pos_sum < math.fabs(neg_sum):
                neg_sum -= punct_emph_amplifier

            total = pos_sum + math.fabs(neg_sum) + neu_count
            pos = math.fabs(pos_sum / total)
            neg = math.fabs(neg_sum / total)
            neu = math.fabs(neu_count / total)

        else:
            compound = 0.0
            pos = 0.0
            neg = 0.0
            neu = 0.0

        sentiment_dict = \
            {"neg": round(neg, 3),
             "neu": round(neu, 3),
             "pos": round(pos, 3),
             "compound": round(compound, 4)}

        return sentiment_dict


if __name__ == '__main__':
    # --- examples -------
    sentences = ["VADER is smart, handsome, and funny.",  # positive sentence example
                 "VADER is smart, handsome, and funny!",
                 # punctuation emphasis handled correctly (sentiment intensity adjusted)
                 "VADER is very smart, handsome, and funny.",
                 # booster words handled correctly (sentiment intensity adjusted)
                 "VADER is VERY SMART, handsome, and FUNNY.",  # emphasis for ALLCAPS handled
                 "VADER is VERY SMART, handsome, and FUNNY!!!",
                 # combination of signals - VADER appropriately adjusts intensity
                 "VADER is VERY SMART, uber handsome, and FRIGGIN FUNNY!!!",
                 # booster words & punctuation make this close to ceiling for score
                 "VADER is not smart, handsome, nor funny.",  # negation sentence example
                 "The book was good.",  # positive sentence
                 # negated negative sentence with contraction
                 "At least it isn't a horrible book.",
                 "The book was only kind of good.",
                 # qualified positive sentence is handled correctly (intensity adjusted)
                 "The plot was good, but the characters are uncompelling and the dialog is not great.",
                 # mixed negation sentence
                 "Today SUX!",  # negative slang with capitalization emphasis
                 "Today only kinda sux! But I'll get by, lol",
                 # mixed sentiment example with slang and constrastive conjunction "but"
                 "Make sure you :) or :D today!",  # emoticons handled
                 "Catch utf-8 emoji such as 💘 and 💋 and 😁",  # emojis handled
                 "Not bad at all"  # Capitalized negation
                 ]

    analyzer = SentimentIntensityAnalyzer()

    print("----------------------------------------------------")
    print(" - Analyze typical example cases, including handling of:")
    print("  -- negations")
    print("  -- punctuation emphasis & punctuation flooding")
    print("  -- word-shape as emphasis (capitalization difference)")
    print("  -- degree modifiers (intensifiers such as 'very' and dampeners such as 'kind of')")
    print("  -- slang words as modifiers such as 'uber' or 'friggin' or 'kinda'")
    print("  -- contrastive conjunction 'but' indicating a shift in sentiment; sentiment of later text is dominant")
    print("  -- use of contractions as negations")
    print("  -- sentiment laden emoticons such as :) and :D")
    print("  -- utf-8 encoded emojis such as 💘 and 💋 and 😁")
    print("  -- sentiment laden slang words (e.g., 'sux')")
    print("  -- sentiment laden initialisms and acronyms (for example: 'lol') \n")
    for sentence in sentences:
        vs = analyzer.polarity_scores(sentence)
        print("{:-<65} {}".format(sentence, str(vs)))
    print("----------------------------------------------------")
    print(" - About the scoring: ")
    print("""  -- The 'compound' score is computed by summing the valence scores of each word in the lexicon, adjusted
     according to the rules, and then normalized to be between -1 (most extreme negative) and +1 (most extreme positive).
     This is the most useful metric if you want a single unidimensional measure of sentiment for a given sentence.
     Calling it a 'normalized, weighted composite score' is accurate.""")
    print("""  -- The 'pos', 'neu', and 'neg' scores are ratios for proportions of text that fall in each category (so these
     should all add up to be 1... or close to it with float operation).  These are the most useful metrics if
     you want multidimensional measures of sentiment for a given sentence.""")
    print("----------------------------------------------------")

    # input("\nPress Enter to continue the demo...\n")  # for DEMO purposes...

    tricky_sentences = ["Sentiment analysis has never been good.",
                        "Sentiment analysis has never been this good!",
                        "Most automated sentiment analysis tools are shit.",
                        "With VADER, sentiment analysis is the shit!",
                        "Other sentiment analysis tools can be quite bad.",
                        "On the other hand, VADER is quite bad ass",
                        "VADER is such a badass!",  # slang with punctuation emphasis
                        "Without a doubt, excellent idea.",
                        "Roger Dodger is one of the most compelling variations on this theme.",
                        "Roger Dodger is at least compelling as a variation on the theme.",
                        "Roger Dodger is one of the least compelling variations on this theme.",
                        "Not such a badass after all.",  # Capitalized negation with slang
                        # "without {any} doubt" as negation
                        "Without a doubt, an excellent idea."
                        ]
    print("----------------------------------------------------")
    print(" - Analyze examples of tricky sentences that cause trouble to other sentiment analysis tools.")
    print("  -- special case idioms - e.g., 'never good' vs 'never this good', or 'bad' vs 'bad ass'.")
    print("  -- special uses of 'least' as negation versus comparison \n")
    for sentence in tricky_sentences:
        vs = analyzer.polarity_scores(sentence)
        print("{:-<69} {}".format(sentence, str(vs)))
    print("----------------------------------------------------")

    # input("\nPress Enter to continue the demo...\n")  # for DEMO purposes...

    print("----------------------------------------------------")
    print(
        " - VADER works best when analysis is done at the sentence level (but it can work on single words or entire novels).")
    paragraph = "It was one of the worst movies I've seen, despite good reviews. Unbelievably bad acting!! Poor direction. VERY poor production. The movie was bad. Very bad movie. VERY BAD movie!"
    print("  -- For example, given the following paragraph text from a hypothetical movie review:\n\t'{}'".format(
        paragraph))
    print(
        "  -- You could use NLTK to break the paragraph into sentence tokens for VADER, then average the results for the paragraph like this: \n")
    # simple example to tokenize paragraph into sentences for VADER
    from nltk import tokenize

    sentence_list = tokenize.sent_tokenize(paragraph)
    paragraphSentiments = 0.0
    for sentence in sentence_list:
        vs = analyzer.polarity_scores(sentence)
        print("{:-<69} {}".format(sentence, str(vs["compound"])))
        paragraphSentiments += vs["compound"]
    print("AVERAGE SENTIMENT FOR PARAGRAPH: \t" +
          str(round(paragraphSentiments / len(sentence_list), 4)))
    print("----------------------------------------------------")

    # input("\nPress Enter to continue the demo...\n")  # for DEMO purposes...

    print("----------------------------------------------------")
    print(" - Analyze sentiment of IMAGES/VIDEO data based on annotation 'tags' or image labels. \n")
    conceptList = ["balloons", "cake", "candles",
                   "happy birthday", "friends", "laughing", "smiling", "party"]
    conceptSentiments = 0.0
    for concept in conceptList:
        vs = analyzer.polarity_scores(concept)
        print("{:-<15} {}".format(concept, str(vs['compound'])))
        conceptSentiments += vs["compound"]
    print("AVERAGE SENTIMENT OF TAGS/LABELS: \t" +
          str(round(conceptSentiments / len(conceptList), 4)))
    print("\t")
    conceptList = ["riot", "fire", "fight",
                   "blood", "mob", "war", "police", "tear gas"]
    conceptSentiments = 0.0
    for concept in conceptList:
        vs = analyzer.polarity_scores(concept)
        print("{:-<15} {}".format(concept, str(vs['compound'])))
        conceptSentiments += vs["compound"]
    print("AVERAGE SENTIMENT OF TAGS/LABELS: \t" +
          str(round(conceptSentiments / len(conceptList), 4)))
    print("----------------------------------------------------")

    # input("\nPress Enter to continue the demo...")  # for DEMO purposes...

    do_translate = input(
        "\nWould you like to run VADER demo examples with NON-ENGLISH text? \n (Note: requires Internet access and uses the 'requests' library) \n Type 'y' or 'n', then press Enter: ")
    if do_translate.lower().lstrip().__contains__("y"):
        import requests
        print("\n----------------------------------------------------")
        print(" - Analyze sentiment of NON ENGLISH text...for example:")
        print("  -- French, German, Spanish, Italian, Russian, Japanese, Arabic, Chinese(Simplified) , Chinese(Traditional)")
        print("  -- many other languages supported. \n")
        languages = ["English", "French", "German", "Spanish", "Italian", "Russian",
                     "Japanese", "Arabic", "Chinese(Simplified)", "Chinese(Traditional)"]
        language_codes = ["en", "fr", "de", "es",
                          "it", "ru", "ja", "ar", "zh-CN", "zh-TW"]
        nonEnglish_sentences = ["I'm surprised to see just how amazingly helpful VADER is!",
                                "Je suis surpris de voir comment VADER est incroyablement utile !",
                                "Ich bin überrascht zu sehen, nur wie erstaunlich nützlich VADER!",
                                "Me sorprende ver sólo cómo increíblemente útil VADER!",
                                "Sono sorpreso di vedere solo come incredibilmente utile VADER è!",
                                "Я удивлен увидеть, как раз как удивительно полезно ВЕЙДЕРА!",
                                "私はちょうどどのように驚くほど役に立つベイダーを見て驚いています!",
                                "أنا مندهش لرؤية فقط كيف مثير للدهشة فيدر فائدة!",
                                "我很惊讶地看到VADER是如此有用!",
                                "我很驚訝地看到VADER是如此有用!"
                                ]
        for sentence in nonEnglish_sentences:
            to_lang = "en"
            from_lang = language_codes[nonEnglish_sentences.index(sentence)]
            if (from_lang == "en") or (from_lang == "en-US"):
                translation = sentence
                translator_name = "No translation needed"
            else:  # please note usage limits for My Memory Translation Service:   http://mymemory.translated.net/doc/usagelimits.php
                # using   MY MEMORY NET   http://mymemory.translated.net
                api_url = "http://mymemory.translated.net/api/get?q={}&langpair={}|{}".format(sentence, from_lang,
                                                                                              to_lang)
                hdrs = {
                    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11',
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                    'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3',
                    'Accept-Encoding': 'none',
                    'Accept-Language': 'en-US,en;q=0.8',
                    'Connection': 'keep-alive'}
                response = requests.get(api_url, headers=hdrs)
                response_json = json.loads(response.text)
                translation = response_json["responseData"]["translatedText"]
                translator_name = "MemoryNet Translation Service"
            vs = analyzer.polarity_scores(translation)
            print("- {: <8}: {: <69}\t {} ({})".format(languages[nonEnglish_sentences.index(sentence)], sentence,
                                                       str(vs['compound']), translator_name))
        print("----------------------------------------------------")

    print("\n\n Demo Done!")
