====================================
VADER-Sentiment-Analysis-FR
====================================

This is a French version of VADER (Valence Aware Dictionary and Sentiment Reasoner). Please visit <https://github.com/cjhutto/vaderSentiment> to see the original version. Vader_FR possesses a manually translated french lexicon.
VADER is a lexicon and rule-based sentiment analysis tool that is *specifically attuned to sentiments expressed in social media*. It is fully open-sourced under the `[MIT License] <http://choosealicense.com/>`_ (the original VADER sincerely appreciate all attributions and readily accept most contributions, but please don't them and us liable).

7LieuesTechnologies aims to support your company, from IT to accounting, by building a virtual assistant which provides multi-technologies and multi-trade help.

==============================
**Install Vader-fr**
==============================
pip install vaderSentiment-fr


==============================
**HOW TO USE Vader-FR**
==============================

from vaderSentiment_fr.vaderSentiment import SentimentIntensityAnalyzer

SIA = SentimentIntensityAnalyzer()

phrase = "Une phrase très cool à analyser"


score = SIA.polarity_scores(phrase)




# Note : You can use polarity_scores_max instead of polarity_scores. polarity_scores_max uses fuzzywuzzy to get the most similar words with your inputs. For example "connar" won't be detected with polarity_scores but with polarity_scores_max
