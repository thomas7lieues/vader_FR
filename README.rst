====================================
VADER-Sentiment-Analysis-FR
====================================

This is a French version of VADER (Valence Aware Dictionary and Sentiment Reasoner). Please visit <https://github.com/cjhutto/vaderSentiment> to see the original version. Vader_FR possesses a manually translated french lexicon.
VADER is a lexicon and rule-based sentiment analysis tool that is *specifically attuned to sentiments expressed in social media*. It is fully open-sourced under the `[MIT License] <http://choosealicense.com/>`_ (VADER sincerely appreciate all attributions and readily accept most contributions, but please don't hold us liable).



==============================
**HOW TO USE Vader-FR**
==============================
>>>>>>> a2bbbaa7bea95f96d7384471871c7836cdedc0a2

from vaderSentiment_fr.vaderSentiment import SentimentIntensityAnalyzer

SIA = SentimentIntensityAnalyzer()

phrase = "Une phrase très cool à analyser"


score = SIA.polarity_scores(phrase)

>>>>>>> a2bbbaa7bea95f96d7384471871c7836cdedc0a2
print(score)

## Output : {'neg': 0.0, 'neu': 0.725, 'pos': 0.275, 'compound': 0.2247}



# Note : You can use polarity_scores_max instead of polarity_scores. polarity_scores_max uses fuzzywuzzy to get the most similar words with your inputs. For example "connar" won't be detected with polarity_scores but with polarity_scores_max
