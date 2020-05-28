import util
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyser = SentimentIntensityAnalyzer()

while True :
    inp = input("(q to quit) : > ")
    if inp.lower()=="q":
        break
    else :
        score = analyser.polarity_scores(inp)
        print(score)
