from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import os
from os import listdir
from os.path import isfile, join
import json
import re
import time
import traceback

def SA_score(sentence):
    score = analyser.polarity_scores(sentence)
    return score

def disp_score(score):
    print("Input : ",score["input"],
          "\nNegativity : ", score["neg"],
          "\nNeutrality : ", score["neu"],
          "\nPositivity : ", score["pos"],
          "\nCompound : ", score["compound"])
    return True

def jauge(input):
    score=SA_score(input)
    dict = { "input" : input,
             "neg" : score["neg"],
             "neu" : score["neu"],
             "pos" : score["pos"],
             "compound" : score["compound"]}
    return dict

def chat():
    to_save=[]
    error = False
    while True:
        inp = input("Entrez (pressez q pour quitter) : > ")
        if inp.lower() =="q":
            break
        else :
            try :
                score = jauge(inp)
                disp_score(score)
                to_save.append(score)
            except Exception as e :
                traceback.print_exc()
                error = True
                break
    return to_save, error

def chat_file(file):
    to_save=[]
    error = False
    with open("fichiers/"+file+".txt", "r") as file :
        fichier = file.read()
    liste_mots = fichier.split("\n")
    print(liste_mots)
    liste_mots = filter(None,liste_mots)


    for inp in liste_mots :
        print(inp)
        try :
            score = jauge(inp)
            time.sleep(0.2)
            disp_score(score)
            to_save.append(score)
        except Exception as e :
            traceback.print_exc()
            error = True
            break
    return to_save, error

def save_as_json(l,i):
    filename = "results/chat_results"+str(i)+".json"
    with open(filename,"w") as file :
        json.dump(l, file, indent = 4, ensure_ascii=False)
    return filename

def save_as_json_with_namefile(l, namefile):
    filename = "results/"+namefile + ".json"
    with open(filename,"w") as file :
        json.dump(l, file, indent = 4, ensure_ascii=False)
    return filename


def get_no():
    current_path = os.getcwd()
    results_path = current_path + "/results"
    files = [f for f in listdir(results_path) if isfile(join(results_path,f))]
    no = []
    for file in files:
        if "chat_results" in file:
            temp=re.findall(r'\d+',file)
            res=list(map(int,temp))
            cur = res[0]
            no.append(cur)
    if no ==[]:
        return(0)
    else :
        return max(no)

def fichier_dispo():
    current_path = os.getcwd()
    fichiers_path = "fichiers"
    files = [f for f in listdir(fichiers_path) if isfile(join(fichiers_path,f))]
    retour = []
    to_print = "Voici la liste des fichiers disponibles : "
    for i in range(len(files)) :
        if files[i].endswith(".txt"):
            to_print += files[i]
            retour.append(files[i].split(".")[0])
            if i != len(files)-1:
                to_print += (", ")
    print(to_print)
    return retour
