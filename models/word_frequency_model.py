from abstract_model import AbstractModel
import re
import string
import pandas as pd
import numpy as np

from nltk import FreqDist
from nltk.stem import WordNetLemmatizer

class BOW(AbstractModel):
    
    ### Preprocessing: accept string text, return a list of word stems
    def _preprocess(text):
        text = re.sub('\s', ' ', text)
        text = text.lower()
        text = text.encode('ascii', 'ignore').decode()
        temp = [c for c in text if c not in string.punctuation]
        text_clean = ''.join(temp)

        li = text_clean.split(' ')
        lemm = WordNetLemmatizer()
        wdlist = [lemm.lemmatize(wd, pos="v") for wd in li]

        return wdlist

    ### Take author and input text, return df (Series) of profiles
    def _makeProfile(auth, text):
        wdlist = _preprocess(text)

        totalvocab = len(wdlist)

        fd = FreqDist(wdlist)
        commonwords = dict(fd.most_common(topN))
        df = pd.DataFrame.from_dict(commonwords, orient='index', columns=[auth])
        df = df/totalvocab #normalize
        df = df.transpose()

        return df

    ### Convert text to features
    def _makeFeatureMatrix(training_data, topN):
        featdf = pd.DataFrame()
        for auth in training_data:
            text = training_data[auth]
            df = _makeProfile(auth, text)
            featdf = featdf.append(df) #, ignore_index=True)

        featdf.fillna(0, inplace=True)
        return featdf

    ## Measures the similarity between two profiles by the angle formed between them ##
    def _cosSimilarity(p1, p2):
        return (p1 @ p2) / (np.linalg.norm(p1) * np.linalg.norm(p2))
    
    def train(self, training_data):
        ##TODO: call _makeFeatureMatrix and convert to desired data structure of o/p profiles
        
    def identify(self, text):
        ##TODO: call _makeProfile on text, find cosSimilarity, convert to probability


