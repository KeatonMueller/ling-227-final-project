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
    def _makeProfile(self, auth, text):
        wdlist = _preprocess(text)

        totalvocab = len(wdlist)

        fd = FreqDist(wdlist)
        commonwords = dict(fd.most_common(self.topN))
        df = pd.DataFrame.from_dict(commonwords, orient='index', columns=[auth])
        df = df/totalvocab #normalize
        df = df.transpose()

        return df

    ### Convert text to features
    def _makeFeatureMatrix(self, training_data):
        featdf = pd.DataFrame()

        for auth in training_data:
            text = training_data[auth]

            df = _makeProfile(self, auth, text)

            featdf = featdf.append(df) #, ignore_index=True)

        featdf.fillna(0, inplace=True)

        #print(featdf)
        return featdf

    ## Measures the similarity between two profiles by the angle formed between them ##
    def _cosSimilarity(p1, p2):
        return (p1 @ p2) / (np.linalg.norm(p1) * np.linalg.norm(p2))
    
    def train(self, training_data, topN=100):
        self.topN = topN

        traindict = {}
        for auth in training_data:
            corpus = ''
            for text in training_data[auth]:
                corpus = corpus+text+' '
            traindict[auth] = corpus

        featmatrix = _makeFeatureMatrix(self, traindict)

        self.profiledf = featmatrix
        return
            
    def _createTest(self, text):
    
        featmatrix = self.profiledf

        pr = _makeProfile(self, 'Test', text)

        testdf = featmatrix.append(pr)
        testdf.fillna(0, inplace=True)

        col_list = featmatrix.columns.tolist()
        testdf = testdf.iloc[-1]

        return testdf[col_list]    
        
    def identify(self, text):
    
        testdf = _createTest(self, text)
        authors = self.profiledf.index.tolist()

        M = len(self.profiledf)
        clist = []
        for i in range(M):
            clist.append(cosSimilarity(self.profiledf.iloc[i], testdf.iloc[0]))

        probdict = {}
        for i in range(M):
            probdict[authors[i]] = clist[i]/sum(clist)

        idlist = sorted([(clist[i]/sum(clist), authors[i]) for i in range(M)], reverse=True)

        return idlist


