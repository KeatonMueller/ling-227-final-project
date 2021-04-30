from abstract_model import AbstractModel
from re import sub
from nltk import ngrams
from numpy import linalg, array, zeros, argsort
import string

from sklearn.svm import SVC

# Character N-gram model
# Expanded from method in Houvardas and Stamatatos, 2006
class CNGM(AbstractModel):
    alph = string.ascii_lowercase


    def __init__(self, N=2, features=0):
        self.N = N
        self.features = features if features != 0 else 26**N


    ## Cleans the text into a form that works for processing ##
    def _clean(self, text):
        text = sub('[\n\t ' + string.punctuation + ']+?', '', text)
        text = text.lower()
        text = text.encode('ascii', 'ignore').decode()
        return text


    ## Generates the author profiles, which are vectors of size 26^N ##
    def _profile(self, text):
        prof = zeros(26**self.N)
        ngs = ngrams(text, self.N)
        for tup in ngs:
            loc = 0
            for i in range(len(tup)):
                loc += (26**i) * self.alph.index(tup[i])
            prof[loc] += 1
        return prof


    ## Reduces features to ones with largest variance ##
    def _reduceFeatures(self):
        group_profiles = {auth : zeros(26**self.N) for auth in set(self.train_data[1])}
        for i in range(len(self.train_data[1])):
            group_profiles[self.train_data[1][i]] += self.train_data[0][i]
        profile_matrix = array([group_profiles[auth] for auth in group_profiles])
        vars = profile_matrix.var(axis=0)
        self.feature_indices = argsort(vars)[-self.features:]
        self.train_data[0] = array([prof[self.feature_indices] for prof in self.train_data[0]])


    def train(self, training_data):
        """
        @params training_data a dict whose keys are author names and values are list-like objects containing the set of training texts per author
        """
        labels = array([auth for auth in training_data for text in training_data[auth]]) # authors per text in the training corpus
        texts = array([text for auth in training_data for text in training_data[auth]])  # texts in the training corpus

        for i in range(len(texts)):
            texts[i] = self._clean(texts[i])

        profiles = zeros((len(texts), 26**self.N))
        for i in range(len(texts)):
            profiles[i] = self._profile(texts[i])

        self.train_data = [profiles, labels]
        self._reduceFeatures()

        self.model = SVC(kernel='linear')
        self.model.fit(self.train_data[0], self.train_data[1])


    def identify(self, text):
        text = self._clean(text)
        t_prof = self._profile(text)
        t_prof = t_prof[self.feature_indices]
        return self.model.predict(t_prof.reshape(1, -1))
