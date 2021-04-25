from abstract_model import AbstractModel
from re import sub
from nltk import ngrams
import string
from numpy import linalg, array, zeros

# Character N-gram model
# Following the method described in Kjell et al., 2004
class CNGM(AbstractModel):
    alph = string.ascii_lowercase

    def cosSimilarity(self, auth, text_profile):
        t_array = array(text_profile)
        a_array = array(self.profiles[auth])
        return (a_array @ t_array) / (linalg.norm(a_array) * linalg.norm(t_array))


    def _clean(self, text):
        text = sub('\n+?', ' ', text)
        text = text.translate(text.maketrans('', '', ' ' + string.punctuation))
        text = text.lower()
        text = text.encode('ascii', 'ignore').decode()
        return text


    def _profile(self, text):
        prof = [0]*(26**self.N)
        ngs = ngrams(text, self.N)
        for tup in ngs:
            loc = 0
            for i in range(len(tup)):
                loc += (26**i) * self.alph.index(tup[i])
            prof[loc] += 1
        return prof


    def train(self, training_data, N=2):
        # create a dictionary to contain the profiles of each author
        self.profiles = {auth : 0 for auth in training_data}
        self.N = N

        # clean the text into a form that works for processing
        for auth in training_data:
            training_data[auth] = self._clean(training_data[auth])

        # generate the author profiles, which are vectors of size 26^N
        for auth in training_data:
            self.profiles[auth] = self._profile(training_data[auth])


    def identify(self, text):
        text = self._clean(text)
        t_prof = self._profile(text)
        cos_sims = {auth : self.cosSimilarity(auth, t_prof) for auth in self.profiles}
        return max(cos_sims, key = (lambda auth: cos_sims[auth]))
