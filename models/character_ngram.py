from abstract_model import AbstractModel
from re import sub
from nltk import ngrams
from numpy import linalg, array, zeros
import string

# Character N-gram model
# Following the method described in Kjell et al., 2004
class CNGM(AbstractModel):
    alph = string.ascii_lowercase

    ## Measures the similarity between two profiles by the angle formed between them ##
    def cosSimilarity(self, auth, text_profile):
        return (self.profiles[auth] @ text_profile) / (linalg.norm(self.profiles[auth]) * linalg.norm(text_profile))

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


    def train(self, training_data, N=2):
        # create a dictionary to contain the profiles of each author
        self.profiles = {auth : 0 for auth in training_data}
        self.N = N

        for auth in training_data:
            training_data[auth] = self._clean(training_data[auth])

        for auth in training_data:
            self.profiles[auth] = self._profile(training_data[auth])


    def identify(self, text):
        text = self._clean(text)
        t_prof = self._profile(text)
        cos_sims = {auth : self.cosSimilarity(auth, t_prof) for auth in self.profiles}
        return max(cos_sims, key = (lambda auth: cos_sims[auth]))
