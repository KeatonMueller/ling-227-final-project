from models.abstract_model import AbstractModel

from re import sub
from nltk import ngrams
import string

from numpy import linalg, array, zeros, argsort
from sklearn.svm import SVC
from scipy.special import softmax


class CNGM(AbstractModel):
    """
    Character N-gram model.
    Based on method in Houvardas and Stamatatos, 2006; expanded to add flexibility.
    """


    alph = string.ascii_lowercase


    def __init__(self, N=2, features=0, extended_alphabet=False):
        """
        N should be an int specifying what size window N-gram to use.
        Default is 2 (bigram).

        features should be an integer specifying how many features
        of the len(alph)^N features to use to build each profile. Default is
        676 (26^2).

        extended_alphabet should be a boolean value specifying whether
        to build profiles using only alphabetical characters (False) or
        including punctuation (True). Default is False.
        """
        self.alph += string.punctuation + ' '
        self.N = N
        self.features = features if features != 0 else len(self.alph)**N


    def _clean(self, text):
        """
        Cleans the text into a form that works for processing by removing
        tabs and linefeeds, and punctuation is extended_alphabet = False,
        along with making the text uniformly lowercase and transforming it
        into ascii-compatible characters.
        """
        if not len(self.alph) == 26:
            text = sub('[\n\t ' + string.punctuation + ']+?', '', text)
        else:
            text = sub('[\n\t]+?', '', text)

        text = text.lower()
        text = text.encode('ascii', 'ignore').decode()
        return text


    def _profile(self, text):
        """
        Generates the author profiles, which are vectors of size len(alph)^N.
        """
        prof = zeros(len(self.alph)**self.N)
        ngs = ngrams(text, self.N)
        for tup in ngs:
            loc = 0
            for i in range(len(tup)):
                loc += (len(self.alph)**i) * self.alph.index(tup[i])
            prof[loc] += 1
        return prof


    def _reduceFeatures(self):
        """
        Reduces features to ones with largest variance. If "features"
        is not specified in instantiation, same as original profile.
        """
        # Adds up all profiles corresponding to each author,
        # then compiles into a matrix of these "group" profiles.
        group_profiles = {auth : zeros(len(self.alph)**self.N) for auth in set(self.train_data[1])}
        for i in range(len(self.train_data[1])):
            group_profiles[self.train_data[1][i]] += self.train_data[0][i]
        profile_matrix = array([group_profiles[auth] for auth in group_profiles])

        # Takes the variances for all features across the "group" profiles,
        # then extracts the indices of the features with the highest variances.
        vars = profile_matrix.var(axis=0)
        self.feature_indices = argsort(vars)[-self.features:]
        # Recompiles the training data.
        self.train_data[0] = array([prof[self.feature_indices] for prof in self.train_data[0]])


    def train(self, training_data, partitions=200):
        """
        Train the model based on the training_data.


        training_data should be a dictionary whose keys are strings
        corresponding to an author's name, and whose values are lists of texts
        written by that author.

        For example:
        training_data = {
            "Alexander Pope": ["First in...", "In every...", ...],
            "John Dryden": [...],
            ...
        }

        The model then builds a profile for each author based on the texts.


        partitions should be an into specifying how many distinct profiles
        to generate from the training data. Does so by dividing the texts
        corresponding to each author into chunks of size len(lines of texts)//partitions.
        Default is 200.
        """
        self.auths = list(training_data.keys())

        # Creates two lists, one of the texts and one of the corresponding author labels.
        labels = []
        texts = []
        for auth in training_data:
            lines = training_data[auth].split('\n')
            step = len(lines) // partitions
            for p in range( step, len(lines), step ):
                labels.append(auth)                         # authors per text in the training corpus
                texts.append('\n'.join(lines[p-step : p]))  # texts in the training corpus
        labels = array(labels)
        texts = array(texts)

        # Cleans the texts
        for i in range(len(texts)):
            texts[i] = self._clean(texts[i])

        # Generates the profiles from these tests
        profiles = zeros((len(texts), len(self.alph)**self.N))
        for i in range(len(texts)):
            profiles[i] = self._profile(texts[i])


        # Reduces the features and fits the model
        self.train_data = [profiles, labels]
        self._reduceFeatures()

        self.model = SVC(kernel='linear')
        self.model.probability = True
        self.model.fit(self.train_data[0], self.train_data[1])


    def identify(self, text):
        # Prepares the text for identification.
        text = self._clean(text)
        t_prof = self._profile(text)
        t_prof = t_prof[self.feature_indices]
        t_prof = t_prof.reshape(1, -1)

        # Gets the probabilities and return them
        ###print(t_prof)
        probs = self.model.predict_proba(t_prof)[0]
        return sorted([(probs[i], self.auths[i]) for i in range(len(probs))], reverse=True)
