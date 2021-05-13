from models.abstract_model import AbstractModel

from re import sub
from nltk import ngrams
import string

from numpy import linalg, array, zeros, argsort, mean
from sklearn.svm import SVC
from scipy.special import softmax


class CNGM(AbstractModel):
    """
    Character N-gram model.
    Based on method in Houvardas and Stamatatos, 2006; expanded to add flexibility.
    https://www.researchgate.net/publication/221655968_N-Gram_Feature_Selection_for_Authorship_Identification
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
        self.alph += (string.punctuation + ' ') if extended_alphabet else ''
        self.N = N
        self.features = features if features != 0 else len(self.alph)**N


    def _clean(self, text):
        """
        Cleans the text into a form that works for processing by removing
        tabs and linefeeds, and punctuation is extended_alphabet = False,
        along with making the text uniformly lowercase and transforming it
        into ascii-compatible characters.
        """
        if len(self.alph) == 26:
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


    def train(self, training_data, chunk_size=100):
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


        chunk_size should be an int specifying how large the distinct profiles
        generated from the training data should be, by number of lines. Default is 500.
        """
        # For some reason, for the SVM to work, the keys need to be in alphabetical order
        training_data = {k : training_data[k] for k in sorted(training_data)}

        # Compile all author texts into one large text to then be broken down
        for auth in training_data:
            training_data[auth] = '\n\n'.join(training_data[auth])

        self.auths = list(training_data.keys())
        self.chunk_size = chunk_size

        # Creates two lists, one of the texts and one of the corresponding author labels.
        labels = []
        texts = []
        for auth in training_data:
            lines = training_data[auth].split('\n')
            for p in range( chunk_size, len(lines), chunk_size ):
                labels.append(auth)                               # authors per text in the training corpus
                texts.append('\n'.join(lines[p-chunk_size : p]))  # texts in the training corpus
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
        # Generates smaller chunks of text (same size as in training) and gets probabilities of
        # each chunk belonging to a certain author.
        lines = text.split('\n')
        net_probs = []
        for p in range(0, len(lines), self.chunk_size):
            chunk = '\n'.join(lines[p : p + self.chunk_size])
            chunk = self._clean(chunk)
            c_prof = c_prof = self._profile(chunk)
            c_prof = c_prof[self.feature_indices]
            c_prof = c_prof.reshape(1, -1)
            probs = self.model.predict_proba(c_prof)[0]
            net_probs.append(probs)
        ###print(array(net_probs))
        # Takes the average over all probabilities
        ###print(self.auths)
        avg_probs = mean(array(net_probs), axis=0)
        ###print(avg_probs)
        return sorted([(avg_probs[i], self.auths[i]) for i in range(len(avg_probs))], reverse=True)
