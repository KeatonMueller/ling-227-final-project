from numpy import exp
from abstract_model import AbstractModel

# add .. to python's path to allow imports from parent directory
import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from utils.lzw import compression_size

class CompressionModel(AbstractModel):
    def __init__(self):
        self._profiles = {}

    def train(self, training_data):
        # reset any previously trained profiles
        self._profiles = {}
        # store the concatenated text and compression size for each author
        for auth in training_data:
            profile_text = '\n'.join(training_data[auth])
            size = compression_size(profile_text)
            self._profiles[auth] = (profile_text, size)

    def identify(self, text):
        # dict storing sizes of the compressed texts
        sizes = {}

        # compute difference between each profile's size and profile + `text`
        for auth in self._profiles:
            profile = self._profiles[auth]
            candidate = '\n'.join([profile[0], text])
            candidate_size = compression_size(candidate)
            # store this profile's representation length of the text
            sizes[auth] = candidate_size - profile[1]

        # perform softmax over the sizes
        denom = sum(exp(list(sizes.values())))
        # flip softmax value so that higher sizes lead to lower probability
        scores = [(1 - (exp(sizes[auth]) / denom), auth) for auth in sizes]

        # return sorted in decreasing order of confidence
        return sorted(scores, key=lambda x: x[0], reverse=True)
