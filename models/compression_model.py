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
        # dict storing size difference of compressed texts
        diffs = {}

        # compute difference between each profile's size and profile + `text`
        for auth in self._profiles:
            profile = self._profiles[auth]
            candidate = '\n'.join([profile[0], text])
            candidate_size = compression_size(candidate)

            # NOTE: percent difference *highly* favors larger training sets
            #       I still need to a read some papers to find a better way of comparing compressed sizes
            pct_diff = candidate_size / profile[1]
            diffs[auth] = pct_diff

        # perform softmax over percent differences
        denom = sum(exp(list(diffs.values())))
        # flip softmax value so that higher difference leads to lower probability
        scores = [(1 - (exp(diffs[auth]) / denom), auth) for auth in diffs]

        # return sorted in decreasing order of confidence
        return sorted(scores, key=lambda x: x[0], reverse=True)
