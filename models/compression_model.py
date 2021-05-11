from numpy import exp
from models.abstract_model import AbstractModel
from utils.lzw import compression_size

class CompressionModel(AbstractModel):
    '''
    Based on ideas presented in: https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.88.048702
    '''
    def __init__(self):
        self._profiles = {}

    def train(self, training_data):
        # reset any previously trained profiles
        self._profiles = {}
        # store the concatenated text and compression size for each author
        for auth in training_data:
            profile_text = ''.join(training_data[auth])
            size = compression_size(profile_text)
            self._profiles[auth] = (profile_text, size)

    def identify(self, text):
        # dict storing sizes of the compressed texts
        sizes = {}

        # compute difference between each profile's size and profile + `text`
        for auth in self._profiles:
            profile = self._profiles[auth]
            candidate = profile[0] + text
            candidate_size = compression_size(candidate)
            # store this profile's representation length of the text
            sizes[auth] = candidate_size - profile[1]

        # normalize sizes to prevent overflow
        # flip sign so that higher size leads to lower confidence
        sizes = { auth: -size / max(sizes.values()) for auth, size in sizes.items() }

        # perform softmax over the sizes
        denom = sum(exp(list(sizes.values())))
        scores = [(exp(sizes[auth]) / denom, auth) for auth in sizes]

        # return sorted in decreasing order of confidence
        return sorted(scores, key=lambda x: x[0], reverse=True)
