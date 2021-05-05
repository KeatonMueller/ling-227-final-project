from models.abstract_model import AbstractModel
from models.character_ngram_svm import CNGM
from models.compression_model import CompressionModel
from models.word_frequency_model import BOW

class Ensemble(AbstractModel):
    def __init__(self, weights=(1/3.0, 1/3.0, 1/3.0), CNGM_specs=(2, 0)):
        assert len(CNGM_specs) == 2
        
        self.bowm = BOW()
        self.comm = CompressionModel()
        self.cngm = CNGM(CNGM_specs[0], CNGM_specs[1])


    def train(self, training_data):
        self.bowm.train(training_data)
        self.comm.train(training_data)
        self.cngm.train(training_data)


    def identify(self, text):
        bow_probs = self.bowm.identify(text)
        bow_probs = sorted(bow_probs, key=lambda p: p[1])
        com_probs = self.comm.identify(text)
        com_probs = sorted(com_probs, key=lambda p: p[1])
        cng_probs = self.cngm.identify(text)
        cng_probs = sorted(cng_probs, key=lambda p: p[1])

        probs = [( (weights[0]*bow_probs[i][0])
                    + weights[1]*com_probs[i][0]
                    + weights[2]*cng_probs[i][0], bow_probs[i][1] ) for i in range(len(bow_probs))]
        return sorted(probs, reverse=True)
