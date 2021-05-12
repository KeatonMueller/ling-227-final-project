from models.abstract_model import AbstractModel
from models.character_ngram_svm import CNGM
from models.compression_model import CompressionModel
from models.word_frequency_model import BOW

class Ensemble(AbstractModel):
    def __init__(self, weights=(1/3.0, 1/3.0, 1/3.0), CNGM_specs=(2, 0, False), BOW_topN=100):
        """
        weights should be a tuple contained the weights for the models, in the following order:
                (BOW weight, CompressionModel weight, CNGM weight)
        default is 1/3 for each.
        
        CNGM_specs should be a tuple containing the specs of the CNGM model in the same order
        as specified in the CNGM class; default is the same as CNGM defaults.
        
        BOW_topN should be an int following the specification of the BOW class.
        """
        assert len(CNGM_specs) == 3
        
        self.weights = weights
        self.bowm = BOW(BOW_topN)
        self.comm = CompressionModel()
        self.cngm = CNGM(CNGM_specs[0], CNGM_specs[1], CNGM_specs[2])


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

        probs = [( (self.weights[0]*bow_probs[i][0])
                    + self.weights[1]*com_probs[i][0]
                    + self.weights[2]*cng_probs[i][0], bow_probs[i][1] ) for i in range(len(bow_probs))]
        return sorted(probs, reverse=True)
