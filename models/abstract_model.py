from abc import ABCMeta, abstractmethod

class AbstractModel(metaclass=ABCMeta):
    """
    Abstract class for authorship identification models.
    All individual model implementations should extend this class.
    """

    @abstractmethod
    def train(self, training_data):
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
        """
        pass

    @abstractmethod
    def identify(self, text):
        """
        Run identification algorithm for the given text.

        Returns a probability distribution over all of the authors
        it was trained on with the probability that author wrote the text.

        Return value should be list of (probability, author) tuples sorted
        in decreasing order of confidence.

        For example:
        [(0.86, "John Dryden"), (0.14, "Alexander Pope")]
        """
        pass