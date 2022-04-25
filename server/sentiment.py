from enum import Enum


class Sentiment(Enum):
    """
    Simple Enum to store the sentiment.
    The integer value of the Enum is based on the label_class used in other places of the code
    """

    POSITIVE = (2, "positive")
    NEGATIVE = (3, "negative")

    def __str__(self):
        return self.value[1]

    def __hash__(self):
        return self.value[0]

    def __eq__(self, other):
        return self.value[0] == other.value[0]

    def get_label_class(self):
        return self.value[0]

    @staticmethod
    def from_string(name) -> 'Sentiment':
        if name == str(Sentiment.POSITIVE):
            return Sentiment.POSITIVE

        if name == str(Sentiment.NEGATIVE):
            return Sentiment.NEGATIVE

        raise RuntimeError("Cannot convert the given string to any sentiment value")
