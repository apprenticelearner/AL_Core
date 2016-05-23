class BaseILP(object):

    def __init__(self):
        pass

    def get_match(self, X):
        pass

    def get_all_matches(self, X):
        pass

    def ifit(self, X, y):
        pass

    def fit(self, X, y):
        """
        Assume that X is a list of dictionaries that represent FOL using
        TRESTLE notation.
         
        y is a vector of 0's or 1's that specify whether the examples are
        positive or negative.
        """
        pass


