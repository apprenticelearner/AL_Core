from copy import deepcopy
from agents.pyibl import Agent

from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from concept_formation.cobweb3 import Cobweb3Tree
import numpy as np

def Wrapper(clf):
	def fun(x=None):
		if x is None:
			return Pipeline([('dict vect', DictVectorizer(sparse=False)), ('clf', clf())])
		else:
			return Pipeline([('dict vect', DictVectorizer(sparse=False)), ('clf', clf(x))])

	return fun

class CustomLogisticRegression(LogisticRegression):

	def fit(self, X, y):
		self.is_single_class = False
		if len(set(y)) == 1:
			self.is_single_class = True
			self.single_label = y[0]
			return self
		else:
			return super(CustomLogisticRegression, self).fit(X,y)

	def predict(self, X):
		if self.is_single_class:
			return np.array([self.single_label for x in X])
		else:
			return super(CustomLogisticRegression, self).predict(X)

class ScikitCobweb(object):

	def __init__(self):
		self.tree = Cobweb3Tree()
	
	def fit(self, X, y):
		X = deepcopy(X)
		for i, x in enumerate(X):
			x['y_label'] = "%i" % y[i]
		self.tree.fit(X, randomize_first=False)

	def predict(self, X):
		return np.array([int(self.tree.categorize(x).predict('y_label')) for x in X])

class ScikitPyIBL(object):

    def __init__(self, defaultUtility=0.0, noise=None, decay=None, temperature=None):
        self.defaultUtility = defaultUtility
        self.noise = noise
        self.decay = decay
        self.temperature = temperature

    def fit(self, X, y):
        self.X = X
        self.y = y

    def train(self):
        num_features = len(self.X[0])
        features = ["Feature %i" % i for i in range(num_features)]
        self.agent = Agent("Agent", *features)
        self.agent.defaultUtility = self.defaultUtility
        self.agent.noise = self.noise
        self.agent.decay = self.decay
        self.agent.temperature = self.temperature

        for i,x in enumerate(self.X):
            zero_situation = self.agent.situationDecision("0", tuple(x))
            one_situation = self.agent.situationDecision("1", tuple(x))
            result = self.agent.choose(zero_situation, one_situation)
            print("Pred: ", result)
            print("Actual: ", self.y[i])
            if int(result) == self.y[i]:
                self.agent.respond(1.0)
                print("positive reward")
            else:
                self.agent.respond(-1.0)
                print("negative reward")

    def predict(self, X):
        predictions = []

        for x in X:
            self.train()
            zero_situation = self.agent.situationDecision("0", tuple(x))
            one_situation = self.agent.situationDecision("1", tuple(x))
            predictions.append(int(self.agent.choose(zero_situation,
                                                 one_situation)))
        return np.array(predictions)

when_learners = {}
when_learners['naive bayes'] = Wrapper(GaussianNB)
when_learners['decision tree'] = Wrapper(DecisionTreeClassifier)
when_learners['logistic regression'] = Wrapper(CustomLogisticRegression)
when_learners['cobweb'] = ScikitCobweb
when_learners['pyibl'] = ScikitPyIBL
#clf_class = Wrapper(GaussianNB)
#clf = clf_class()
#
#X = [{'color': 'red'}, {'color': 'green'}]
#y = [0, 1]
#
#clf.fit(X,y)
#print(clf.predict(X))
