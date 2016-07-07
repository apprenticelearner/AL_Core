from pprint import pprint
from copy import deepcopy
from agents.pyibl import Agent

from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from concept_formation.cobweb3 import Cobweb3Tree
from ilp.foil import Foil
import numpy as np

# cobweb, pyibl, nearest neighbor, logistic regression

class CustomPipeline(Pipeline):
        def ifit(self, x, y):
                if not hasattr(self, 'X'):
                        self.X = []
                if not hasattr(self, 'y'):
                        self.y = []
                self.X.append(x)
                self.y.append(y)
                return self.fit(self.X, self.y)

def iFitWrapper(clf):
        def fun(x=None):
                if x is None:
                        return CustomPipeline([('clf', clf())])
                else:
                        return CustomPipeline([('clf', clf(**x))])

        return fun

def DictVectWrapper(clf):
        def fun(x=None):
                if x is None:
                        return CustomPipeline([('dict vect', DictVectorizer(sparse=False)), ('clf', clf())])
                else:
                        return CustomPipeline([('dict vect', DictVectorizer(sparse=False)), ('clf', clf(**x))])

        return fun

class ScikitFoil(object):

        def __init__(self, params=None):
                if params is None:
                        self.foil = Foil()
                else:
                        self.foil = Foil(**params)

        def fit(self, X, y):
                T = [tuple() for x in X] 
                self.foil.fit(T, X, y)

        def predict(self, X):
                yh = [1 if len([m for m in self.foil.get_matches(x)]) > 0 else 0
                      for x in X]
                return np.array(yh)

class ScikitCobweb(object):

        def __init__(self, params=None):
                if params is None:
                        self.tree = Cobweb3Tree()
                else:
                        self.tree = Cobweb3Tree(**params)

        def ifit(self, x, y):
                x = deepcopy(x)
                x['y_label'] = y
                self.tree.ifit(x)
        
        def fit(self, X, y):
                X = deepcopy(X)
                for i, x in enumerate(X):
                        x['y_label'] = "%i" % y[i]
                self.tree.fit(X, randomize_first=False)

        def predict(self, X):
                return np.array([int(self.tree.categorize(x).predict('y_label')) for x in X])

class ScikitPyIBL(object):

    def __init__(self, params=None):
        self.defaultUtility = 0.0
        self.noise = None
        self.decay = None
        self.temperature = None
        if params:
                if 'defaultUtility' in params:
                    self.defaultUtility = params['defaultUtility']
                if 'noise' in params:
                    self.noise = params['noise']
                if 'decay' in params:
                    self.decay = params['decay']
                if 'temperature' in params:
                    self.temperature = params['temperature']

    def ifit(self, x, y):
        if 'X' not in self:
            self.X = [] 
        if 'y' not in self:
            self.y = [] 
        self.X.append(x)        
        self.y.append(y)

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
            pprint(tuple(x))
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


class CustomSVM(SVC):

        def fit(self, X, y):
                self.is_single_class = False
                if len(set(y)) == 1:
                        self.is_single_class = True
                        self.single_label = y[0]
                        return self
                else:
                        return super(CustomSVM, self).fit(X,y)

        def predict(self, X):
                if self.is_single_class:
                        return np.array([self.single_label for x in X])
                else:
                        return super(CustomSVM, self).predict(X)


class CustomSGD(SGDClassifier):
        
                
        
        def fit(self, X, y):
                
                self.is_single_class = False
                if len(set(y)) == 1:
                        self.is_single_class = True
                        self.single_label = y[0]
                        return self
                else:
                        return super(CustomSGD, self).fit(X, y)

        def predict(self, X):
                if self.is_single_class:
                        return np.array([self.single_label for x in X])
                else:
                        return np.array(super(CustomSGD, self).predict(X))


class CustomKNeighborsClassifier(KNeighborsClassifier):
        
        def fit(self, X, y):
                self.is_sample_less = False
                if len(y) < self.n_neighbors :
                        self.is_sample_less = True
                        if sum(y) >= len(y)-sum(y): 
                                self.prediction = 1
                        else:
                                self.prediction = 0
                        return self
                else:
                        return super(CustomKNeighborsClassifier, self).fit(X,y)

        def predict(self, X):
                if self.is_sample_less:
                        return np.array([int(self.prediction) for x in X])
                else:
                        return super(CustomKNeighborsClassifier, self).predict(X)


parameters_nearest = {'n_neighbors':3}
parameters_sgd = {'loss' : 'perceptron'}


when_learners = {}
when_learners['naive bayes'] = DictVectWrapper(GaussianNB)
when_learners['decision tree'] = DictVectWrapper(DecisionTreeClassifier)
when_learners['logistic regression'] = DictVectWrapper(CustomLogisticRegression)
when_learners['nearest neighbors'] = DictVectWrapper(CustomKNeighborsClassifier)
when_learners['random forest'] = DictVectWrapper(RandomForestClassifier)
when_learners['svm'] = DictVectWrapper(CustomSVM)
when_learners['sgd'] = DictVectWrapper(CustomSGD)


when_learners['cobweb'] = ScikitCobweb
when_learners['pyibl'] = DictVectWrapper(ScikitPyIBL)
when_learners['foil'] = iFitWrapper(ScikitFoil)

#clf_class = Wrapper(GaussianNB)
#clf = clf_class()
#
#X = [{'color': 'red'}, {'color': 'green'}]
#y = [0, 1]
#
#clf.fit(X,y)
#print(clf.predict(X))
