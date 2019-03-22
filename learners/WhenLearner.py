from pprint import pprint
from copy import deepcopy
from learners.pyibl import Agent
import numpy as np

from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier
# from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from concept_formation.cobweb3 import Cobweb3Tree
from concept_formation.trestle import TrestleTree
from concept_formation.preprocessor import Tuplizer
from concept_formation.preprocessor import Flattener

# from ilp.foil_classifier import FoilClassifier

# cobweb, pyibl, nearest neighbor, logistic regression


class CustomPipeline(Pipeline):

    def ifit(self, x, y):
        if not hasattr(self, 'X'):
            self.X = []
        if not hasattr(self, 'y'):
            self.y = []

        ft = Flattener()
        tup = Tuplizer()

        # pprint(x)
        self.X.append(tup.undo_transform(ft.transform(x)))
        self.y.append(int(y))
        # print(self.y)
        return self.fit(self.X, self.y)

    def predict(self, X):
        ft = Flattener()
        tup = Tuplizer()
        X = [tup.undo_transform(ft.transform(x)) for x in X]
        return super(CustomPipeline, self).predict(X)

    def __repr__(self):
        return repr(self.named_steps['clf'])


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
            return CustomPipeline([('dict vect', DictVectorizer(sparse=False)),
                                   ('clf', clf())])
        else:
            return CustomPipeline([('dict vect', DictVectorizer(sparse=False)),
                                   ('clf', clf(**x))])

    return fun




class WhenLearner(object):
    STATE_FORMAT_OPTIONS = ["var_foas_state",  "state_only"]
    WHEN_TYPE_OPTIONS = ["one_learner_per_skill", "one_learner_per_label"]

    def __init__(self, learner_name, when_type = "one_learner_per_skill", state_format="var_foas_state",learner_kwargs={}):
        assert state_format in x.__class__.STATE_FORMAT_OPTIONS
        assert when_type in x.__class__.WHEN_TYPE_OPTIONS

        self.learner_kwargs = learner_kwargs
        self.learner_name = learner_name
        self.type = when_type
        self.state_format = state_format
        self.skill_dict = {}
            
        # (self.type == "one_learner_per_skill"):
        self.learners = {}


    def add_skill(self, skill, skill_label):
        if(self.type == "one_learner_per_skill"):
            self.learners[skill] = get_when_agent(self.learner_name,learner_kwargs)
        skills = self.skill_dict.get(skill_label,[])
        skills.append(skill)
        self.skill_dict[skill_label] = skills


    def ifit(self,skill, skill_label, state,reward):
        if(self.type == "one_learner_per_label"):
            if(not skill_label in self.learner):
                self.learner[skill_label] = get_when_agent(self.learner_name,learner_kwargs)
            self.learner[skill_label].ifit(state,reward)
        elif(self.type == "one_learner_per_skill"):
            self.learners[skill].ifit(state,reward)

    def predict(self,skill,skill_label,state,reward):
        if(self.type == "one_learner_per_label"):
            return self.predict(state)
        elif(self.type == "one_learner_per_skill"):
            return self.learners[skill].predict(state)        

    def applicable_skills(self,state,skill_label,skills=None):
        if(skills == None): skills = self.skill_dict[skill_label]
        raise NotImplementedError("Still need to write applicable_skills")




class ScikitTrestle(object):

    def __init__(self, **kwargs):
        self.tree = TrestleTree(**kwargs)
        self.state_format = "var_foas_state"

    def ifit(self, x, y):
        x = deepcopy(x)
        x['_y_label'] = float(y)
        self.tree.ifit(x)

    def fit(self, X, y):
        X = deepcopy(X)
        for i, x in enumerate(X):
            x['_y_label'] = float(y)
        self.tree.fit(X, randomize_first=False)

    def predict(self, X):
        return [self.tree.categorize(x).predict('_y_label') for x in X]


class ScikitCobweb(object):

    def __init__(self, **kwargs):
        self.tree = Cobweb3Tree(**kwargs)
        self.state_format = "var_foas_state"

    def ifit(self, x, y):
        x = deepcopy(x)
        x['_y_label'] = float(y)
        self.tree.ifit(x)

    def fit(self, X, y):
        X = deepcopy(X)
        for i, x in enumerate(X):
            x['_y_label'] = float(y)
        self.tree.fit(X, randomize_first=False)

    def predict(self, X):
        return [self.tree.categorize(x).predict('_y_label') for x in X]


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
        self.state_format = "var_foas_state"

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

        for i, x in enumerate(self.X):
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
            return super(CustomLogisticRegression, self).fit(X, y)
        self.state_format = "var_foas_state"

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
            return super(CustomSVM, self).fit(X, y)
        self.state_format = "var_foas_state"

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
        self.state_format = "var_foas_state"

    def predict(self, X):
        if self.is_single_class:
            return np.array([self.single_label for x in X])
        else:
            return np.array(super(CustomSGD, self).predict(X))


class CustomKNeighborsClassifier(KNeighborsClassifier):

    def fit(self, X, y):
        self.is_sample_less = False
        if len(y) < self.n_neighbors:
            self.is_sample_less = True
            if sum(y) >= len(y)-sum(y):
                self.prediction = 1
            else:
                self.prediction = 0
            return self
        else:
            return super(CustomKNeighborsClassifier, self).fit(X, y)
        self.state_format = "var_foas_state"

    def predict(self, X):
        if self.is_sample_less:
            return np.array([int(self.prediction) for x in X])
        else:
            return super(CustomKNeighborsClassifier, self).predict(X)


class AlwaysTrue(object):

    def __init__(self):
        self.state_format = "var_foas_state"

    def ifit(self, x, y):
        pass

    def fit(self, X, y):
        pass

    def predict(self, X):
        return [1 for x in X]


class MajorityClass(object):

    def __init__(self):
        self.pos = 0
        self.neg = 0
        self.state_format = "var_foas_state"

    def ifit(self, x, y):
        if y == 1:
            self.pos += 1
        elif y == 0:
            self.neg += 1
        else:
            raise Exception("y must equal 0 or 1")

    def fit(self, X, y):
        for i in range(len(X)):
            self.ifit(X[i], y[i])

    def predict(self, X):
        if self.pos >= self.neg:
            return [1 for x in X]
        else:
            return [0 for x in X]


parameters_nearest = {'n_neighbors': 3}
parameters_sgd = {'loss': 'perceptron'}


#######-------------------UTILITIES----------------#######


def get_when_agent(name,learner_kwargs={}):
    return WHEN_LEARNER_AGENTS[name.lower().replace(' ', '').replace('_', '')](**learner_kwargs)

def get_when_learner(name,learner_kwargs={}):
    inp_d = WHEN_LEARNERS[name.lower().replace(' ', '').replace('_', '')]
    inp_d['learner_kwargs'] = learner_kwargs
    return WhenLearner(**inp_d)


WHEN_LEARNERS ={
    "decisiontree" : {"learner" : "decisiontree", "when_type" : "one_learner_per_skill", "state_format":"var_foas_state"}
    "cobweb" : {"learner" : "cobweb", "when_type" : "one_learner_per_skill", "state_format":"var_foas_state"}
    "trestle" : {"learner" : "cobweb", "when_type" : "one_learner_per_skill", "state_format":"var_foas_state"}
}

WHEN_LEARNER_AGENTS = {
    'naivebayes': DictVectWrapper(BernoulliNB),
    'decisiontree': DictVectWrapper(DecisionTreeClassifier),
    'logisticregression': DictVectWrapper(CustomLogisticRegression),
    'nearestneighbors': DictVectWrapper(CustomKNeighborsClassifier),
    'random_forest': DictVectWrapper(RandomForestClassifier),
    'svm': DictVectWrapper(CustomSVM),
    'sgd': DictVectWrapper(CustomSGD),
    'cobweb': ScikitCobweb,
    'trestle': ScikitTrestle,
    'pyibl': DictVectWrapper(ScikitPyIBL),
    'majorityclass': MajorityClass,
    'alwaystrue': AlwaysTrue
}

# clf_class = Wrapper(GaussianNB)
# clf = clf_class()

# X = [{'color': 'red'}, {'color': 'green'}]
# y = [0, 1]

# clf.fit(X,y)
# print(clf.predict(X))
