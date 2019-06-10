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


class WhenLearner(object):
    STATE_FORMAT_OPTIONS = ["variablized_state",  "state_only"]
    WHEN_TYPE_OPTIONS = ["one_learner_per_rhs", "one_learner_per_label"]
    CROSS_RHS_INFERENCE = ["none", "implicit_negatives", "rhs_in_y"]

    def __init__(self, learner, when_type="one_learner_per_rhs",
                 state_format="variablized_state",
                 cross_rhs_inference="none", learner_kwargs={}):
        assert state_format in self.__class__.STATE_FORMAT_OPTIONS, \
               "state_format must be one of %s but got %s" % \
               (self.__class__.STATE_FORMAT_OPTIONS, state_format)
        assert when_type in self.__class__.WHEN_TYPE_OPTIONS, \
               "when_type must be one of %s but got %s" % \
               (self.__class__.WHEN_TYPE_OPTIONS, when_type)
        assert cross_rhs_inference in self.__class__.CROSS_RHS_INFERENCE, \
               "cross_rhs_inference must be one of %s but got %s" % \
               (self.__class__.CROSS_RHS_INFERENCE, cross_rhs_inference)

        if(cross_rhs_inference == "rhs_in_y"):
            assert when_type == "one_learner_per_label", \
                   "when_type must be 'one_learner_per_label' if using \
                    cross_rhs_inference = 'rhs_in_y', but got %r " % when_type

        self.learner_name = learner
        self.learner_kwargs = learner_kwargs
        self.type = when_type
        self.state_format = state_format
        self.cross_rhs_inference = cross_rhs_inference
        self.rhs_by_label = {}
        if(cross_rhs_inference == "implicit_negatives"):
            self.examples = {}
            self.implicit_examples = {}
        # (self.type == "one_learner_per_rhs"):
        self.learners = {}

    def add_rhs(self, rhs):
        if(self.type == "one_learner_per_rhs"):
            self.learners[rhs] = get_when_sublearner(self.learner_name,
                                                     **self.learner_kwargs)
            key = rhs
        else:
            key = rhs.label
        rhs_list = self.rhs_by_label.get(rhs.label, [])
        rhs_list.append(rhs)
        self.rhs_by_label[rhs.label] = rhs_list

        if(self.cross_rhs_inference == "implicit_negatives"):
            self.examples[key] = {}
            self.examples[key]['state'] = []
            self.examples[key]['reward'] = []
            self.implicit_examples[key] = {}
            self.implicit_examples[key]['state'] = []
            self.implicit_examples[key]['reward'] = []

    def ifit(self, rhs, state, reward):
        # print("FIT_STATe", state)
        # print([str(x.input_rule) for x in self.learners.keys()])
        # print("REQARD", reward)
        # print("LEARNERS",self.learners)
        # print("LEARNER",id(self.learners[rhs]))
        if(self.cross_rhs_inference == "implicit_negatives"):
            if(self.type == "one_learner_per_label"):
                key = rhs.label
            elif(self.type == "one_learner_per_rhs"):
                key = rhs

            states = self.examples[key]['state']
            rewards = self.examples[key]['reward']
            states.append(state)
            rewards.append(reward)

            implicit_states = self.implicit_examples[key]['state']
            implicit_rewards = self.implicit_examples[key]['reward']

            if(reward > 0):
                for other_key, other_impl_exs in self.implicit_examples.items():
                    if(other_key != key):

                        # Add implicit negative examples to any rhs that doesn't already have this state
                        # TODO: Do this for all bindings not just for the given state
                        if(state not in self.examples[other_key]['state']):
                            other_impl_exs['state'].append(state)
                            other_impl_exs['reward'].append(-1)

                # Remove any implicit negative examples in this rhs with the current state
                try:
                    index_value = self.implicit_examples[key]['state'].index(state)
                except ValueError:
                    index_value = -1

                if(index_value != -1):
                    del self.implicit_examples[key]['state'][index_value]
                    del self.implicit_examples[key]['reward'][index_value]

            # PRINT AREA
            # for x in self.implicit_examples:
            #     print("%s : %s" % (x.input_rule,self.implicit_examples[x]['reward']))
            # for x in self.examples:
            #     print("%s : %s" % (x.input_rule,self.examples[x]['reward']))
            # pprint(self.implicit_examples)

            self.learners[key] = get_when_sublearner(self.learner_name,
                                                     **self.learner_kwargs)
            # pprint(states)
            # pprint(rewards)
            self.learners[key].fit(states+implicit_states,
                                   rewards+implicit_rewards)
            # print(self.learners[key])
        else:
            if(self.type == "one_learner_per_label"):
                if(rhs.label not in self.learners):
                    self.learners[rhs.label] = get_when_sublearner(
                                                self.learner_name,
                                                **self.learner_kwargs)
                if(self.cross_rhs_inference == "rhs_in_y"):
                    self.learners[rhs.label].ifit(state, (rhs._id_num, reward))
                else:
                    self.learners[rhs.label].ifit(state, reward)
            elif(self.type == "one_learner_per_rhs"):
                self.learners[rhs].ifit(state, reward)

    def predict(self, rhs, state):
        # print("STATE:",state, type(state))
        if(self.type == "one_learner_per_label"):
            prediction = self.learners[rhs.label].predict([state])[0]
        elif(self.type == "one_learner_per_rhs"):
            # print(self.learners[rhs].predict([state])[0]        )
            prediction = self.learners[rhs].predict([state])[0]

        if(self.cross_rhs_inference == "rhs_in_y"):
            rhs_pred, rew_pred = prediction
            if(rhs_pred != rhs._id_num):
                rew_pred = -1
            return rew_pred
        else:
            return prediction

    def skill_info(self, rhs, state):
        key = rhs if(self.type == "one_learner_per_rhs") else rhs.label
        sublearner = self.learners[key]

        if(isinstance(sublearner, Pipeline)):
            feature_names = sublearner.named_steps["dict vect"].get_feature_names()

        return sublearner.skill_info(state)


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
        self.y.append(int(y) if not isinstance(y, tuple) else y)

        # print("IFIT:",self.X)
        # print(self.y)
        return super(CustomPipeline, self).fit(self.X, self.y)

    def fit(self, X, y):

        # print("X",X)
        # NOTE: Only using boolean values
        X = [{k: v for k, v in d.items() if isinstance(v, bool)} for d in X]
        print("FITX", X)
        ft = Flattener()
        tup = Tuplizer()

        self.X = [tup.undo_transform(ft.transform(x)) for x in X]
        self.y = [int(x) if not isinstance(x, tuple) else x for x in y]

        super(CustomPipeline, self).fit(self.X, self.y)

    def predict(self, X):
        ft = Flattener()
        tup = Tuplizer()
        X = [tup.undo_transform(ft.transform(x)) for x in X]
        # print("BEEP", X)
        # print("PRED:",X)
        # print("VAL:", super(CustomPipeline, self).predict(X))
        return super(CustomPipeline, self).predict(X)

    def skill_info(self, X):
        X = [X] if not isinstance(X, list) else X
        feature_names = self.named_steps["dict vect"].get_feature_names()
        classifier = self.steps[-1][-1]

        ft = Flattener()
        tup = Tuplizer()
        X = [tup.undo_transform(ft.transform(x)) for x in X]

        # X = self.named_steps["dict vect"].transform(x)
        # X = self._transform(X)

        # ft = Flattener()
        # tup = Tuplizer()
        # print("BAE1",X)
        # print("PEEP",feature_names)
        # print("BAE",[tup.transform(x) for x in X])
        # print(type(self))
        # X = [tup.undo_transform(ft.transform(x)) for x in X]
        Xt = X
        for name, transform in self.steps[:-1]:
            if transform is not None:
                # print("HEY",transform.get_feature_names())
                Xt = transform.transform(Xt)
                # print("BAE_"+name,Xt)

        return classifier.skill_info(Xt,feature_names)

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
        dv = DictVectorizer(sparse=False, sort=False)
        if x is None:
            return CustomPipeline([('dict vect', dv),
                                   ('clf', clf())])
        else:
            return CustomPipeline([('dict vect', dv),
                                   ('clf', clf(**x))])

    return fun

    # def applicable_skills(self,state,skill_label,skills=None):
    #     if(skills == None): skills = self.skills_by_label[skill_label]
    #     raise NotImplementedError("Still need to write applicable_skills")


from sklearn.tree import _tree
class DecisionTree(DecisionTreeClassifier):

    # def fit(self,X,y):
    #     print("MOOP",X)
    #     super(DecisionTree,self).fit(X,y)

    def predict(self, X):
        print("MOOP", X)
        return super(DecisionTree, self).predict(X)

    def skill_info(self, examples, feature_names=None):
        print("SLOOP", examples)
        tree = self
        tree_ = tree.tree_
        print("feature_names", feature_names)
        feature_name = [
            feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
            for i in tree_.feature
        ]
        node_indicator = tree.decision_path(examples)
        dense_ind = np.array(node_indicator.todense())

        def recurse(node, ind):
            if(tree_.feature[node] != _tree.TREE_UNDEFINED):
                l = tree_.children_left[node]
                less = ind[l]
                if(not less):
                    s = recurse(tree_.children_right[node], ind)
                else:
                    s = recurse(tree_.children_left[node], ind)

                name = feature_name[node]
                ineq = "<=" if less else ">"
                thresh = str(tree_.threshold[node])
                return [(name.replace("?ele-", ""), ineq, thresh)] + s
            else:
                return []
        for ind in dense_ind:
            return recurse(0, ind)


class ScikitTrestle(object):

    def __init__(self, **kwargs):
        self.tree = TrestleTree(**kwargs)
        self.state_format = "variablized_state"

    def ifit(self, x, y):
        x = deepcopy(x)
        x['_y_label'] = float(y)
        self.tree.ifit(x)

    def fit(self, X, y):
        X = deepcopy(X)
        for i, x in enumerate(X):
            x['_y_label'] = float(y)
        self.tree.fit(X, randomize_first=False)

    def skill_info(self, X):
        raise NotImplementedError("Not implemented Erik H. says there is a way \
             to serialize this -> TODO")

    def predict(self, X):
        return [self.tree.categorize(x).predict('_y_label') for x in X]


class ScikitCobweb(object):

    def __init__(self, **kwargs):
        self.tree = Cobweb3Tree(**kwargs)
        self.state_format = "variablized_state"

    def ifit(self, x, y):
        x = deepcopy(x)
        x['_y_label'] = float(y)
        self.tree.ifit(x)

    def fit(self, X, y):
        X = deepcopy(X)
        for i, x in enumerate(X):
            # print(y)
            x['_y_label'] = float(y) if not isinstance(y,list) else y[i]
        self.tree.fit(X, randomize_first=False)

    def skill_info(self, X):
        raise NotImplementedError("Not implemented Erik H. says there \
                 is a way to serialize this -> TODO")

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
        self.state_format = "variablized_state"

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
        self.state_format = "variablized_state"

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
        self.state_format = "variablized_state"

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
        self.state_format = "variablized_state"

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
        self.state_format = "variablized_state"

    def predict(self, X):
        if self.is_sample_less:
            return np.array([int(self.prediction) for x in X])
        else:
            return super(CustomKNeighborsClassifier, self).predict(X)


class AlwaysTrue(object):

    def __init__(self):
        self.state_format = "variablized_state"

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
        self.state_format = "variablized_state"

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

# --------------------------------UTILITIES--------------------------------


def get_when_sublearner(name, **learner_kwargs):
    return WHEN_CLASSIFIERS[name.lower().replace(' ', '').replace('_', '')](**learner_kwargs)


def get_when_learner(name, learner_kwargs={}):
    inp_d = WHEN_LEARNERS[name.lower().replace(' ', '').replace('_', '')]
    inp_d['learner_kwargs'] = learner_kwargs
    return WhenLearner(**inp_d)


WHEN_LEARNERS = {
    "decisiontree": {"learner": "decisiontree",
                     "state_format": "variablized_state"},
    "cobweb": {"learner": "cobweb", "when_type": "one_learner_per_rhs",
               "state_format": "variablized_state"},
    "trestle": {"learner": "cobweb", "when_type": "one_learner_per_rhs",
                "state_format": "variablized_state"}
}

WHEN_CLASSIFIERS = {
    'naivebayes': DictVectWrapper(BernoulliNB),
    'decisiontree': DictVectWrapper(DecisionTree),
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
