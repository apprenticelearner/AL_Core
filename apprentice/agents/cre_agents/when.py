import numpy as np
import warnings
from abc import ABCMeta
from abc import abstractmethod
from .extending import new_register_decorator, registries


# ------------------------------------------------------------------------
# : BaseWhen

register_when = new_register_decorator("when", full_descr="when-learning mechanism")

# TODO: COMMENTS
class BaseWhen(metaclass=ABCMeta):
    def __init__(self, skill,**kwargs):
        self.skill = skill
        self.agent = skill.agent

    def ifit(self, state, match, reward):
        """
        
        :param state: 
        """
        raise NotImplemented()

    def fit(self, states, matches, reward):
        """
        
        """
        raise NotImplemented()

    def score(self, state, match):
        """
        
        """
        raise NotImplemented()

    def as_conditions(self):
        """
        
        """
        raise NotImplemented()

    def predict(self, state, match):
        """
        
        """
        raise NotImplemented()


# ------------------------------------------------------------------------
# : When Learning Mechanisms


# --------------------------------------------
# : VectorTransformMixin

class VectorTransformMixin():
    def __init__(self, skill, encode_relative=True, 
                starting_state_format='flat_featurized', **kwargs):
        self.starting_state_format = 'flat_featurized'
        self.encode_relative = encode_relative
        agent = skill.agent

        # Initialize or retrive vectorizer
        if(hasattr(agent, 'vectorizer')):
            self.vectorizer = agent.vectorizer
        else:
            from numba.types import f8, string, boolean
            from cre.transform import Vectorizer
            self.vectorizer = Vectorizer([f8, string, boolean])

        # Initialize or retrive relative_encoder
        if(encode_relative):
            if(hasattr(agent, 'relative_encoder')):
                self.relative_encoder = agent.relative_encoder
            else:
                from cre.transform import RelativeEncoder
                # TODO: Backup won't work without fact_types
                self.relative_encoder = RelativeEncoder()

        self.X = []
        self.Y = []
        self.X_width = 0

    def transform(self, state, match):
        featurized_state = state.get("flat_featurized")
        if(self.encode_relative):
            wm = state.get("working_memory")
            self.relative_encoder.set_in_memset(wm)
            _vars = self.skill.where_lrn_mech._ensure_vars(match)

            featurized_state = self.relative_encoder.encode_relative_to(
                featurized_state, match, _vars)

            # print(state.get("flat"))
            # print(featurized_state)

        return self.vectorizer(featurized_state)

    def append_and_flatten_vecs(self, state, match, reward):
        continuous, nominal = self.transform(state, match)
        
        self.X_width = max(len(nominal),self.X_width)
        self.X.append(nominal)
        self.Y.append(1 if reward > 0 else -1)

        X = np.zeros((len(self.X), self.X_width), dtype=np.int64)
        for i, x in enumerate(self.X):
            X[i, :len(x)] = x

        Y = np.array(self.Y, dtype=np.int64)
        return X, Y



# --------------------------------------------
# : SklearnDecisionTree

@register_when
class SklearnDecisionTree(BaseWhen, VectorTransformMixin):
    def __init__(self, skill,**kwargs):
        super().__init__(skill,**kwargs)
        from sklearn.tree import DecisionTreeClassifier

        VectorTransformMixin.__init__(self, skill, **kwargs)
        self.classifier = DecisionTreeClassifier()
        self.X = []
        self.Y = []

    def ifit(self, state, match, reward):
        from sklearn.tree import DecisionTreeClassifier

        X,Y = self.append_and_flatten_vecs(state, match, reward)
        
        self.classifier.fit(X, self.Y)

    def predict(self, state, match):
        continuous, nominal = self.transform(state, match)

        return self.classifier.predict(nominal[:self.X_width].reshape(1,-1))[0]


