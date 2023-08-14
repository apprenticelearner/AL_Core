import numpy as np
import warnings
from abc import ABCMeta
from abc import abstractmethod
from .extending import new_register_decorator, registries
from cre.utils import PrintElapse


# ------------------------------------------------------------------------
# : BaseWhen

register_when = new_register_decorator("when", full_descr="when-learning mechanism")

# TODO: COMMENTS
class BaseWhen(metaclass=ABCMeta):
    def __init__(self, skill,**kwargs):
        self.skill = skill
        self.agent = skill.agent

        # Note this line makes it possible to call 
        super(BaseWhen, self).__init__(skill, **kwargs)

    def ifit(self, state, skill_app, reward):
        """
        
        :param state: 
        """
        raise NotImplemented()

    def fit(self, states, skill_apps, reward):
        """
        
        """
        raise NotImplemented()

    def score(self, state, skill_app):
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

    def get_info(self, **kwargs):
        return {}


# ------------------------------------------------------------------------
# : Helpful mixins for resusing code in when-learning mechanisms

# --------------------------------------------
# : RefittableMixin

class RefittableMixin():
    '''A mixin which implements add_example() and remove_example().
        self.examples maps skill_apps to tuples of (index, reward)
     '''
    def __init__(self, skill,**kwargs):
        self.examples = {}

    def transform(self, state, match):
        '''Should take in a state and skill_app and return '''
        raise NotImplemented()

    def insert_transformed(self, transformed_state, skill_app, reward, index):
        raise NotImplemented()

    def rebase_examples(self):
        raise NotImplemented()        

    def _assert_skill_app_from_state(self, state, skill_app):
        assert state.get('__uid__') == skill_app.state_uid, (
            f"Provided state {state.get('__uid__')} is not associated with skill app: {skill_app}"
        )

    def add_example(self, state, skill_app, reward):
        ''' Adds a new training example. Applies transform() and insert_transformed() 
            from the child class. Checks to see if the example is new or a repeat and 
            returns the new index or possibly old index for the example and the add 
            attempt changed the set of training examples.
        ''' 
        # self._assert_skill_app_from_state(state, skill_app)
        did_change = False
        index, old_reward = self.examples.get(skill_app,(-1,0))
        if(index != -1): 
            print("REPLACING FEEDBACK", skill_app, old_reward, "->", reward, "@ index", index)
        new_index = len(self.examples) if index == -1 else index

        # NOTE: temping to only re-insert on reward changes, but meta-features can make it helpful
        #  to retrain if possible new feautres. 
        self.examples[skill_app] = (new_index, reward)
        transformed_state = self.transform(state, skill_app.match)
        self.insert_transformed(transformed_state, skill_app, reward, new_index)
                
        return new_index

    def remove_example(self, state, skill_app):
        ''' Attempt to remove a skill_app instance from the training set.
            Shifts the set of indicies and calls rebase_examples() from the
            child class.
        '''
        # self._assert_skill_app_from_state(state, skill_app)
        did_change = False
        index, old_reward = self.examples.get(skill_app,(-1,0))
        if(index != -1):
            for skill_app, (ind, reward) in self.examples.items():
                if(ind > index):
                    self.examples[skill_app] = (ind-1, reward)
            del self.examples[skill_app]
            did_change = True
            self.rebase_examples()

        return index, did_change



# --------------------------------------------
# : VectorTransformMixin

class VectorTransformMixin(RefittableMixin):
    def __init__(self, skill, encode_relative=True, one_hot=False,
                encode_missing=None,
                starting_state_format='flat_featurized',
                extra_features=[],
                 **kwargs):
        super().__init__(skill,**kwargs)
        self.starting_state_format = 'flat_featurized'
        self.encode_relative = encode_relative
        self.extra_features = extra_features
        self.one_hot = one_hot
        
        agent = skill.agent

        # Encode Missing By Default
        if(encode_missing is None):
            self.encode_missing = one_hot

        # Initialize Vectorizer
        from numba.types import f8, i8, string, boolean
        from cre.transform import Vectorizer
        self.vectorizer = Vectorizer([f8, string, boolean], one_hot, self.encode_missing)

        if(self.agent.enumerizer):

            # Recovers original keys and values before enumerization and vectorization  
            def inv_mapper(key_ind, val_nom):
                from cre import Var, CREFunc
                from cre.tuple_fact import TupleFactProxy
                if(self.one_hot):
                    key, nom = self.vectorizer.unvectorize(key_ind)
                else:
                    key, nom = self.vectorizer.unvectorize(key_ind, val_nom)

                if(isinstance(key, TupleFactProxy)):
                    key_head = key[0]
                    if(isinstance(key_head, (Var, CREFunc))):
                        typ = key_head.return_type

                    # TODO: Perhaps there is a less hardcoded way of doing this.
                    elif(key_head == "SkillCand:" or key_head == "SkillValueCount:"):
                        typ = key[1].return_type
                else:
                    typ = key.return_type

                if((not self.one_hot or self.encode_missing) and nom == 0):
                    val = None
                else:
                    val = self.agent.enumerizer.from_enum(nom, typ)
                return key, val
            self.inv_mapper = inv_mapper

        # Initialize or retrive relative_encoder
        if(encode_relative):
            from cre.transform import RelativeEncoder
            # TODO: Backup won't work without fact_types
            self.relative_encoder = RelativeEncoder(agent.fact_types)

        self.X_nom = np.empty((0,0), dtype=np.int64)
        self.Y = np.empty(0, dtype=np.int64)

    def transform(self, state, match):
        featurized_state = state.get("flat_featurized")

        # if(not self.encode_relative):
        featurized_state = featurized_state.copy()
        agent = self.skill.agent

        
        for extra_feature in self.extra_features:
            featurized_state = extra_feature(self, state, featurized_state, match)


        wm = state.get("working_memory")

        if(self.encode_relative):
            self.relative_encoder.set_in_memset(wm)
            _vars = self.skill.where_lrn_mech._ensure_vars(match)
            # _vars = self.skill.where_lrn_mech._ensure_vars(match)
            # print(":::", self.skill.id_num, "_vars", _vars[0].base_ptr)

            # print(featurized_state)

            # Featurize state relative to selection
            #  NOTE: Could also use arguments, but there is currently a hard
            #  to find bug associated with this
            # print(match[0], type(match[0]))
            # print(featurized_state)
            featurized_state = self.relative_encoder.encode_relative_to(
                featurized_state, [match[0]], [_vars[0]])
        # print("vvvvvvvvvvvvvvvvvvvvvvvvvv")
        # print(featurized_state)
        # print("^^^^^^^^^^^^^^^^^^^^^^^^^^")
        # if(repr(self.skill.how_part) == "NumericalToStr(TensDigit(Add3(CastFloat(a.value), CastFloat(b.value), CastFloat(c.value))))" and
        #    "S_Qr9" in state.get('__uid__')):
            # print('--------------------------------------')
            # print(self.conds)
            # print(matches)
            # print('--------------------------------------')
            

        continuous, nominal = self.vectorizer(featurized_state)
        #### -------Print mapping------------####
        # print("---------------------------------------")
        # for (ind, val) in self.vectorizer.make_inv_map().items():
        #     print("*", ind, nominal[ind], ind,val)
        # ind_vals = sorted([(ind, str(val)) for (ind, val) in self.vectorizer.get_inv_map().items()],
        #                 key=lambda t : t[1])
        # for ind, val in ind_vals:
        #     print(ind, ":", nominal[ind], val)
        # print("---------------------------------------")
        #####
        return continuous, nominal

    def insert_transformed(self, transformed_state, skill_app, reward, index):
        continuous, nominal = transformed_state

        n, m = self.X_nom.shape
        new_shape = (max(n, index+1), max(m, len(nominal)))
        print("NEW SHAPE", new_shape, n, index+1, m, len(nominal))
        if(new_shape != self.X_nom.shape):
            # Copy old data into new matrix
            new_X_nom = np.zeros(new_shape, dtype=np.int64)
            new_Y = np.zeros(new_shape[0], dtype=np.int64)
            new_X_nom[:n, :m] = self.X_nom
            new_Y[:n] = self.Y

            # Copy old data into new training matrix
            self.X_nom = new_X_nom
            self.Y = new_Y

        self.X_nom[index] = nominal
        self.Y[index] = reward
        # print(self.X_nom,  self.Y)

# --------------------------------------------
# : SklearnDecisionTree

# @register_when(name="decisiontree")
@register_when
class SklearnDecisionTree(BaseWhen, VectorTransformMixin):
    def __init__(self, skill, **kwargs):
        super().__init__(skill,**kwargs)
        from sklearn.tree import DecisionTreeClassifier

        VectorTransformMixin.__init__(self, skill, one_hot=True, **kwargs)
        self.classifier = DecisionTreeClassifier()

    def ifit(self, state, skill_app, reward):
        self.add_example(state, skill_app, reward) # Insert into X_nom, Y
        self.classifier.fit(self.X_nom, self.Y) # Re-fit

    def predict(self, state, match):
        continuous, nominal = self.transform(state, match)
        X_nom_subset = nominal[:self.X_nom.shape[1]].reshape(1,-1)
        prediction = self.classifier.predict(X_nom_subset)[0]        
        return prediction

# --------------------------------------------
# : DecisionTree (i.e. STAND library implementation)

@register_when
class DecisionTree(BaseWhen, VectorTransformMixin):
    def __init__(self, skill, impl="decision_tree",
                **kwargs):
        super().__init__(skill, **kwargs)
        from stand.tree_classifier import TreeClassifier

        VectorTransformMixin.__init__(self, skill, one_hot=False, **kwargs)
        self.classifier = TreeClassifier(impl, inv_mapper=self.inv_mapper)

    def ifit(self, state, skill_app, reward):
        self.add_example(state, skill_app, reward) # Insert into X_nom, Y
        self.classifier.fit(self.X_nom, None, self.Y) # Re-fit
        # print(self.classifier)

    def predict(self, state, match):
        continuous, nominal = self.transform(state, match)
        X_nom_subset = nominal[:self.X_nom.shape[1]].reshape(1,-1)
        prediction = self.classifier.predict(X_nom_subset, None)[0]        
        return prediction

    def __str__(self):
        return str(self.classifier)


# --------------------------------------------
# : STAND

@register_when
class STAND(BaseWhen, VectorTransformMixin):
    def __init__(self, skill,
                **kwargs):
        super().__init__(skill, **kwargs)
        from stand.stand import STANDClassifier

        VectorTransformMixin.__init__(self, skill, one_hot=False, **kwargs)
        self.classifier = STANDClassifier(inv_mapper=self.inv_mapper, **kwargs)

    def ifit(self, state, skill_app, reward):
        self.add_example(state, skill_app, reward) # Insert into X_nom, Y
        ia = self.classifier.instance_ambiguity(self.X_nom[-1], None)
        print("WILL LEARN", ia > 0, ia)
        self.classifier.fit(self.X_nom, None, self.Y) # Re-fit
        # print(self.classifier)
        

    def predict(self, state, match):
        continuous, nominal = self.transform(state, match)
        X_nom_subset = nominal[:self.X_nom.shape[1]].reshape(1,-1)
        # prediction = self.classifier.predict(X_nom_subset, None)[0]
        ia = self.classifier.instance_ambiguity(X_nom_subset[-1], None)
        # print("IA", ia)

        probs = self.classifier.predict_prob(X_nom_subset, None)[0]

        for a in probs:
            if(a['y_class']==1):
                prob = a['prob']
                if(prob > 0):
                    return prob
            elif(a['y_class']==-1):
                return -a['prob']
                # return a['y_class']
        return 1

    def __str__(self):
        return str(self.classifier)
        # print(probs)

        # return prediction
