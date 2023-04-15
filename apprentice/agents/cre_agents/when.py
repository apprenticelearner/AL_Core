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

    def get_info(self, **kwargs):
        return {}


# ------------------------------------------------------------------------
# : When Learning Mechanisms


# --------------------------------------------
# : VectorTransformMixin

class VectorTransformMixin():
    def __init__(self, skill, encode_relative=False, one_hot=False,
                encode_missing=None,
                starting_state_format='flat_featurized',
                extra_features=[],
                 **kwargs):
        self.starting_state_format = 'flat_featurized'
        self.encode_relative = encode_relative
        self.extra_features = extra_features
        self.one_hot = one_hot
        
        agent = skill.agent

        # Encode Missing By Default
        if(encode_missing is None):
            self.encode_missing = one_hot


        # Initialize or retrive vectorizer
        # if(hasattr(agent, 'vectorizer')):
        #     self.vectorizer = agent.vectorizer
        # else:
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
                    elif(key_head == "SkillCand:"):
                        typ = key[1].return_type
                else:
                    typ = key.return_type

                if((not self.one_hot or self.encode_missing) and nom == 0):
                    val = "MISSING"
                else:
                    val = self.agent.enumerizer.from_enum(nom, typ)
                return key, val
            self.inv_mapper = inv_mapper

        # if(one_hot):
        #     from sklearn.preprocessing import OneHotEncoder
        #     self.one_hot_encoder = OneHotEncoder(sparse=False)

        # Initialize or retrive relative_encoder
        if(encode_relative):
            # if(hasattr(agent, 'relative_encoder')):
            #     self.relative_encoder = agent.relative_encoder
            # else:
            from cre.transform import RelativeEncoder
            # TODO: Backup won't work without fact_types
            self.relative_encoder = RelativeEncoder(agent.fact_types)

        self.X = []
        self.Y = []
        self.X_width = 0
        self.bloop = []

    def transform(self, state, match):
        from cre import TF
        from cre.gval import new_gval
        featurized_state = state.get("flat_featurized")

        # if(not self.encode_relative):
        featurized_state = featurized_state.copy()
        agent = self.skill.agent

        
        for extra_feature in self.extra_features:
            featurized_state = extra_feature(self, state, featurized_state, match)

        # print(featurized_state)
        # Add skill app candidates
        # for skill in agent.skills:
        #     for match in skill.where_lrn_mech.get_matches(state):
        #         val = list(skill(*match).inputs.values())[0]
        #         head = TF("Skill:", skill.how_part, skill.id_num, *match)
        #         gval = new_gval(head, val)
        #         featurized_state.declare(gval)

        # Add match
        # sel_tup = new_gval(TF("Sel:", match[0]), "")
        # featurized_state.declare(sel_tup)
        # if(len(match) > 1):
        #     for m in match[1:]:
        #         arg_tup = new_gval(TF("arg{}:", m),"")
        #         featurized_state.declare(arg_tup)



        # for skill in agent.skills:
        #     for match in skill.where_lrn_mech.get_matches(state):
        #         val = list(skill(*match).inputs.values())[0]
        #         head = TF("Skill:", skill.how_part, skill.id_num, *[m.id for m in match])
        #         gval = new_gval(head, val)
                

        # self.bloop.append(featurized_state)
        # for fact in featurized_state:
        #     print(repr(fact), hash(fact))
        # else:
        wm = state.get("working_memory")
        # print("WM BEF RE RECOUNT", wm._meminfo.refcount)
        # print(wm)
        if(self.encode_relative):
            self.relative_encoder.set_in_memset(wm)
            _vars = self.skill.where_lrn_mech._ensure_vars(match)
            # print(":::", self.skill.id_num, "_vars", _vars[0].base_ptr)

            # print(featurized_state)

            # Featurize state relative to selection
            #  NOTE: Could also use arguments, but there is currently a hard
            #  to find bug associated with this
            featurized_state = self.relative_encoder.encode_relative_to(
                featurized_state, [match[0]], [_vars[0]])
            # print(featurized_state)
            # featurized_state, match, _vars)

        # print(shorthand_state_rel(featurized_state))
            
        
        # print("WM AFT RE RECOUNT", wm._meminfo.refcount)            

            # if(self.skill.id_num == 0):
            #     print(featurized_state)


            # print(state.get("flat"))
        # print(featurized_state)
        # fs = sorted([str(gval) for gval in featurized_state])
        # for q in fs:
        #     print(q)


        continuous, nominal = self.vectorizer(featurized_state)

        # print(nominal.shape)


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


        #     a = d.get(str(val),[])
        #     a.append((ind, val)) 
        #     d[str(val)] = a

        # for a in d.values():
        #     if(len(a) > 1):
        #         for ind, val in a:
        #             try:
                        
        #                 di = val.deref_infos
        #                 print(ind, val, di, val.base_ptr, val.get_ptr())
        #             except:
        #                 pass
        #         break
                
        # print(hash(match[0]))
        # print(nominal)

        # print("nominal", nominal)

        # if(hasattr(self, 'one_hot_encoder')):
        #     self.one_hot_encoder.ifit(nominal)
        #     nominal = self.one_hot_encoder.transform([nominal])[0]

        # print("one_hot", nominal)

        #TODO should also have a seperate width for continous
        

        return continuous, nominal

    def append_and_flatten_vecs(self, state, match, reward):
        # with PrintElapse("\ttransform"):

        continuous, nominal = self.transform(state, match)


        # with PrintElapse("\tcopy_array"):
        self.X_width = max(len(nominal),self.X_width)
        
        self.X.append(nominal)
        self.Y.append(1 if reward > 0 else -1)

        X = np.zeros((len(self.X), self.X_width), dtype=np.int64)
        for i, x in enumerate(self.X):
            X[i, :len(x)] = x

        # self.bloop.append(state.get("flat_featurized"))
        # for b in self.bloop:
        #     print(b)

        



        # print("X shape:", X.shape)

        Y = np.array(self.Y, dtype=np.int64)

        # for i, (x, y) in enumerate(zip(X, Y)):
        #     print(x,y)
        # print(Y)
        return X, Y



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
        self.X = []
        self.Y = []

    def ifit(self, state, match, reward):
        X,Y = self.append_and_flatten_vecs(state, match, reward)
        self.classifier.fit(X, Y)

    def predict(self, state, match):
        continuous, nominal = self.transform(state, match)
        prediction = self.classifier.predict(nominal[:self.X_width].reshape(1,-1))[0]
        return prediction


np.set_printoptions(edgeitems=300000, linewidth=10000000)

@register_when
class DecisionTree(BaseWhen, VectorTransformMixin):
    def __init__(self, skill, impl="decision_tree",
                **kwargs):
        super().__init__(skill, **kwargs)
        from stand.tree_classifier import TreeClassifier

        VectorTransformMixin.__init__(self, skill, one_hot=False, **kwargs)
        self.classifier = TreeClassifier(impl, inv_mapper=self.inv_mapper)
        # self.X = []
        # self.Y = []

    def ifit(self, state, match, reward):
        X,Y = self.append_and_flatten_vecs(state, match, reward)

        # print("AAAAH")
        # print("fit", X[-1], reward)

        # print(f"T{self.skill.id_num} {[m.id for m in match]}\t", int(reward), X[-1])
        
        # print(X[-1])
        # print(Y)
        # with PrintElapse("A"):
        self.classifier.fit(X, None, Y)
        print(self.classifier)



    def predict(self, state, match):
        continuous, nominal = self.transform(state, match)
        prediction = self.classifier.predict(nominal[:self.X_width].reshape(1,-1), None)[0]
        # if(self.skill.id_num == 4):
        #     print(f"P{self.skill.id_num} {[m.id for m in match]}\t", prediction, nominal[:self.X_width])

        # print(self.skill, match)
        # print("predict", nominal[:self.X_width].reshape(1,-1), self.classifier.predict(nominal[:self.X_width].reshape(1,-1))[0])

        return prediction






from .debug_utils import shorthand_state_rel


@register_when
class STAND(BaseWhen, VectorTransformMixin):
    def __init__(self, skill,
                **kwargs):
        super().__init__(skill, **kwargs)
        from stand.stand import STANDClassifier

        VectorTransformMixin.__init__(self, skill, one_hot=False, **kwargs)
        self.classifier = STANDClassifier()

    def ifit(self, state, match, reward):
        X,Y = self.append_and_flatten_vecs(state, match, reward)

        from stand.fnvhash import hasharray

        
                
        
        # print(X[-1])
        # print(f"{self.skill.id_num} fit: ", hasharray(X[-1]))
        # print(shorthand_state_wm(state))
        # print(shorthand_state_flat(state))
        # print("AAAAH")
        # print("fit", X[-1], reward)

        # print(f"T{self.skill.id_num} {[m.id for m in match]}\t", int(reward), X[-1])
        
        # print(X)
        # print(Y)
        # try:
        # print("xf", X[-1])
        print("IA", self.classifier.instance_ambiguity(X[-1], None))
        # except Exception as e:
        #     print(self.classifier)
        #     raise e

        # try:
        self.classifier.fit(X, None, Y)
        # print(self.classifier)
        # except Exception as e:
        #     print(self.classifier)
        #     raise e


    def predict(self, state, match):
        from stand.fnvhash import hasharray


        # print([(x.id, get_value(x)) for x in state.get("working_memory").get_facts()])
        

        continuous, nominal = self.transform(state, match)
        # try:
        #     print("SKILL", self.skill.id_num)

        
        # print(nominal)    
        # print("xp", nominal)

        prediction = self.classifier.predict(nominal[:self.X_width].reshape(1,-1), None)[0]

        # print(shorthand_state_wm(state))
        # print(shorthand_state_flat(state))
        # print(f"{self.skill.id_num} {prediction} pred:", hasharray(nominal), [m.id for m in match])
        
            # if(prediction > 0):
        
            # print()    
        # except Exception as e:
        #     print(self.classifier)
        #     raise e 
        # if(self.skill.id_num == 4):
        #     print(f"P{self.skill.id_num} {[m.id for m in match]}\t", prediction, nominal[:self.X_width])

        # print(self.skill, match)
        # print("predict", nominal[:self.X_width].reshape(1,-1), self.classifier.predict(nominal[:self.X_width].reshape(1,-1))[0])

        return prediction
