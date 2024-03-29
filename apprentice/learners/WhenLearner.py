from pprint import pprint
from copy import deepcopy
from apprentice.learners.pyibl import Agent
import numpy as np
import re, json

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
from concept_formation.preprocessor import Preprocessor
from stand.tree_classifier import TreeClassifier

# from ilp.foil_classifier import FoilClassifier

# cobweb, pyibl, nearest neighbor, logistic regression


class WhenLearner(object):
    STATE_FORMAT_OPTIONS = ["variablized_state",  "state_only"]
    WHEN_TYPE_OPTIONS = ["one_learner_per_rhs", "one_learner_per_label"]
    CROSS_RHS_INFERENCE = ["none", "implicit_negatives", "rhs_in_y"]

    def __init__(self, agent, learner, when_type="one_learner_per_rhs",
                 state_format="variablized_state",
                 cross_rhs_inference="none",
                 **learner_kwargs):
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

        self.agent = agent
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
        self.sub_learners = {}

    # def all_where_parts(self,state):
    #     for rhs in self.sub_learners.keys():
    #         for match in self.where_learner.get_matches(rhs, state):
    #             if(len(match) != len(set(match))):
    #                 continue
    #             yield rhs,match 

    def add_rhs(self, rhs):
        if(self.type == "one_learner_per_rhs"):
            self.sub_learners[rhs] = get_when_sublearner(self.learner_name,
                                                rhs,**self.learner_kwargs)
            key = rhs
        else:
            key = rhs.label
        rhs_list = self.rhs_by_label.get(rhs.label, [])
        rhs_list.append(rhs)
        self.rhs_by_label[rhs.label] = rhs_list

        if(self.cross_rhs_inference == "implicit_negatives"):
            self.examples[key] = {}
            self.examples[key]['base_state'] = []
            self.examples[key]['state'] = []
            self.examples[key]['reward'] = []
            self.implicit_examples[key] = {}
            self.implicit_examples[key]['base_state'] = []
            self.implicit_examples[key]['overridden'] = []
            self.implicit_examples[key]['state'] = []
            self.implicit_examples[key]['reward'] = []
            self.implicit_examples[key]['mapping'] = []

    def ifit(self, rhs, state, mapping, reward):
        # print("FIT_STATe", state)
        # print([str(x.input_rule) for x in self.sub_learners.keys()])
        # print("REQARD", reward)
        # print("LEARNERS",self.sub_learners)
        
        if(self.cross_rhs_inference == "implicit_negatives"):
            base_state = state.get_view('flat_ungrounded')
            base_state = {k:v for k,v in base_state.items() if (k[0]=='contentEditable' and ('out' == k[1][:3] or 'carry' == k[1][:5]))}
            base_state = [0 if(v) else 1 for v in base_state.values()]
            # print(base_state)
            this_state = state.get_view(("variablize",rhs,tuple(mapping)))
            # pprint(this_state)
            if(self.type == "one_learner_per_label"):
                key = rhs.label
            elif(self.type == "one_learner_per_rhs"):
                key = rhs


            base_states = self.examples[key]['base_state']
            states = self.examples[key]['state']
            rewards = self.examples[key]['reward']

            base_states.append(base_state)
            states.append(this_state)
            rewards.append(reward)

            if(key not in self.sub_learners):
                self.sub_learners[key] = get_when_sublearner(self.learner_name,
                                                    rhs, **self.learner_kwargs)
            # pprint(states)
            # pprint(rewards)

            implicit_base_states = self.implicit_examples[key]['base_state']
            implicit_states = self.implicit_examples[key]['state']
            implicit_rewards = self.implicit_examples[key]['reward']

            # Remove any implicit negative examples in this rhs with the current state
            to_remove = []
            for i,bs in enumerate(self.implicit_examples[key]['base_state']):
                if(bs == base_state): to_remove.append(i)
                    
            if(len(to_remove) > 0):
                print("To REMOVE", to_remove)
            for i in reversed(to_remove):
                del self.implicit_examples[key]['base_state'][i]
                del self.implicit_examples[key]['state'][i]
                del self.implicit_examples[key]['reward'][i]
                del self.implicit_examples[key]['mapping'][i]
            

                
            
            self.implicit_examples[key]['overridden'].append(base_state)

            print("MAIN:",str(rhs))
            self.sub_learners[key].fit(states+implicit_states,
                                   rewards+implicit_rewards)

            all_matches = {}
            for k in state.views.keys():
                if(isinstance(k,tuple) and k[0] == "variablize"):
                    # print("K",k)
                    matches = all_matches.get(k[1],[])
                    matches.append(k[2])
                    all_matches[k[1]] = matches
            # print("all_matches")
            # pprint(all_matches)
            # print("implicit_rewards")
            # print(implicit_rewards)
            if(reward > 0):
                for other_key, other_impl_exs in self.implicit_examples.items():
                    
                    if(other_key != key):
                        # print("OTHER:",str(other_key))
                        # print(other_key, key)
                        # Add implicit negative examples to any rhs that doesn't already have this state
                        # TODO: Do this for all bindings not just for the given state
                        # other_state = state.get_view(("variablize",other_key,tuple(mapping)))
                        # pprint(other_impl_exs['overridden'])
                        if(base_state not in other_impl_exs['overridden'] and 
                            base_state not in other_impl_exs['base_state']):
                            used_sels = set()
                            for m in all_matches.get(other_key,[]):
                            # print("other_map: ",m)
                            
                            # print("other_state:")
                            # pprint(other_state)
                                print("IMPL NEG", other_key, m)
                            
                                other_state = state.get_view(("variablize",other_key,m))
                                # print("BEFORE",other_impl_exs['reward'])
                                other_impl_exs['base_state'].append(base_state)
                                other_impl_exs['state'].append(other_state)
                                other_impl_exs['reward'].append(-1)
                                other_impl_exs['mapping'].append(m)
                                if(m[0] not in used_sels):
                                    self.agent.which_learner.ifit(other_key, other_state, m, -1)    
                                    used_sels.add(m[0])
                                # print("AFTER",other_impl_exs['reward'])
                        # print("TOTAL:",len(self.examples[other_key]['reward']))
                        # print("IMPL:",len(other_impl_exs['reward']))
                            self.sub_learners[other_key].fit(self.examples[other_key]['state']+other_impl_exs['state'],
                                                         self.examples[other_key]['reward']+other_impl_exs['reward'])

               

            # PRINT AREA
            # for x in self.implicit_examples:
            #     print("%s : %s" % (x.input_rule,self.implicit_examples[x]['reward']))
            # for x in self.examples:
            #     print("%s : %s" % (x.input_rule,self.examples[x]['reward']))
            # pprint(self.implicit_examples)

            
            # print(self.sub_learners[key])
        else:
            state = state.get_view(("variablize",rhs,tuple(mapping)))
            # print("WHEN LEARNER")
            pprint(state)
            if(self.type == "one_learner_per_label"):
                if(rhs.label not in self.sub_learners):
                    self.sub_learners[rhs.label] = get_when_sublearner(
                                                self.learner_name,
                                                rhs,
                                                **self.learner_kwargs)
                if(self.cross_rhs_inference == "rhs_in_y"):
                    self.sub_learners[rhs.label].ifit(state, (rhs._id_num, reward))
                else:
                    self.sub_learners[rhs.label].ifit(state, reward)
            elif(self.type == "one_learner_per_rhs"):
                # print("------------------")
                # print([str(x) for x in self.sub_learners.keys()])
                # print([id(x) for x in self.sub_learners.values()])
                # print("FIT:", str(rhs), reward)
                # print("------------------")
                self.sub_learners[rhs].bloop = getattr(self.sub_learners[rhs], "bloop", [])
                self.sub_learners[rhs].bloop.append(state)

                # for b in self.sub_learners[rhs].bloop:
                #     pprint(b)

                self.sub_learners[rhs].ifit(state, reward)

    def predict(self, rhs, state):
        # print("STATE:",state, type(state))
        # print("------------")
        # print(str(rhs))
        if(self.type == "one_learner_per_label"):
            prediction = self.sub_learners[rhs.label].predict([state])[0]
        elif(self.type == "one_learner_per_rhs"):
            # print(self.sub_learners[rhs].predict([state])[0]        )
            prediction = self.sub_learners[rhs].predict([state])[0]
            # print("X")

            # print("-")# print(self.sub_learners[rhs].X)
            # print("y")
            # print(self.sub_learners[rhs].y)
        # print("--->",prediction)
        # print("------------")

        # print("BLLEEPERS")
        # print([str(x) for x in self.sub_learners.keys()])
        if(self.cross_rhs_inference == "rhs_in_y"):
            rhs_pred, rew_pred = prediction
            if(rhs_pred != rhs._id_num):
                rew_pred = -1
            return rew_pred
        else:
            return prediction

    def skill_info(self, rhs, state):
        key = rhs if(self.type == "one_learner_per_rhs") else rhs.label
        sublearner = self.sub_learners[key]

        if(isinstance(sublearner, Pipeline)):
            feature_names = sublearner.named_steps["dict vect"].get_feature_names()

        return sublearner.skill_info(state)

class ListValueFlattener(Preprocessor):
    def transform(self, instance):
        if isinstance(instance,(list,tuple)):
            return [self._transform(x) for x in instance]
        else:
            return self._transform(instance)
        

    def _transform(self,instance):
        out = {}
        for key,value in instance.items():
            if(isinstance(value,(list,tuple))):
                for i, v in enumerate(value):
                    out[(key,str(i))] = v if v is not None else ""
            else:
                out[key] = value if value is not None else ""
        return out 


    def undo_transform(self, instance):
        raise NotImplementedError()

def tree_condition_inds(tree):
    if(tree is None or len(tree.nodes) <= 1): return []
    #start w/ leaves and go up
    # nodes = [n for n in tree.nodes if n.ttype == 2 and n.counts[1] > n.counts[0]]

    next_nodes = [(tree.nodes[0], [])]
    chains = []
    while(len(next_nodes) > 0):
        next_next_nodes = []
        for i, (node,chain) in enumerate(next_nodes):
            if(node.ttype == 1):
                split_on, ithresh, left, right, nan = node.split_data[0]

                next_next_nodes.append((tree.nodes[left],chain + [-split_on]))
                next_next_nodes.append((tree.nodes[right],chain+ [split_on]))
            elif(len(node.counts) == 2 and node.counts[1] > node.counts[0]):
                chains.append((f'{node.counts[1]}'.zfill(3), chain))
        next_nodes = next_next_nodes
    for c in chains:
        print(c)
    # pprint(chains)



# class DecisionTree2(TreeClassifier):
class DecisionTree2(object):
    # def __init__(self, impl="decision_tree", use_missing=False):
    def __init__(self, impl="sklearn", use_missing=False):
        # print("IMPL:",impl)
        if(impl == "sklearn"):
            self.dt = DecisionTreeClassifier()
        else:
            self.dt = TreeClassifier(impl)
        
        self.impl = impl
        self.X = []
        self.y = []
        self.slots = {}
        self.inverse = []
        self.slots_count = 0
        self.X_list = []
        self.use_missing = use_missing
        

    def _designate_new_slots(self,x):
        ''' Makes new slots for unseen keys and values'''
        for k, v in x.items():
            if(k not in self.slots):
                vocab = self.slots[k] = {chr(0) : self.slots_count}         
                self.slots_count += 1
                self.inverse.append(f'!{k}')
            else:
                vocab = self.slots[k]

            if(v not in vocab): 
                vocab[v] = self.slots_count
                self.slots_count += 1
                self.inverse.append(f'{k}=={v}')

    def _dict_to_onehot(self,x,silent_fail=False):
        x_new = [0]*self.slots_count
        for k, vocab in self.slots.items():
            # print(k, vocab)
            val = x.get(k,chr(0))
            if(silent_fail):
                if(val in vocab): x_new[vocab[val]] = 1
            else:
                x_new[vocab[val]] = 1
        return np.asarray(x_new,dtype=np.bool)

    def _gen_feature_weights(self, strength=1.0):
        weights = [0]*self.slots_count
        for k, vocab in self.slots.items():
            # print(k, vocab)
            w = (1.0-strength) + (strength * (1.0/max(len(vocab)-1,1)))
            for val,ind in vocab.items():
                weights[ind] = w

        return np.asarray(weights,dtype=np.float64)


    def _compose_one_hots(self):
        X = np.empty( (len(self.X_list), self.slots_count), dtype=np.uint8)
        missing_vals = [None]*len(self.X_list)
        for i, one_hot_x in enumerate(self.X_list):
            X[i,:len(one_hot_x)] = one_hot_x
            X[i,len(one_hot_x):] = 2 if self.use_missing else 0 # missing

            # miss = np.empty((self.slots_count-len(one_hot_x),2),dtype=np.int64)
            # miss[:,0] = i
            # miss[:,1] = np.arange(len(one_hot_x),self.slots_count)
            # missing_vals[i] = miss
        # missing_vals = np.concatenate(missing_vals)
        return X



    def ifit(self, x, y):
        self._designate_new_slots(x)
        one_hot_x = self._dict_to_onehot(x)

        #### -------Print mapping------------####
        # print("L:",len(one_hot_x))
        # for i, x in enumerate(one_hot_x):
        #     print(int(x), self.inverse[i])
        #### --------------------------------####
            # print(X_mat.shape)
            # print(x_mat.shape)
            # print(np.matmul(X_mat, x_mat))

        self.X_list.append(one_hot_x)
        self.X = self._compose_one_hots()

        
        # print(self.X)
        
        self.y.append(int(y) if not isinstance(y, tuple) else y)
        Y = np.asarray(self.y,dtype=np.int64)

        self.fit(self.X,Y)

    def fit(self, X, Y):
        if(not isinstance(X,np.ndarray)):
            self.X_list = []
            for x in X:
                self._designate_new_slots(x)
                self.X_list.append(self._dict_to_onehot(x))
            self.X = X = self._compose_one_hots()

        # self.Y = Y
        # print("W:",self._gen_feature_weights())

        Y = np.asarray(Y,dtype=np.int64)

        # if(len(X) > 20):
        #     print("CUTTING")
        #     X = X[10:]
        #     Y = Y[10:]

        # print(X.shape, Y.shape)

        if(self.impl == "sklearn"):
            return self.dt.fit(X, Y)
        else:
            tree_str = str(self.dt) if getattr(self.dt, "tree",None) is not None else ''
            # [n.split_on for n in self.dt.tree.nodes]
            inds = [int(x.split(" : (")[1].split(")")[0]) for x in re.findall(r'NODE.+',tree_str)]

            print()
            print("---", self.rhs, "---")
            tree_condition_inds(self.dt.tree)
            # print(tree_str)

            
            if(False):
                ft_weights = self._gen_feature_weights()
                print(json.dumps({ind: str(self.inverse[ind])+f"(w:{ft_weights[ind]:.2f})" for ind in inds},indent=2)[2:-2])
            else:
                ft_weights = np.ones((X.shape[1]),dtype=np.float64)
                print(json.dumps({ind: str(self.inverse[ind]) for ind in inds},indent=2)[2:-2])
            # print(json.dumps({ind: str(self.inverse[ind]) for ind in inds},indent=2)[2:-2])
            

            # return self.dt.fit(X, None, Y, None)
            return self.dt.fit(X, None, Y, None, ft_weights)


    def predict(self, X):

        onehot_X = np.empty((len(X), self.slots_count),dtype=np.bool)
        for i, x in enumerate(X):
            # self._designate_new_slots(x)
            onehot_x = self._dict_to_onehot(x,silent_fail=True)


            # if(len(self.X) > 0):
            #     # print("MATMUL", self.rhs)
            #     X_mat = np.asarray(self.X)

            #     inf_gain = self.dt.inf_gain(self.X, None, np.asarray(self.y,dtype=np.int64))
            #     inf_gain = inf_gain > 0.0

            #     # inf_gain /= np.linalg.norm(inf_gain)
            #     # print(inf_gain)


            #     x_mat = np.expand_dims(inf_gain*onehot_x[:X_mat.shape[1]],1)/np.sum(inf_gain)
            #     inner = np.matmul(X_mat, x_mat)[:,0]
            #     # print(inner)
            #     max_inner = np.max(inner)
            #     nearestNs = (inner >= max_inner *.95).nonzero()[0]
            #     nn_ys = [_y for i, _y in enumerate(self.y) if i in nearestNs]
            #     nn_pred = sum(nn_ys)
            #     if(nn_pred < 0):
            #         nn_pred = -1  
            #     elif(nn_pred > 0):
            #         nn_pred = 1;
            #     else:
            #         nn_pred = -0


            onehot_X[i] = onehot_x
        # onehot_X = np.concatenate(onehot_X,dtype=np.bool)


        if(self.impl == "sklearn"):
            pred = self.dt.predict(onehot_X)
        else:
            # print("PRED:",self.rhs, self.dt.predict(onehot_X,None))
            pred = self.dt.predict(onehot_X,None)

        # if(pred[0] != nn_pred and nn_pred != 0 ):
        #     if(nn_pred == -1):
        #         pred[0] = nn_pred
        #     print("---------------")
        #     print(self.rhs)
        #     print(max_inner, "pred:", pred[0], "nn_pred:", nn_pred, nn_ys)
        #     print("---------------")
        return pred




class CustomPipeline(Pipeline):

    def ifit(self, x, y):
        # print(x)
        if not hasattr(self, 'X'):
            self.X = []
        if not hasattr(self, 'y'):
            self.y = []

        ft = Flattener()
        tup = Tuplizer()
        lvf = ListValueFlattener()

        x = lvf.transform(x)
        x = tup.undo_transform(ft.transform(x))
        # pprint(x)`
        self.X.append(x)
        self.y.append(int(y) if not isinstance(y, tuple) else y)

        # print("IFIT:",self.X)
        # print(self.y)
        return super(CustomPipeline, self).fit(self.X, self.y)

    def fit(self, X, y):

        # print("X",X)
        # NOTE: Only using boolean values
        # X = [{k: v for k, v in d.items() if isinstance(v, bool)} for d in X]
        # print("FITX", X)
        # print("GIN JEF", X[-1])
        ft = Flattener()
        tup = Tuplizer()
        lvf = ListValueFlattener()

        X = lvf.transform(X)
        # print("GIN FEF", X[-1])

        self.X = [tup.undo_transform(ft.transform(x)) for x in X] 
        self.y = [int(x) if not isinstance(x, tuple) else x for x in y] 
        # print("GIN IN")
        # print(self.X[-1])
        # print("BLOOP:",len(self.y))
        super(CustomPipeline, self).fit(self.X, self.y)

    def predict(self, X):
        ft = Flattener()
        tup = Tuplizer()
        lvf = ListValueFlattener()

        X = lvf.transform(X)

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
        lvf = ListValueFlattener()

        X = lvf.transform(X)
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

from sklearn.base import TransformerMixin, BaseEstimator







def DictVectWrapper(clf):
    def fun(x=None):
        dv = DictVectorizer(sparse=False, sort=False)

        class PrintTransform(BaseEstimator, TransformerMixin):
            def fit(self, X, Y):
                return self

            def transform(self, X):
                # print(dv.inverse_transform(X))
                # try:
                #     for (k,v) in  dv.inverse_transform(X)[-1].items():
                #         print(":", k, v)
                # except:
                #     pass
                return X

            def inverse_transform(self, target):
                return target

        pt = PrintTransform()

        if x is None:
            return CustomPipeline([('dict vect', dv),
                                    ('pt', pt),
                                   ('clf', clf()),
                                   ])
        else:
            return CustomPipeline([('dict vect', dv),
                                    ('pt', pt),
                                   ('clf', clf(**x)),
                                   ])

    return fun


def SpecialDictVectWrapper(clf):
    def fun(**kwargs):
        return SpecialVectorizePipeline([('clf', clf(**kwargs))])
    return fun

    # def applicable_skills(self,state,skill_label,skills=None):
    #     if(skills == None): skills = self.skills_by_label[skill_label]
    #     raise NotImplementedError("Still need to write applicable_skills")


count = 0
def export_tree(dt):
    global count
    import pydotplus
    from sklearn.datasets import load_iris
    from sklearn import tree
    import collections


    # Visualize data
    dot_data = tree.export_graphviz(dt,
                                    feature_names=None,
                                    out_file=None,
                                    filled=True,
                                    rounded=True)
    graph = pydotplus.graph_from_dot_data(dot_data)

    colors = ('turquoise', 'orange')
    edges = collections.defaultdict(list)

    for edge in graph.get_edge_list():
        edges[edge.get_source()].append(int(edge.get_destination()))

    for edge in edges:
        edges[edge].sort()    
        for i in range(2):
            dest = graph.get_node(str(edges[edge][i]))[0]
            dest.set_fillcolor(colors[i])

    print("EXPORT",'trees/tree%s.png'%count)
    graph.write_png('trees/tree%s.png'%count)
    count += 1
    

from sklearn.tree import _tree
class DecisionTree(DecisionTreeClassifier):
    def fit(self,X,y):
        # if(len(y) > 1 and y[-1] > 0 and self.n_features_ == X.shape[1]): 
        #     print("SKIP FIT", len(y), self.n_features_, X.shape[1])
        #     return # Don't refit if positive
        # else:
        #     print("FIT", len(y), X.shape[1])
        # print("   X",len(X[0]))
        # pprint(X)
        # pprint(y)
        # print("y",len(y))
        # pprint(y)
        # print("--^--^--")
        # for b in self.bloop:
        #     print(b)
        # for x, _y in zip(X,y):
        #     print(x, _y)


        super(DecisionTree,self).fit(X,y)



        # print(hex(id(self)))
        # print("X")
        # print(X)
        # print("y",y)
        # export_tree(self)
        # print("------------")
        

    def predict(self, X):
        # print("PREDICT")
        # print(hex(id(self)))
        # print("X")
        # print(X)
        # export_tree(self)
        
        
        return super(DecisionTree, self).predict(X)

    def skill_info(self, examples, feature_names=None):
        # print("SLOOP", examples)
        tree = self
        tree_ = tree.tree_
        # print("feature_names", feature_names)
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
        x = deepcopy(dict(x))
        x['_y_label'] = float(y)
        self.tree.ifit(x)

    def fit(self, X, y):
        X = deepcopy(dict(X))
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
        x = deepcopy(dict(x))
        x['_y_label'] = float(y)
        self.tree.ifit(x)

    def fit(self, X, y):
        X = deepcopy(dict(X))
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


def get_when_sublearner(name, rhs, **kwargs):
    s = WHEN_CLASSIFIERS[name.lower().replace(' ', '').replace('_', '')](**kwargs)
    s.rhs = rhs
    return s


def get_when_learner(agent, name, **kwargs):
    inp_d = WHEN_LEARNERS[name.lower().replace(' ', '').replace('_', '')]
    inp_d.update(kwargs)
    return WhenLearner(agent, **inp_d)


WHEN_LEARNERS = {
    "decisiontree2": {"learner": "decisiontree2",
                     "state_format": "variablized_state"},
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
    'decisiontree2': DecisionTree2,
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
