import numpy as np
from itertools import permutations
from copy import copy
import warnings
from apprentice.agents.cre_agents.learning_mechs.where import Generalize
import abc

MAKE_WHERES = False

class SymbolFactory():
    def __init__(self, sym_ord=65, grammar=None):
        self.sym_ord = sym_ord
        self.grammar = grammar

    def __call__(self):
        while(True):
            if(self.sym_ord <= 90):
                name = chr(self.sym_ord)
            else:
                name = "NT" + str(self.sym_ord-90)
            self.sym_ord += 1
            if(self.grammar is None or 
                name not in self.grammar.macro_skills):
                break
        return name

# ---------------------------------------------------------
# : Skills

class SkillBase(object):
    def __init__(self, _id):
        assert isinstance(_id, str)
        self.skill_apps = []
        # self.matches = []
        self._id = _id
        self.agent = None
        self.where_lrn_mech = Generalize(self)
        self.n_vars = 0

    def __str__(self):
        return self._id

    __repr__ = __str__

    def merge(self, other):
        # Don't merge if "other" is already a skill_app of "self" 
        self_state_uids = [sa.state.get('__uid__') for sa in self.skill_apps]
        if(other.state.get("__uid__") in self_state_uids):
            return copy(self)

        new_skill = copy(self)
        new_skill.where_lrn_mech = Generalize(self)
        new_skill.skill_apps = []

        if(isinstance(other, SkillApp)):
            skill_apps = [other]            
        else:
            skill_apps = list(other.skill_apps)
        
        skill_apps = self.skill_apps + skill_apps
        # print("MERGE", self, [x.prob_uid for x in self.skill_apps], other.prob_uid)

        for sa in skill_apps:
            new_skill.add_app(sa)
            
        return new_skill

    def add_app(self, skill_app):
        match = skill_app.match

        var_names = None
        if(not isinstance(skill_app.skill, PrimSkill)):
            var_names = [f"Arg{i}" for i in range(len(skill_app.match))]
        
        if(skill_app not in self.skill_apps):
            # print("MATCH", self, [m.id for m in match])
            if(MAKE_WHERES):
                self.where_lrn_mech.ifit(skill_app.state, match, var_names=var_names)
                self.n_vars = len(self.where_lrn_mech.vars)
            self.skill_apps.append(skill_app)
        # else:
        #     print("ALREADY IN IT", self)
            # print(f"N VARS {self} == {self.n_vars}")

    def get_matches(self, state, match=None, min_score=.5, allow_partial=True):
        # with PrintElapse("get_matches"):
        whr_lrn = self.where_lrn_mech
        itr = None
        # print("**")
        # print(whr_lrn.conds)
        # print("**")
        out = []
        if(match is None):
            itr = whr_lrn.get_matches(state)
            out = [(1.0, m) for m in itr]
            print("DO NORMAL", self, len(out))
            # if(itr is not None):
            #     itr_len = len(itr) if isinstance(itr, list) else itr.est_len()[0]
        # print(state)
        # print("ITR", self, itr.est_len() if itr else None, match)
        print(allow_partial, match, len(out))
        if(match is not None or (allow_partial and len(out) == 0)):
            
            itr = whr_lrn.get_partial_matches(state, match)
            out = [(s, m) for s,m in itr if s > min_score]
            print("DO PARTIAL", self, len(out))
        if(len(out) > 0):
            print('GET MATCHES', [m.id for m in out[0][1]], f"1 of {len(out)}")
            # print("Backup", [(s,m.id) for s, m in out])

        return out

    # def arg_overlap(self, other):
    #     skill_apps = getattr(other, 'skill_apps', [other])
    #     m = 0.0
    #     for sa in skill_apps:
    #         v = max([x.arg_overlap(sa) for x in self.skill_apps])
    #         if(v > m):
    #             m = v
    #     return m    

    # @property
    # def n_vars(self):
    #     sa = next(iter(self.skill_apps))
    #     return len(sa.match)

# ------------------------
# : Primative Skills

class PrimSkill(SkillBase):
    def __init__(self, how_part=None, skill=None):
        self.how_part = how_part
        self.skill = skill
        super().__init__(how_part)

    def skill_eq(self, other):
        if(isinstance(other, SkillApp)):
            other = other.skill
        if(not isinstance(other, PrimSkill)):
            return False

        h_b = other
        if(hasattr(other, 'how_part')):
            h_b = other.how_part
        elif(hasattr(other, 'skill')):
            h_b = other.skill.how_part
        return self.how_part == h_b

    def overlap(self, other):
        skill_apps = getattr(other, 'skill_apps', [other])
        m = 0.0
        for sa in skill_apps:
            v = max([x.overlap(sa) for x in self.skill_apps])
            if(v > m):
                m = v
        return m

    def match_overlap(self, other):
        skill_apps = getattr(other, 'skill_apps', [other])
        m = 0.0
        for sa in skill_apps:
            v = max([x.match_overlap(sa) for x in self.skill_apps])
            if(v > m):
                m = v
        return m


    def get_apps(self, state, match=None, _depth=0):
        for score, match in self.get_matches(state, match):
            skill_app = SkillApp(macro, match, state=state, match_score=score, depth=_depth)
            yield skill_app

# ------------------------
# : Compound Skill Base

class CompoundSkillBase(SkillBase):
    def __init__(self, _id):
        self.child_arg_maps = []
        self.child_params = []
        super().__init__(_id)

    @property
    @abc.abstractmethod
    def child_skills(self):
        pass

    def _update_child_params(self):
        # print("REMAPS!", remap_info)
        child_skills = self.child_skills
        if(len(child_skills) == 0):
            return

        self_vars = self.where_lrn_mech.vars
        self_var_ptrs = np.array([x.get_ptr() for x in self_vars], dtype=np.int64)
        
        self.child_params = [None]*len(child_skills)
        self_conds = self.where_lrn_mech.conds

        # For each child fill in a set of param variables.
        for ind, child in enumerate(child_skills):
            arg_map = self.child_arg_maps[ind]                        
            child_vars = child.where_lrn_mech.vars
            params = [None]*len(child_vars)
            

            # print("##", ind, method, remap)
            # print("macvars", macro_vars)
            # print("methVars", method_vars)

            # Try to fill in param with direct mappings of 
            #   variables between the macro and method.
            for i, var_ind in enumerate(arg_map):
                # var_ind = arg_map[i]
                if(var_ind != -1):
                    params[i] = (var_ind, self_vars[var_ind])
            
            # print(arg_map, "S",len(self_vars), "C",len(child_vars))
            # For any remaining unmapped vars look for a literal in
            #   the method conditions that can map the variable by 
            #   dereferencing from an attribute of an already mapped var.

            known_var_ptrs = np.array([v.get_ptr() for i, v in enumerate(child_vars) if params[i]], dtype=np.int64)
            child_var_ptrs = np.array([v.get_ptr() for i, v in enumerate(child_vars)], dtype=np.int64)
            for i, child_var in enumerate(child_vars): 
                if(params[i] is None):
                    child_var_ptr = child_var.get_ptr()
                    param_var = find_adj_in_conds(
                        child.where_lrn_mech.conds,
                        child_var_ptr, 
                        known_var_ptrs#self_var_ptrs
                    )
                    base_ind = -1
                    if(param_var):
                        base_ptr = param_var.get_base_ptr()
                        base_ind = arg_map[np.nonzero(child_var_ptrs==base_ptr)[0][0]]
                    params[i] = (base_ind, param_var)

                    # params[i] = ()
                    # base_ptr = params[i].get_base_ptr()
                    # print(child_vars)
                    # print(params[i], base_ptr, known_var_ptrs)
                    # if(base_ptr in child_var_ptrs):
                    #     arg_map[i] = np.nonzero(child_var_ptrs==base_ptr)[0]
                    # print("IS NONE", params[i])
            # print([str(x) for x in macro_vars],"->", [str(x) for x in params])
            # print([m.id for m in list(method.skill_apps)[0].match])
            # print("PARAMS", self, "->", child, [str(x) for x in params])
            self.child_params[ind] = params

            # if(any([x is None for x in params])):
            #     raise ValueError()


# ------------------------
# : Macro (i.e. Abstract) Skills

# def drop_like_remap(remap, other_remap):
#     print(":", remap, len(remap), other_remap, len(other_remap))
#     return remap[other_remap!=-1]
    # return np.select([other_remap != -1],[remap],default=-1)

def invert_remap(remap, L_other):
    inv_remap = -np.ones(L_other,dtype=np.int16)
    for i, j in enumerate(remap):
        if(j != -1):
            inv_remap[j] = i
    return inv_remap

class MacroSkill(CompoundSkillBase):
    def __init__(self, _id, methods=[]):
        super().__init__(_id)
        self.methods = []
        # self.method_params = []
        # self.child_arg_maps = []
        for method in methods:
            assert isinstance(method, MethodSkill)
            self.add_method(method, do_update=False)
        self._update(is_init=True)

    @property
    def child_skills(self):
        return self.methods

    # def _check_id_overlap(self, remap, ):


    def _update_where_from_method(self, method, meth_ind, is_init=False):
        
        # Update where-part by incorperating a new or updated method
        assert meth_ind <= len(self.child_arg_maps)
        assert not is_init or meth_ind == 0 

        # print("MID", meth_ind, is_init)
        if(is_init):
            # If is initializing method then where-part is same as method's
            new_whr = method.where_lrn_mech
            self.n_vars = len(new_whr.vars)
            arg_map = np.arange(self.n_vars)
            self.child_arg_maps = [arg_map]
            # print("INIT", method, ":", self.n_vars, [m.id if m else m for m in method.skill_apps[0].match])
            # print("??'", self.child_arg_maps)
            self.cum_arg_ids = self.method_arg_ids[meth_ind]
        else:

            # Update where by antiunifying with where of method. 
            meth_whr = method.where_lrn_mech
            # print("ANTI", method, ":", len(meth_whr.vars), [m.id if m else m for m in method.skill_apps[0].match])
            # print("B'", len(meth_whr.vars))
            # print(meth_whr.conds)
            new_whr, remap_info = self.where_lrn_mech.antiunify(
                    meth_whr, return_remap=True)
            # print(new_whr.conds)
            _,_,remap,_,_ = remap_info
            # print("B'", remap, len(meth_whr.vars))
            remap = remap[:self.n_vars]

            # Note: This is kind of hacky 
            # Drop any vars which don't intersect with cum_arg_ids
            meth_arg_ids = self.method_arg_ids[meth_ind]
            cum_arg_ids = []
            to_remove = []
            for i, ind in enumerate(remap):
                if(ind >= 0 and ind < len(meth_arg_ids)):
                    intr = self.cum_arg_ids[i].intersection(meth_arg_ids[ind])
                    union = self.cum_arg_ids[i].union(meth_arg_ids[ind])
                    # print(self.cum_arg_ids[i], meth_arg_ids[ind], ":", intr)
                    if(len(intr) == 0):
                        remap[i] = -1
                        to_remove.append(i)
                        self.where_lrn_mech.conds.remove(i)
                    else:
                        cum_arg_ids.append(union)
                else:
                    remap[i] = -1

            if(len(to_remove) > 0):
                # print("to_remove", to_remove)
                # print("remap", remap)
                # whr = self.where_lrn_mech
                # whr.vars = [v for i,v in enumerate(whr.vars) if i not in to_remove]
                # whr.conds = whr.conds.remove(to_remove)
                new_whr, remap_info = self.where_lrn_mech.antiunify(
                    meth_whr, return_remap=True, fixed_inds=remap)
                remap = remap[:self.n_vars]


            # if()
            # print("??", self, "->", method)
            # print("B", remap, len(meth_whr.vars))
            # remap = remap[:self.n_vars]
            # remap[(remap >= len(meth_whr.vars))] = -1
            
            # print("R", remap)
            arg_map = invert_remap(remap, len(meth_whr.vars))
            # print("AR", arg_map, meth_ind, len(self.child_arg_maps))

            # Ensure a mapping of the macro's conds to the method's conds
            if(meth_ind == len(self.child_arg_maps)):
                self.child_arg_maps.append(arg_map)
            
            # Update the arg_map for all methods by dropping any 
            #  variables dropped from antiunifying.
            md = {-1:-1}
            c = 0 
            for i in range(self.n_vars):
                if(remap[i] != -1):
                    md[i] = c
                    c += 1
                else:
                    md[i] = -1

            for i in range(len(self.child_arg_maps)):
                if(len(self.child_arg_maps[i]) == 0):
                    continue
                # print(self.child_arg_maps[i])
                self.child_arg_maps[i] = np.vectorize(md.__getitem__)(self.child_arg_maps[i])

            
                # mask = (arg_map != -1) & (arg_map < self.n_vars)
                # self.child_arg_maps[i] = self.child_arg_maps[i][mask]
            # self.child_arg_maps.append(remap[remap!=-1])
            self.n_vars = len(new_whr.vars)
        

        # print("$CAM", self.child_arg_maps)
        self.where_lrn_mech = new_whr
        method.macro = self
        
        

    def _update_skill_apps_from_method(self, method, meth_ind):
        # Fill macro's SkillApps as the method SkillApps
        #  restated w.r.t the macro's generalization.

        meth_whr = method.where_lrn_mech
        arg_map = self.child_arg_maps[meth_ind]
        # print("$$", self, meth_ind, arg_map)
        # print(":::", remap, self.n_vars)
        for sa in method.skill_apps:
            # print("POOP", self, sa, arg_map, self.n_vars)

            match = [None]*self.n_vars
            for j,i in enumerate(arg_map):
                if(i != -1):
                    match[i] = sa.match[j]#[sa.match[j] for  if -1 < j < len(sa.match)]
            # match = [sa.match[j] for j in arg_map if -1 < j < len(sa.match)]
            
            # print("MPO", [m.id if m else m for m in match ], self.n_vars)

            new_sa = SkillApp(self, match, 
                state=sa.state, prob_uid=sa.prob_uid)
            sa.parent = new_sa
            self.skill_apps.append(new_sa)

    
    # def _update_method_params(self):
    #     # print("REMAPS!", remap_info)

    #     if(len(self.methods) == 0):
    #         return

    #     macro_vars = self.where_lrn_mech.vars
    #     macro_var_ptrs = np.array([x.get_ptr() for x in macro_vars], dtype=np.int64)

    #     self.method_params = [None]*len(self.methods)
    #     self_conds = self.where_lrn_mech.conds

    #     # For each method fill in a set of param variables.
    #     for ind, method in enumerate(self.methods):
    #         arg_map = self.child_arg_maps[ind]                        
    #         method_vars = method.where_lrn_mech.vars
    #         params = [None]*len(method_vars)

    #         # print("##", ind, method, remap)
    #         # print("macvars", macro_vars)
    #         # print("methVars", method_vars)

    #         # Try to fill in param with direct mappings of 
    #         #   variables between the macro and method.
    #         for i in range(len(macro_vars)):
    #             meth_ind = arg_map[i]
    #             if(meth_ind != -1 and 
    #                 meth_ind < len(method_vars)):
    #                 params[meth_ind] = macro_vars[i]
            
    #         # For any remaining unmapped vars look for a literal in
    #         #   the method conditions that can map the variable by 
    #         #   dereferencing from an attribute of an already mapped var.
    #         for i, me_v in enumerate(method_vars): 
    #             if(params[i] is None):
    #                 params[i] = find_adj_in_conds(
    #                     method.where_lrn_mech.conds,
    #                     me_v.get_ptr(), 
    #                     macro_var_ptrs
    #                 )
    #         # print([str(x) for x in macro_vars],"->", [str(x) for x in params])
    #         # print([m.id for m in list(method.skill_apps)[0].match])

    #         self.method_params[ind] = params

    def _update(self, meth_ind=None, is_init=False):

        # Note: part of a slight hack
        self.method_arg_ids = []
        for method in self.methods:
            ids = []
            for i in range(method.n_vars):
                s = set()
                for sa in method.skill_apps:
                    # print([m.id for m in sa.match], len(method))
                    m = sa.match[i]
                    if m:
                        s.add(m.id)
                ids.append(s)
            self.method_arg_ids.append(ids)
        # print("method_arg_ids", self, self.method_arg_ids)
             # = [[[sa.match[i].id if sa.match[i] else None for sa in method.skill_apps] ] ]

        # TODO, need to be able to clear just skill_apps
        #  associated with the changed method
        if(MAKE_WHERES):
            if(meth_ind is not None):
                method = self.methods[meth_ind]
                self._update_where_from_method(method, meth_ind, is_init=is_init)
                # self._update_skill_apps_from_method(method, meth_ind)    
            else:
                # print("N METHOD", len(self.methods))
                for ind, method in enumerate(self.methods):
                    # print("ind", ind)
                    self._update_where_from_method(method, ind, is_init=(is_init and ind==0))

            self.skill_apps = []    
            for ind, method in enumerate(self.methods):
                self._update_skill_apps_from_method(method, ind)

            self._update_child_params()

    def add_method(self, method, do_update=True):
        already_in = any([x is method for x in self.methods])
        remap_info = None
        if(not already_in):
            meth_ind = len(self.methods)
            self.methods.append(method)
            if(do_update):
                # with PrintElapse("Update"):
                self._update(is_init=True)#(meth_ind, is_init=meth_ind==0)

        # if(str(self) == "A"):
        #     print()
        #     print("THIS ONE!", method, already_in)
        #     print([len(x) for x in self.method_params])
        #     print([[str(y) for y in x] for x in self.method_params])
        #     print([len(sa.match) for sa in self.skill_apps])

    def replace_method(self, method, new_method):
        for i in range(len(self.methods)):
            if(self.methods[i] is method):
                self.methods[i] = new_method
                new_method.macro = self
                self._update() # Replacements require full update
                return

        raise ValueError(f"MethodSkill {method} not present in MacroSkill {self}.")


    def skill_eq(self, other):
        if(isinstance(other, SkillApp)):
            other = other.skill
        if(not isinstance(other, MacroSkill)):
            return False

        return self._id == other._id

    def overlap(self, other):
        # if(isinstance(other, SkillApp)):
        skill_apps = getattr(other, 'skill_apps', [other])
        # print("->", self.skill_apps, skill_apps)
        m = 0.0
        for sa in skill_apps:
            # print("->", sa)
            v = max([x.overlap(sa) for x in self.skill_apps])
            if(v > m):
                m = v
        return m

    def get_apps(self, state, match=None, _depth=0):
        macro_apps = []
        for i, method in enumerate(self.methods):
            # Get the SkillApps for each method
            arg_mapping = self.child_arg_maps[i]
            if(match is not None):
                _match = [match[ind] if ind >= 0 else None for ind in arg_mapping]
                method_apps = method.get_apps(state, _match, _depth=_depth+1)
            else:
                method_apps = method.get_apps(state, _depth=_depth+1)

            # Build a match and App for the Macro from each method match
            for method_app in method_apps:
                method_match = method_app.match
                macro_match = [None]*self.n_vars
                for j, ind in enumerate(arg_mapping):
                    macro_match[ind] = method_match[j]
                macro_app = SkillApp(self, macro_match, state=state, depth=_depth)
                macro_app.child_apps = [method_app]
                macro_apps.append(macro_app)
        return macro_apps

from numba import njit
from cre.conditions import conds_get_distr_dnf, VarType, cast, i8
from cre.utils import PrintElapse

@njit(cache=True)
def find_adj_in_conds(conds, target_ptr, src_ptrs):
    distr_dnf = conds_get_distr_dnf(conds)
    for k, distr_conjunct in enumerate(distr_dnf):
        for j, var_conjuct in enumerate(distr_conjunct):
            # print("----")
            for i, lit in enumerate(var_conjuct):
                if(not lit.is_alpha):
                    va = cast(lit.op.root_arg_infos[0].ptr, VarType)
                    vb = cast(lit.op.root_arg_infos[1].ptr, VarType)
                    if(cast(va,i8) == target_ptr):
                        return vb
                        # print("HAS TARGET", va, vb)
                    elif(cast(vb,i8) == target_ptr):
                        # print("HAS TARGET", vb, va)
                        return va
                        
                    else:
                        continue

# -----------------------
# : Method Skills
def _first_not_none(lst):
    for x in lst:
        if(x is not None):
            return x

class MethodSkill(CompoundSkillBase):
    def __new__(cls, items, *args, **kwargs):
        # Copy or instantiate
        if(isinstance(items, MethodSkill)):
            self = items.__copy__()
        else:
            self = super().__new__(cls)
            
        return self

    def __init__(self, items, macro=None, unordered=None, optionals=None, fix_bad_opts=False):
        super().__init__(_id=str(id(self)))

        self.macro = macro
        self.unordered = False if not unordered else unordered
        self.fix_bad_opts = fix_bad_opts

        # Handle optionals
        if(isinstance(optionals, np.ndarray) and optionals.dtype == np.bool_):
            self.optional_mask = optionals
        else:
            self.optional_mask = np.zeros(len(items), dtype=np.bool_)
            if(optionals is not None):
                self.optional_mask[optionals] = 1

        if(not isinstance(items, MethodSkill)):
            # Fill in items
            self.items = []
            for i, item in enumerate(items):
                if(isinstance(item, SkillApp)):
                    self.items.append(item.skill)
                elif(isinstance(item, SkillBase)):
                    self.items.append(item)
                elif(isinstance(item, (str))):
                    self.items.append(MacroSkill(item))
        # print("NEW", [list(x.skill_apps)[0] for x in self.items])
        self.item_skill_apps_by_uid = {}
        self._update()
        # print("NEW", [list(x.skill_apps)[0] for x in self.items])

    @property
    def child_skills(self):
        return self.items

    # def _update_params(self):
    #     # Called externally by MacroSkill
    #     macro_vars = self.macro.where_lrn_mech.vars
    #     macro_var_ptrs = np.array([x.get_ptr() for x in macro_vars], dtype=np.int64)
    #     self_vars = self.where_lrn_mech.vars
    #     # print(macro_vars, self_vars)
    #     self.params = [None]*len(self_vars)
    #     for i, sv in enumerate(self_vars): 
    #         for mv in macro_vars:
    #             if(mv.get_ptr() == sv.get_ptr()):
    #                 self.params[i] = mv
    #                 break
        
    #     for i, sv in enumerate(self_vars): 
    #         if(self.params[i] is None):
    #             # print(self.where_lrn_mech.conds)
    #             self.params[i] = find_adj_in_conds(
    #                 self.where_lrn_mech.conds,
    #                 sv.get_ptr(), 
    #                 macro_var_ptrs
    #             )
                
                # print("params", self.params)
                # raise ValueError("Have not implemented case where")
        # print([str(x) for x in self.params])
    # def _update_skill_apps(self):

    def _group_child_skill_apps(self):
        # Group child skill apps by the state uid of the 
        #  first skill app in the sequence 
        skill_apps_by_uid = {}
        for i, item in enumerate(self.items):
            for sa in item.skill_apps:
                if(sa.prob_uid not in skill_apps_by_uid):
                    skill_apps_by_uid[sa.prob_uid] = [None]*len(self.items)
                skill_apps_by_uid[sa.prob_uid][i] = sa
        return skill_apps_by_uid

    def _update(self):
        self.child_arg_maps = [np.full(item.n_vars, -1, dtype=np.int16) for item in self.items]
        self.id_ind_maps_by_uid = {}
        self.n_vars = 0
        skill_apps_by_uid = self._group_child_skill_apps()
        if(MAKE_WHERES):
            self._update_child_mapping()#(skill_apps_by_uid)
            self._update_skill_apps(skill_apps_by_uid)
            self._update_where()
            self._update_child_params()

    def _update_child_mapping(self):
        # Determine a mapping from variables in this method
        #  to variables in each item in the method sequence. 
            
        # Go through each item. Cover optionals last. 
        order = np.argsort(self.optional_mask)
        required_probs = None
        # print("ORDER", order)
        for i in order:
            item = self.items[i]
            arg_map = self.child_arg_maps[i]
            # print("i",i)
            # print(self, ":", item, arg_map)
            # print(self, [[m.id if m else m for m in sa.match] for sa in item.skill_apps])
            is_opt = self.optional_mask[i]
            covered_probs = {sa.prob_uid for sa in item.skill_apps}

            if(required_probs is None and not is_opt):
                required_probs = copy(covered_probs)
            elif(not is_opt and len(arg_map) > 0):                
                # print(required_probs, covered_probs)
                diff = required_probs.difference(covered_probs)
                if(len(diff) > 0):
                    if(self.fix_bad_opts):
                        is_opt = self.optional_mask[i] = True
                    else:
                        raise RuntimeError(
                            f"When build a skill app for method {self} the item {item} was " +
                            f"found to be missing a skill app for problems uid={diff}. But {item} " +
                            f"was not marked as optional."
                        )

            for sa in item.skill_apps:
                # print("UPD", self, ":", item, [m.id if m else m for m in sa.match])
                id_ind_map = self.id_ind_maps_by_uid.get(sa.prob_uid, {})
                # covered_probs.add(sa.prob_uid)
                # print("L", [m.id if m else m for m in sa.match], len(sa.match))
                # if(isinstance(item, MacroSkill)):
                #     for method in item.methods:
                #         print(method)
                #         print([[m.id if m else m for m in sa.match] for sa in method.skill_apps])
                    # print(item.methods)
                # print(sa.prob_uid[:5], ">>", item, arg_map, [m.id for m in sa.match], item.n_vars, id_ind_map, arg_map)
                for j, fact in enumerate(sa.match):
                    if(fact is None):
                        continue
                    _id = fact.id

                    # Use arg_map to fill id_ind_map
                    if(arg_map[j] != -1):
                        id_ind_map[_id] = arg_map[j]
                    else:
                        # Use id_ind_map to fill arg_map
                        if(_id in id_ind_map):
                            arg_map[j] = id_ind_map[_id]

                        # Add a new variable, but only for 
                        #  non-optional items
                        elif(not is_opt):
                            id_ind_map[_id] = self.n_vars
                            arg_map[j] = self.n_vars
                            self.n_vars += 1

                self.id_ind_maps_by_uid[sa.prob_uid] = id_ind_map

            # print("item", i, item, [m.id if m is not None else None for m in sa.match], [sa.prob_uid for sa in item.skill_apps])
            # Assertion for debugging purposes 
            

            
                    
        # print("**CAM", self, self.child_arg_maps)

    # def _update_child_params(self):



    # def _update_child_mapping(self, skill_apps_by_uid):
    #     # Determine the variables for this method and record in 
    #     #  child_arg_maps how they are mapped into each item.
        
    #     for itr, (uid, item_skill_apps) in enumerate(skill_apps_by_uid.items()):
    #         # If child_arg_maps has been filled at all use 
    #         #  those assignments to fill id_ind_map.
    #         id_ind_map = {}
    #         if(itr != 0):
    #             for i, sa in enumerate(item_skill_apps):
    #                 if(sa is None):
    #                     continue
    #                 arg_map = self.child_arg_maps[i]
    #                 print(arg_map)
    #                 print("<<$", i, sa, [m.id for m in sa.match], sa.skill.n_vars)
    #                 for j, arg in enumerate(sa.match):
    #                     if(arg_map[j] != -1):
    #                         id_ind_map[_id] = arg_map[j]


    #         # Fill any yet unseen variable usages into id_ind_map
    #         #  Add new variables as needed.
    #         for i, sa in enumerate(item_skill_apps):
    #             if(sa is None):
    #                 continue
    #             arg_map = self.child_arg_maps[i]
    #             print()
    #             print(arg_map)
    #             print("<<<", i, sa, [m.id for m in sa.match], sa.skill.n_vars)
    #             for j, arg in enumerate(sa.match):                    
    #                 _id = arg.id
    #                 if(_id in id_ind_map):
    #                     arg_map[j] = id_ind_map[_id]
    #                 else:
    #                     if(arg_map[j] == -1):
    #                         id_ind_map[_id] = self.n_vars
    #                         arg_map[j] = self.n_vars
    #                         self.n_vars += 1
    #                     else:
    #                         id_ind_map[_id] = arg_map[j]
    #                     # first_instances.append(entry_wm.get_fact(id=a))
    #                     # if(arg_map[j] == -1):
    #             print(arg_map)
    #             print()
    #         print("{{}}",id_ind_map)        
    #         print("**", self.child_arg_maps)
    #         self.id_ind_maps_by_uid[uid] = id_ind_map

    def _update_skill_apps(self, skill_apps_by_uid):
        # print("START UPDATE")
        self.skill_apps = []
        for itr, (uid, item_skill_apps) in enumerate(skill_apps_by_uid.items()):
            # print("?", [[m.id if m else m for m in sa.match] if sa else sa for sa in item_skill_apps], uid[:5])
            id_ind_map = self.id_ind_maps_by_uid[uid]
            # print(id_ind_map)
            entry_state = _first_not_none(item_skill_apps).state
            entry_wm = entry_state.get("working_memory")
            match = [None]*self.n_vars
            for _id, ind in id_ind_map.items():
                # print(_id, ind, entry_wm is not None)
                match[ind] = entry_wm.get_fact(id=_id)
            # print("!", self, self.n_vars, [m.id if m else m for m in match], _first_not_none(item_skill_apps).prob_uid[:5])
            sa = SkillApp(self, match, state=entry_state, prob_uid=uid)
            self.skill_apps.append(sa)

    def _update_where(self):
        # for sa in self.skill_apps:
            # if(None in sa.match):
            # print(self.n_vars)
            # print(sa, [m.id if m is not None else None for m in sa.match])
        for sa in self.skill_apps:
            # print("BBEP", sa, [m.id if m else None for m in sa.match])

            # if(None in sa.match):
            # #     print(self.n_vars)
            #     print(sa, [m.id if m is not None else None for m in sa.match])
            #     raise ValueError()
            var_names = [f"A{i}" for i in range(len(sa.match))]
            self.where_lrn_mech.ifit(sa.state, sa.match, var_names=var_names)

    # def _init_skill_apps(self):

    #     # Organize skill_apps by the problem they occured in
    #     # print("\nitems", self.items)
        

    #     # List of list of arg inds
    #     # self.item_params = [[None]*item.n_vars for item in self.items]
    #     # print(self.item_params)
    #     # self.item_params = [[None]*len(item.where_lrn_mech.conds.vars) for item in self.items]
    #     # print([item.n_vars for item in self.items])


    #     var_fact_ids = []
    #     n_params = 0
        
                    
    #         # print(uid[:5], arg_id_map)

    #     # print(self.item_params)
    #     for uid, item_skill_apps in skill_apps_by_uid.items():
    #         # The working memory for the state at the point 
    #         #  when this method's first primatives were applied
    #         entry_state = _first_not_none(item_skill_apps).state
    #         entry_wm = entry_state.get("working_memory")
    #         match = [entry_wm.get_fact(id=a) for a in var_fact_ids]
    #         # print("MMATCH", uid[:5], [m.id for m in match])
    #         var_names = [f"A{i}" for i in range(len(match))]
    #         self.where_lrn_mech.ifit(entry_state, match, var_names=var_names)
    #         self.skill_apps.add(SkillApp(self, match, state=entry_state, prob_uid=uid))


    #     # print("---------!")
    #     # print(var_fact_ids,len(var_fact_ids), n_params)
    #     # print(self.item_params)
    #     # print("---------!")
    #     # _vars = [None]*n_params
    #     # c = 0
    #     # for i, p in enumerate(self.item_params):
    #     #     for v_ind, p_ind in enufmerate(p):
    #     #         if(_vars[p_ind] is None):
    #     #             wlm = self.items[i].where_lrn_mech
    #     #             _vars[p_ind] = wlm.vars[v_ind].with_alias(f"A{c}")
    #     #             c += 1
    #     # print(_vars)
    #     # if(len(self.skill_apps) > 1):
    #     #     print("THIS!", self.macro)
    #     #     print(self.where_lrn_mech.conds)


    def is_optional(self, ind):
        return self.optional_mask[ind]

    @property
    def is_recursive(self):
        return self.macro in self.items

    def set_optional(self, ind, is_optional=1):
        self.optional_mask[ind] = is_optional

    def __copy__(self):
        new_rhs = MethodSkill([*self.items],
                    macro=self.macro,
                    unordered=self.unordered,
                    optionals=self.optional_mask.copy()
                    )
        return new_rhs


    def __str__(self, bracket=True):
        item_strs = []
        for i, skill in enumerate(self.items):
            name = skill._id
            if(self.optional_mask[i]):
                item_strs.append(f"{name}*")
            else:
                item_strs.append(str(name))

        rhs_str = " ".join(item_strs)

        # underline if unordered
        if(self.unordered):
            rhs_str = f"\033[4m{rhs_str}\033[0m"                         

        if(bracket):
            rhs_str = f"[{rhs_str}]"
        return rhs_str

    __repr__ = __str__

    # RHS instances are only equal to themselves
    def __eq__(self, other):
        return self is other

    def __hash__(self):
        return id(self)


    def __len__(self):
        return len(self.items)

    def get_apps(self, state, match=None, _depth=0):
        method_apps = []

        matches = self.get_matches(state, match)
        if(len(matches) == 0):
            raise ValueError(f"{self}: NO MATCH!")

        for score, match in matches:
            print(f"S={score}, M={[m.id for m in match]}")
            skill_app = SkillApp(self, match, state=state,
                 match_score=score, depth=_depth)

            skill_app.get_child_apps()
            method_apps.append(skill_app)
        return method_apps
        

        # macro_apps = []
        # for i, method in enumerate(self.methods):
        #     arg_mapping = self.child_arg_maps[i]
        #     method_apps = method.get_apps(state, match, depth=_depth)
        #     for method_app in method_apps:
        #         method_match = method_app.match
        #         macro_match = [None]*len(self.n_vars)
        #         for j, ind in enumerate(arg_mapping):
        #             macro_match[ind] = method_match[j]
        #         macro_app = SkillApp(self, macro_match, state=state)
        #         macro_apps.append(macro_app)

    # def __getitem__(self, key):
    #     # Implement RHS Slicing
    #     if(isinstance(key, slice)):
            
    #         # print("SLICE", key.__dict__ )
    #         subitems = self.items[key]
    #         subopt = self.optional_mask[key]
    #         new_rhs = MethodSkill(subitems)
    #         # new_rhs.items = subitems
    #         new_rhs.optional_mask = subopt
    #         new_rhs.unordered = self.unordered
    #         new_rhs.skill = self.skill

    #         # s,e = key.start, key.stop
    #         # s = 0 if s is None else s
    #         # e = len(self.items)-1 if e is None else e
    #         # new_rhs.rel_lits = [(r, o0-s,o1-s,a0,a1) for r, o0,o1,a0,a1 in self.rel_lits 
    #         #                     if (o0 >= s and o1 >= s and o0 < e and o1 < e)]
    #         # print(s, e,)# [(o0 >= s, o1 >= s, o0 < e, o1 < e) for o0,o1,a0,a1 in self.eq_toks ])
    #         # print("NEW")                                 
    #         # print(self, self.rel_lits)
    #         # print("SLICE", new_rhs.eq_toks)


    #         return new_rhs
    #     raise NotImplemented

    # def insert(self, i, val, optional=False):
    #     self.items.insert(i, val)
    #     self.optional_mask = np.insert(self.optional_mask, i, bool(optional))
    #     # self.rel_lits += [(r, o0+1 if o0 >= i else o0, o1+1 if o1 >= i else o1, a0, a1)
    #     #                  for r,o0,o1,a0,a1 in self.rel_lits]

    # def __add__(self, other):
    #     s = len(self)

    #     new_rhs = copy(self)
    #     new_rhs.items += other.items

    #     new_rhs.optional_mask = np.zeros(len(new_rhs.items),dtype=np.bool_)
    #     optionals = self.optional_mask.nonzero()[0]
    #     optionals += s + other.optional_mask.nonzero()[0]
    #     new_rhs.optional_mask[optionals] = True

    #     # new_rhs.rel_lits = self.rel_lits
    #     # new_rhs.rel_lits += [(r, o0+s,o1+s,a0,a1) for r,o0,o1,a0,a1 in other.rel_lits]


    #     return new_rhs


# ------------------------------------------------
# : Sym

# class Sym:
#     def __new__(cls, name, prods=[]):
#         if(isinstance(name, Sym)):
#             return name

#         self = super().__new__(cls)
#         global symbol_index
#         self.name = name
        
#         self.prods = prods
#         return self

#     def add_RHS(self, items, unordered=False):
#         self.prods.append(RHS(items, symbol=self, unordered=unordered))

#     def __eq__(self, other):
#         if(isinstance(other, str)):
#             return self.name == other
#         elif(hasattr(other, 'name')):
#             return self.name == other.name
#         else:
#             return False

#     def __hash__(self):
#         return hash(self.name)

#     def __str__(self):
#         return self.name

#     def __repr__(self):
#         return self.name



# def _get_method_matches(method):
#     matches = method.get_matches(state, match)
#     if(len(matches) == 0):
#         raise ValueError(f"{method}: NO MATCH!")

#     for score, match in matches:
#         print(f"S={score}, M={[m.id for m in match]}")
#         skill_app = SkillApp(method, match, state=state,
#              match_score=score, depth=_depth)

#         skill_app.get_child_apps()
#         method_apps.append(skill_app)

class MatchTree():
    def __init__(self, skill, matches, child_trees=None, depth=0):
        self.skill = skill
        self.matches = matches
        self.child_trees = child_trees
        self.depth = depth
        self.parent = None
        self.parent_match_ind = None

        self.cum_scores = np.zeros(len(matches))
        self.scores = np.zeros(len(matches))
        self.skill_apps = []
        for i, (score, match) in enumerate(matches):
            skill_app = SkillApp(skill, match)
            self.skill_apps.append(skill_app)

            cum_score = score
            self.scores[i] = score
            if(self.child_trees and i < len(self.child_trees)):
                cts = self.child_trees[i]
                for c in cts: 
                    c.parent = self
                    c.parent_match_ind = i
                    for c_skill_app in c.skill_apps:
                        c_skill_app.parent = skill_app

                cum_score *= np.mean([c.max_score for c in cts])
            self.cum_scores[i] = cum_score
        if len(matches) > 0:
            self.max_score = np.max(self.scores)
            self.max_cum_score = np.max(self.cum_scores)
        else:
            self.max_score = 0.0
            self.max_cum_score = 0.0

    # def as_skill_apps(self)

    def __str__(self):
        s = ""
        for i, (score, match) in enumerate(self.matches):
            m_str = ','.join([m.id if m else "None" for m in match]) if match else None
            sk_str = str(self.skill)
            if(i != 0):
                sk_str = " "*len(sk_str)
            s += f"{self.depth*' '}{score:.2f} {sk_str}({m_str})\n"
            if(self.child_trees and i < len(self.child_trees)):
                cts = self.child_trees[i]
                for child_tree in cts:
                    s += str(child_tree)
        return s

    def __repr__(self):
        return f"MatchTree({self.skill}, n={len(self.matches)}, d={self.depth})" 

    def _ensure_trees_by_depth(self):
        if(hasattr(self, '_trees_by_depth')):
            return 
        _trees_by_depth = {}

        arr = _trees_by_depth.get(self.depth, [])
        arr.append(self)
        _trees_by_depth[self.depth] = arr

        if(self.child_trees is not None):
            for cts in self.child_trees:
                for child_tree in cts:
                    child_tree._ensure_trees_by_depth()
                    for depth, _arr in child_tree._trees_by_depth.items():                        
                        arr = _trees_by_depth.get(depth, [])
                        _trees_by_depth[depth] = arr+_arr

        # print(list(_trees_by_depth.keys()))
        self._max_depth = max(_trees_by_depth.keys())
        self._trees_by_depth = _trees_by_depth

    def get_depth_trees(self, depth):
        self._ensure_trees_by_depth()
        return self._trees_by_depth[depth]

    @property
    def max_depth(self):
        self._ensure_trees_by_depth()
        return self._max_depth

    @property
    def trees_by_depth(self):
        self._ensure_trees_by_depth()
        return self._trees_by_depth

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self is other

    def overlap(self, skill_app):
        max_score = 0.0
        best_m_skill_app = None
        for s0 in self.skill_apps:
            if(hasattr(s0,'overlap')):
                score = s0.overlap(skill_app)
            else:
                score = s0 == skill_app

            if(score > max_score):
                max_score = score
                best_m_skill_app = s0
        return max_score, best_m_skill_app

    def find_exact_match(self, skill_app):
        for depth, trees in self.trees_by_depth.items():
            for tree in trees:
                for sa in tree.skill_apps:
                    if(sa.overlap(skill_app) == 1.0):
                        return tree 
        return None


def _macro_match_from_child_match(macro, arg_mapping, method_match):
    if(method_match is None):
        return None
    macro_match = [None]*macro.n_vars
    for j, ind in enumerate(arg_mapping):
        if(ind >= 0):
            macro_match[ind] = method_match[j]
    return macro_match

def _child_match_from_params(skill, child_ind, match):
    if(match is None):
        return None
    params = skill.child_params[child_ind]
    mapping =  skill.child_arg_maps[child_ind]
    child_match = []

    # print(skill, child_ind, [f"({ind}, {str(v)})" for ind, v in params])
    for i, (ind,v) in enumerate(params):
        if(v is not None and ind < len(match) and ind >= 0):
            arg = match[ind]
            child_match.append(v(arg))
        else:
            child_match.append(None)
    # print(skill, child_ind, [m.id if m else m for m in child_match])
    # print()
    return child_match

def _get_child_skill_matches(skill, i, child_skill, state, match,
        allow_partial=False, cheat=True):

    # If match is given then try to resolve the child method's match 
        #  by using param relationships 
    # print("PARENT MATCH", [m.id if m else m for m in match] if match else None)
    child_match = _child_match_from_params(skill, i, match)
    if(child_match is not None):
        # If  all items in are specified then use it as is        
        if(None not in child_match):
            # print("PARMS RESOLVED", skill, i)#[m.id for m in skill_match])
            return [(1.0, child_match)]

        
        # else:
    # If not all items in match specified then use get_matches
    # print(skill, "_>", [m.id if m else m for m in child_match] if child_match else None)
    # print("allow_partial",  allow_partial)
    child_matches = child_skill.get_matches(state, child_match, allow_partial=allow_partial)

    # Cheat match by filling in missing 
    if(cheat and len(child_matches) == 0 and child_match is not None):
        wm = state.get("working_memory")
        for c_sa in child_skill.skill_apps:
            all_agree = True
            for f_old, f_new in zip(c_sa.match, child_match):
                if(f_new is not None):
                    if(f_old.id != f_new.id):
                        all_agree = False
            if(all_agree):
                for i, (f_old, f_new) in enumerate(zip(c_sa.match, child_match)):
                    child_match[i] = wm.get_fact(id=f_old.id)


    # print([m.id if m else m for m in child_skill.skill_apps[0].match])
    # print(skill, "=>", [m.id if m else m for m in child_matches[0][1]] if len(child_matches) else None)
        # skill_matches = skill.get_matches(state, allow_partial=allow_partial)
    if(len(child_matches) == 0 and child_match is not None):
        score = sum([m is not None for m in child_match])/len(child_match)
        return [(score, child_match)]        
    return child_matches

# def rollout_method_app_tree(method, state, match)

def build_macro_match_tree(macro, state, match=None, depth=0):
    # macro_matches = {}
    method_tree_infos = []
    for i, method in enumerate(macro.methods):
        
        # Try to find matches for each method in the macro
        method_matches = _get_child_skill_matches(macro, i, method,
            state, match, allow_partial=True)

        # print("ENTER", i, ":", method, len(method_matches), [m.id if m else None for m in match] if match else None)

        # If any full matches can be found then use them to
        #  make item matches, but otherwise use None as a placeholder,
        #  to force sub-skills to handle their own matching.
        if(len(method_matches) == 0):
            method_matches = [(0.0, None)]

        method_tree_parts = []
        for score, method_match in method_matches:
            item_tree_set = []
            for j, item in enumerate(method.items):
                # Base case for primative skills                     
                if(isinstance(item, PrimSkill)):

                    item_matches = _get_child_skill_matches(method, j, item,
                        state, method_match, allow_partial=True)
                    # print("ITEM MATCHES", item, [m.id if m else None for m in item_matches[0][1]] if len(item_matches) > 0 else None, len(item_matches))
                    # print("!!", item, len(item_matches))
                    mt = MatchTree(item, item_matches, None, depth=depth+2)    

                # Otherwise recurse into new macro
                else:
                    # print("m--", method, [m.id if m else None for m in method_match])
                    item_match = _child_match_from_params(method, j, method_match)
                    # print("THIS CASE", [m.id if m else None for m in item_match])
                    mt = build_macro_match_tree(item, state, item_match, depth=depth+2)
                item_tree_set.append(mt)
            # item_trees.append(item_tree_set)

            # If we used None as a placeholder then recover the
            #  matches for the calling method from the item match trees
            if(method_match is None):
                # TODO
                method_tree_parts.append(((0.0, None), item_tree_set))
            else:
                method_tree_parts.append(((score, method_match), item_tree_set))
            
        # method_tree = MatchTree(method, *zip(*method_tree_parts), depth=_depth+1)
        # print("L", method_matches, method_tree_parts)
        method_tree_infos.append((method, method_tree_parts))
    
    # Organize the method applications by the macro match they
    #  would produce
    tree_parts = {}
    for i, (method, method_tree_parts) in enumerate(method_tree_infos):
        # print(method, len(method_tree_parts))
        arg_mapping = macro.child_arg_maps[i]
        # continue
        for (score, method_match), item_tree_set in method_tree_parts:
            macro_match = _macro_match_from_child_match(
                macro, arg_mapping, method_match)
        
            tup = tuple([m.id if m else None for m in macro_match]) if macro_match else None
            arr = tree_parts.get(tup,(None, []))[1]
            arr.append((i, method, score, method_match, item_tree_set))
            tree_parts[tup] = (macro_match, arr)

    # Make the match Trees for each method match
    macro_tree_parts = []
    for macro_match, arr in tree_parts.values():
        method_tree_set = []
        from itertools import groupby
        grpd = groupby(arr, lambda x: x[0])
        for i, group in grpd:
            method_tree_parts = []
            for i, method, score, method_match, item_tree_set in group:
                method_tree_parts.append(((score, method_match), item_tree_set))
            # print(method_tree_parts)
            method_tree = MatchTree(method, *zip(*method_tree_parts), depth=depth+1)
            method_tree_set.append(method_tree)
        macro_tree_parts.append(((score, macro_match), method_tree_set))

    # print(macro_tree_parts)
    # print("&&&&&&&-&&&&&&&&&")
    macro_tree = MatchTree(macro, *zip(*macro_tree_parts), depth=depth)
    # print(macro_tree)
    # print("&&&&&&&-&&&&&&&&&")
    

    return macro_tree




# ------------------------------------------------
# : Grammar

class SkillGrammar:
    def __init__(self, macro_skills=None):
        # print(macro_skills)
        if(isinstance(macro_skills, dict)):
            self.macro_skills = macro_skills
        else:
            self.macro_skills = {}
            if(macro_skills is not None):
                for macro_skill in macro_skills:
                    self.add(macro_skill)

        self._assert_has_all_macros()
            
    def _assert_has_all_macros(self):
        # Check that the items 
        for macro_skill in self.macro_skills.values():
            # print(macro_skill)
            for method in macro_skill.methods:
                for item in method.items:
                    if(isinstance(item, MacroSkill) and 
                       item._id not in self.macro_skills):
                        warnings.warn(
                            f"Macro {item} in grammar's tree but not added to grammar.",
                         RuntimeWarning)


    def add(self, ms):
        # if(ms._id in self.macro_skills):
        #     raise ValueError(f"MetaSkill with name: {ms._id} already in SkillGrammar.")
        self.macro_skills[ms._id] = ms

    def replace(self, old_ms, new_ms):
        _id = old_ms if(isinstance(old_ms,str)) else old_ms._id
        self.macro_skills[_id] = new_ms        
    def remove(self, old_ms):
        _id = old_ms if(isinstance(old_ms,str)) else old_ms._id
        del self.macro_skills[_id]

    def __str__(self):
        lines = []
        for name, ms in self.macro_skills.items():
            methods = ms.methods
            if(len(methods) > 0):
                meth_strs = []
                for method in methods:
                    # rhs_str = "".join([sym.name for sym in rhs.items])

                    # # underline if unordered
                    # if(rhs.unordered):
                    #     rhs_str = f"\033[4m{rhs_str}\033[0m"                         

                    meth_strs.append(method.__str__(bracket=False))
                lines.append(f'{name} -> {" | ".join(meth_strs)}')
        return "\n".join(lines)

    def __copy__(self):
        new_macro_skills = {}
        for symb, ms in self.macro_skills.items():
            new_macro_skills[symb] = copy(ms)
        return SkillGrammar(new_macro_skills)

    def _ensure_root_symbols(self):
        if(getattr(self, '_root_symbols', None) is None):
            self._root_symbols = []
            for _id, ms in self.macro_skills.items():
                if(len(self.get_RHSs_with(_id)) == 0):
                    self._root_symbols.append(ms)

    @property
    def root_symbols(self):
        self._ensure_root_symbols()
        return self._root_symbols

    @property
    def methods(self):
        methods = []
        for _id, ms in self.macro_skills.items():
            methods += ms.methods
        return methods

    @property
    def preterminal_RHSs(self):
        if(getattr(self, '_preterminals', None) is None):
            self._preterminal_RHSs = {}
            for sym, ms in self.macro_skills.items():
                for method in ms.methods:
                    all_items_terminal = True
                    for item in method.items:
                        if(len(self.methods.get(item._id,[])) != 0):
                            all_items_terminal = False
                            break
                    if(all_items_terminal):
                        self._preterminal_RHSs[method] = []
        return self._preterminal_RHSs


    def _ensure_rhss_with(self):
        if(getattr(self,'_rhss_with_map', None) is None):
            self._rhss_with_map = {}        
            for _, ms in self.macro_skills.items():
                for method in ms.methods:
                    for item in method.items:
                        lst = self._rhss_with_map.get(item,[])
                        lst.append(method)
                        self._rhss_with_map[item] = lst
        
    def get_RHSs_with(self, symbol):
        ''' Get all RHSs that contain symbol '''
        self._ensure_rhss_with()    
        return self._rhss_with_map.get(symbol,[])

    def _ensure_upstream_rhss(self):
        if(getattr(self,'_upstream_rhs_map', None) is None):
            self._upstream_rhs_map = {}
            syms = self.root_symbols
            # Base case roots have no upstream
            for sym in self.root_symbols:
                self._upstream_rhs_map[sym] = set()
            covered_syms = set() # track just in case of recursion
            while(len(syms) > 0):
                new_syms = []
                for sym in syms:
                    if(sym in covered_syms):
                        continue
                    sym_upstrm = self._upstream_rhs_map[sym]

                    ms = self.macro_skills.get(sym,None)
                    rhss = [] if ms is None else ms.rhss
                    for rhs in rhss:
                        for _sym in rhs.items:
                            _sym_upstrm = self._upstream_rhs_map.get(_sym, set())
                            _sym_upstrm.add(rhs)
                            for x in sym_upstrm:
                                _sym_upstrm.add(x)
                            self._upstream_rhs_map[_sym] = _sym_upstrm
                            new_syms.append(_sym)
                        self._upstream_rhs_map[rhs] = sym_upstrm
                    covered_syms.add(sym)
                syms = new_syms


    def get_upstream_RHSs(self, sym_or_rhs):
        # print(":::", self._upstream_rhs_map)
        self._ensure_upstream_rhss()
        # print(":::", self._upstream_rhs_map)
        return self._upstream_rhs_map[sym_or_rhs]
        # if(isinstance(sym_or_rhs, RHS)):
        #     rhss = [sym_or_rhs]
        # else:
        #     rhss = self.get_RHSs_with(rhs_or_symbol) 
        # return rhss


    def _clear_internals(self):
        self._roots = None
        self._preterminals = None
        self._rhss_with_map = None
        self._upstream_rhs_map = None

    def _simplify(self):
        # print("START SIMPLIFY")
        # print(self)
        lone_macros = []
        for sym, macro in self.macro_skills.items():
            methods = macro.methods

            # TODO: Iterating backwards isn't a gaurentee
            #  that down stream will come before upstream
            for i in range(len(methods)-1, -1, -1):
                method = methods[i]
                # If a method consists of a single non-terminal symbol then
                #  substitute it for all of its downstream methods.
                if(len(method) == 1 and 
                   isinstance(method.items[0], MacroSkill)):
                    lone_macro = method.items[0]
                    lone_macros.append(lone_macro)
                    del methods[i]
                    # _ms = self.macro_skills
                    # for _sym, _ms in self.macro_skills.items():
                    for l_meth in lone_macro.methods:
                        macro.add_method(l_meth)
                        # l_meth.skill = ms
                        # rhss.append(l_rhss)

        self._clear_internals()

        # Remove any symbols that never appear in RHSss
        for ms in lone_macros:
            if(len(self.get_RHSs_with(sym)) == 0):
                del self.macro_skills[ms._id]


class SkillApp(object):
    def __init__(self, skill, match, state=None,

                 # These need to become part of SkillApplication
                 next_state=None,
                 prob_uid=None, 
                 short_name=None, 
                 in_process=False,

                 # Params Related to Top-down Application 
                 depth=None,
                 parent=None, match_score=None,
                 child_apps=None):
        self.skill = skill
        self.match = match
        self.state = state
        self.next_state = next_state
        self.short_name = short_name
        self.state = state
        self.prob_uid = prob_uid
        self.depth = depth
        self.parent = parent
        self.match_score = match_score
        self.child_apps = None
        self.in_process = in_process
        # if(isinstance(skill, SkillBase)):
        #     skill.add_app(self)


    def match_overlap(self, other):
        if(not hasattr(other, 'match')):
            return 0.0

        s_a = set([a.id if a else None for a in self.match])
        s_b = set([a.id if a else None for a in other.match])
        if(len(s_a) == 0 and  len(s_b) == 0):
            return 1.0

        denom = max(len(s_a),len(s_b))
        if(denom == 0):
            return 0.0
        return len(s_a.intersection(s_b)) / denom

    def arg_overlap(self, other):
        if(not hasattr(other, 'match')):
            return 0.0

        s_a = set([a.id if a else None for a in self.match[1:]])
        s_b = set([a.id if a else None for a in other.match[1:]])
        if(len(s_a) == 0 and  len(s_b) == 0):
            return 1.0

        denom = max(len(s_a),len(s_b))
        if(denom == 0):
            return 0.0
        return len(s_a.intersection(s_b)) / denom

    def sel_overlap(self, other):
        # print(self, len(self.match), other, len(other.match))
        if(not self.match or not self.match[0] or len(self.match) == 0 or
           not other.match or not self.match[0] or len(other.match) == 0):
            return 0.0
        return float(self.match[0].id == other.match[0].id)

    def how_overlap(self, other):
        if(isinstance(self.skill, PrimSkill)):
            if(not isinstance(other.skill, PrimSkill)):
                return False
            return float(self.skill.how_part == other.skill.how_part)
        elif(isinstance(self.skill, MacroSkill)):
            if(not isinstance(other.skill, MacroSkill)):
                return False
            return float(self.skill._id == other.skill._id)
        else:
            return float(self.skill == other.skill)

    # In principle this could be replaced with
    #  structure mapping score of where-part
    def overlap(self, other):
        if(not isinstance(other, SkillApp)):
            return 0.0

        # if(self.how_overlap(other) > 0):
        #     print("H", self.how_overlap(other),
        #           "S", self.sel_overlap(other),
        #           "M", self.match_overlap(other),
        #           "|", repr(self), repr(other))
        return (self.how_overlap(other) +
                self.sel_overlap(other) +
                self.arg_overlap(other)) / 3

    def depends_on(self, other):
        return other.match[0] in self.match

    def __eq__(self, other):
        if(not isinstance(other, SkillApp)):
            return False

        return (self.prob_uid == other.prob_uid and 
                self.skill_eq(other) and 
                self.match == other.match)

    def __hash__(self):
        return hash((self.prob_uid, self.skill, *self.match))

    def skill_eq(self, other):
        # print("skill_eq", type(self), type(other))
        s_a = self.skill
        s_b = other
        if(isinstance(other, SkillApp)):
            s_b = s_b.skill

        if(isinstance(s_a, PrimSkill)):
            return (isinstance(s_b, PrimSkill) and
                    s_b.how_part == s_a.how_part)
        elif(isinstance(s_a, MacroSkill)):
            return (isinstance(s_b, MacroSkill) and 
                    s_a is s_b)
        else:
            return False
        # TODO: 

        # h_b = other
        # if(hasattr(other, 'how_part')):
        #     h_b = other.how_part
        # elif(hasattr(other, 'skill')):
        #     h_b = other.skill.how_part

        # return self.skill.how_part == h_b

    def __hash__(self):
        # sel = self.sel if not self.sel or isinstance(self.sel, str) else self.sel.id
        # if(len(self.match) > 0)
        match = [m.id if m is not None else m for m in self.match]  
        return hash((getattr(self.skill,"how_part", None), tuple(match)))

    def __repr__(self):
        match = self.match if len(self.match) > 0 and isinstance(self.match[0], str) else [m.id for m in self.match]
        if(isinstance(self.skill, PrimSkill)):
            return f"{self.skill}({','.join(match[1:])})->{match[0]}"                
        else:
            return f"{self.skill}({','.join(match)})"

    def __str__(self):
        if(self.short_name):
            return self.short_name
        else:
            return str(self.skill)


    @property
    def name(self):
        return str(self)

    def child_match_from_params(self, i):
        params = self.skill.child_params[i]
        mapping =  self.skill.child_arg_maps[i]
        # args = self.match[:len(params)]  
        # print("args", [self.match[ind] for ind in mapping])
        # print([m.id for m in self.args])
        # print([m.id for m in args])
        # print("%",[(str(x)) for x in args])
        # print("%",[str(x) for x in params])
        child_match = []
        for i, (ind,v) in enumerate(zip(mapping,params)):
            if(v is None or ind > len(self.match)):
                child_match.append(None)
            else:
                arg = self.match[ind]
                child_match.append(v(arg))
        return child_match

    def get_child_apps(self):
        if(self.child_apps is not None):
            return self.child_apps

        skill = self.skill
        apps = []

        if(isinstance(skill, CompoundSkillBase)):
            for i, child_skill in enumerate(skill.child_skills):
                child_match = self.child_match_from_params(i)
                if(None in child_match):
                    c_whrln = child_skill.where_lrn_mech
                    child_matches = list(c_whrln.get_partial_matches(self.state, child_match))
                    # print([f":>{(s,[m.id for m in mtch])}\n" for (s,mtch) in child_matches])
                    child_matches = [mtch for (s,mtch) in child_matches]
                    
                else:    
                    child_matches = [child_match]
                # child_args = [v(x) if x is not None else None for x,v in zip(args, params)]

                # assert None not in child_match, "None found in child match"
                # print("c_args", [m.id for m in child_args])
                for child_match in child_matches:
                    apps.append(SkillApp(child_skill, child_match,
                        state=self.state, parent=self, depth=self.depth+1))

        else:
            raise NotImplemeted(f"No implementation of .get_child_app() for skill of type {type(skill)}.")

        self.child_apps = apps
        return apps

    def rollout_child_apps(self):
        skill_apps = [self]
        if(isinstance(self.skill, PrimSkill)):
            return skill_apps
        for child_app in self.get_child_apps():
            child_app.parent = self
            skill_apps += child_app.rollout_child_apps()
        return skill_apps






