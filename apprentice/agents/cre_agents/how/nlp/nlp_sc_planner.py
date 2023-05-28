import numpy as np
from cre.sc_planner import SetChainingPlanner
from cre import Fact, Var, cre_context, CREFunc, cre_context
from itertools import chain

from numba import njit, types
from numba.typed import List, Dict
from numba.types import unicode_type, i8, f8, ListType, DictType, Tuple
from cre.utils import _struct_from_ptr
from cre.sc_planner import ExplanationTreeType
from cre.func import CREFuncType, arg_infos_type, ARGINFO_VAR, ARGINFO_FUNC, ARGINFO_CONST 
from cre.var import var_from_ptr
from cre.utils import cast
from collections import Counter
import types
# from cre.default_funcs import Add, Subtract, Multiply, Divide, CastFloat

from apprentice.agents.cre_agents.how import BaseHow, SetChaining, register_how, ExplanationSet
from apprentice.agents.cre_agents.funcs import (
    Add, Subtract, Multiply, Divide, Half, OnesDigit, TensDigit, Square, Power,
    Double, Increment, Decrement, Log2, Sin, Cos
)

# ------------------------------------
# : Default Functions / Patterns

# Add = Add(f8,f8)
# Subtract = Subtract(f8,f8)
# Multiply = Multiply(f8,f8)
# Divide = Divide(f8,f8)



default_func_dictionary = {
    "sum" : Add,
    "add" : Add,
    "plus" : Add,

    "product" : Multiply,
    "multiply" : Multiply,
    "times" : Multiply,

    "subtract" : Subtract,
    "minus" : Subtract,
    "difference" : Subtract,

    "divide" : Divide,
    "quotient" : Divide,
    "ratio" : Divide,
    "proportion" : Divide,

    "half" : Half,

    "ones" : OnesDigit,

    "tens" : TensDigit,

    "square" : Square,

    "power" : Power,

    "double" : Double,
    "twice" : Double,

    "increment" : Increment,
    "decrement" : Decrement,
    "log2" : Log2,

    "sin" : Sin,
    "sine" : Sin,

    "cos" : Cos,
    "cosine" : Cos,
}

defualt_special_patterns = {
  r"(\S+)\sdivided\sby\s(\S+)" : Divide,
  r"(\S+)\sover\s(\S+)" : Divide,
  r"(\S+)\stimes\s(\S+)" : Multiply,
  r"(\S+)\sminus\s(\S+)" : Subtract,
  r"(\S+)\splus\s(\S+)" : Add,


  # r"\(*\s*1\s*\/\s*2\s*\)*" : Half,

  # r"(\S+)\s*\/\s*(\S+)" : Divide,
  # r"(\S+)\s*\*\s*(\S+)" : Multiply,
  # r"(\S+)\s*-\s*(\S+)" : Subtract,
  # r"(\S+)\s*\+\s*(\S+)" : Add,
  
  r"ones\s(digit|place)" : OnesDigit,
  r"ones'\s(digit|place)" : OnesDigit,
  r"one's\s(digit|place)" : OnesDigit,
  r"last\s(digit|place)" : OnesDigit,
  r"final\s(digit|place)" : OnesDigit,

  r"tens\s(digit|place)" : TensDigit,
  r"tens'\s(digit|place)" : TensDigit,
  r"ten's\s(digit|place)" : TensDigit,
}


# Place holder value for ignoring particualr keyword arguments 
IGNORE = None

@njit(unicode_type(ExplanationTreeType,i8), cache=False)
def tree_str(root,ind=0):
    # print("START STR TREE")
    # if(len(root.children) == 0): return "?"
    s_ind = ' '*ind
    s = ''
    for entry in root.entries:
        # print("child.is_op", child.is_op)
        if(entry.is_func):
            func, child_arg_ptrs = entry.func, entry.child_arg_ptrs
        #     # for i in range(ind): s += " "
                
            s += f"\n{s_ind}{func.origin_data.name}("
        #     # print(child_arg_ptrs)
            for i, ptr in enumerate(child_arg_ptrs):
                
                ch_expl = _struct_from_ptr(ExplanationTreeType, ptr)
                # print(ch_expl)
                # tree_str(ch_expl, ind+1)
        #         # print("str",tree_str(ch_expl, ind+1))
                s += f'{tree_str(ch_expl, ind+1)}'
                if (i != len(child_arg_ptrs) -1): s += ","
                    
                # s += ","
            s += ")"
        else:
            s += "?"
    return s


cf_ai_lst_tup = Tuple((CREFuncType, CREFuncType, arg_infos_type[::1]))
lst_cf_ai_lst_tup = ListType(cf_ai_lst_tup)
policy_dict_typ = ListType(lst_cf_ai_lst_tup)

@njit(policy_dict_typ(CREFuncType),cache=True)
def func_to_arg_info_policy(func):
    ''' Generates a policy representation for a CREFunc which 
        uses each func's arg_infos instead of argument values'''
    policy_dict = Dict.empty(i8, lst_cf_ai_lst_tup)
    stack = List()
    cf = func
    depth_policy = List.empty_list(cf_ai_lst_tup)
    depth_policy.append((cf.origin, cf, cf.root_arg_infos))
    policy_dict[cf.depth] = depth_policy
    i = 0 
    keep_looping = True
    while(keep_looping):        
        if(i < len(cf.root_arg_infos)):
            arg_info = cf.root_arg_infos[i]
            if(arg_info.type == ARGINFO_FUNC):
                stack.append((cf, i+1))
                cf = cast(arg_info.ptr, CREFuncType)
                if(cf.depth not in policy_dict):
                    policy_dict[cf.depth] = List.empty_list(cf_ai_lst_tup)                    
                depth_policy = policy_dict[cf.depth]
                depth_policy.append((cf.origin, cf, cf.root_arg_infos))
                i = 0 
            else:
                i += 1
        while(i >= len(cf.root_arg_infos)):
            if(len(stack) == 0):
                keep_looping = False
                break

            cf, i = stack.pop(-1)

    out = List.empty_list(lst_cf_ai_lst_tup)
    for i in range(func.depth):
        out.append(policy_dict[i+1])

    return out


def func_to_policy(func, args, conv_funcs=[]):
    arg_inf_policy = func_to_arg_info_policy(func)

    base_var_ind_map = {}
    for i, var_ptr in enumerate(func.base_var_ptrs):
        base_var_ind_map[var_ptr] = i

    context = cre_context()
    conv_unq_strs = [f.unique_str for f in conv_funcs]

    policy = []
    conversion_vals = {}
    for dp in arg_inf_policy:
        depth_policy = []
        for (orig_cf, cf, root_arg_infos) in dp:
            is_conv = orig_cf.unique_str in conv_unq_strs
            cf_args = []

            depth_vars = []
            unq_vars = {}

            for i, ai in enumerate(root_arg_infos):
                ai_type = ai['type']
                
                if(ai_type == ARGINFO_VAR):
                    var = Var.from_ptr(ai['ptr'])
                    base_ptr = var.base_ptr
                    ind = base_var_ind_map[base_ptr]
                    head_val = args[ind]
                    if(len(var.deref_infos) > 0):
                        head_val = args[ind].resolve_deref(var)

                    if(base_ptr not in unq_vars):
                        typ = context.get_type(t_id=var.head_t_id)
                        unq_vars[base_ptr] = Var(typ,chr(97+len(unq_vars)), typ)
                        cf_args.append(head_val)

                    if(is_conv):
                        # print(head_val)
                        conversion_vals[cf.get_ptr()] = (orig_cf(head_val), base_ptr)
                    # print("V_ARG", var, base_ptr, len(unq_vars), unq_vars[base_ptr])

                    depth_vars.append(unq_vars[base_ptr])
                elif(ai_type == ARGINFO_CONST):
                    typ = context.get_type(t_id=ai['t_id'])
                    const = cf.get_const_root_arg(i, typ)
                    if(const not in unq_vars):
                        unq_vars[const] = Var(typ,chr(97+len(unq_vars)), typ)
                        cf_args.append(const)
                        depth_vars.append(unq_vars[const])
                    # print("C_ARG", var, const, len(unq_vars), unq_vars[const])
                elif(ai_type == ARGINFO_FUNC):
                    # TODO: have some format for funcs as args 
                    #  that the planner understands
                    cf_ptr = ai['ptr']
                    if(cf_ptr in conversion_vals):
                        head_val, base_ptr = conversion_vals[cf_ptr]
                        arg_cf = CREFunc.from_ptr(cf_ptr)
                        unq_key = (arg_cf.unique_str, base_ptr)
                    else:
                        unq_key = cf_ptr

                    if(unq_key not in unq_vars):
                        typ = context.get_type(t_id=ai['t_id'])
                        unq_vars[unq_key] = Var(typ,chr(97+len(unq_vars)), typ)
                        if(cf_ptr in conversion_vals):
                            head_val, base_ptr = conversion_vals[cf_ptr]
                            cf_args.append(head_val)
                        
                
                    depth_vars.append(unq_vars[unq_key])

                    
                else:
                    raise ValueError("Bad arg_info type")

            # print(orig_cf, depth_vars, cf_args)
            # print(orig_cf, depth_vars)
            if(not is_conv):
                depth_policy.append((orig_cf(*depth_vars), cf_args))
        policy.append(depth_policy)


    
    # if(len(policy) > 0):
    #     conv_unq_strs = [f.unique_str for f in conv_funcs]
    #     for fn, args in 
    # If empty bit if there were conversions at depth 0 
    if(len(policy[0]) == 0):
        policy = policy[1:]
    return policy


def search_phase(phase_num):
    def phase_decorator(phase_method):
        phase_method.phase_method_phase = phase_num
        return phase_method
    return phase_decorator

@register_how
class NLPSetChaining(BaseHow):
    def __init__(self, agent=None, func_dictionary=None, special_patterns=None,
                 search_depth=3, min_stop_depth=-1, float_to_str=True,
                 spacy_model="en_core_web_trf",  verbosity=0,
                 max_phase_expls=100, display_parse=False, **kwargs):
        self.agent = agent

        if(func_dictionary is None):
            func_dictionary = default_func_dictionary

        if(special_patterns is None):
            special_patterns = defualt_special_patterns

        if(kwargs.get('function_set') is not None):
            # TODO: Use some better way to uniquely id funcs rather than name
            function_set = kwargs['function_set']
            f_names = set(f.name for f in function_set)
            # print(kwargs['function_set'])
                        
            new_fd = {}
            for k,v in func_dictionary.items():
                if(v.name in f_names):
                    new_fd[k] = v
            if(len(new_fd) < len(func_dictionary)):
                func_dictionary = new_fd
                print("Inclusion of function_set has reduced func_dictionary to:", func_dictionary)

            new_sp = {}
            for k,v in special_patterns.items():
                if(v.name in f_names):
                    new_sp[k] = v
            if(len(new_sp) < len(special_patterns)):
                special_patterns = new_sp
                print("Inclusion of function_set has reduced special_patterns to:", special_patterns)
        # else:
        #     raise NotImplemented()
        #     function_set = set(func_dictionary.values())

        # Fill annotated phase methods
        cls = type(self)
        self.phase_methods = []
        for _, val in cls.__dict__.items():
            phase_num = getattr(val, 'phase_method_phase', None)
            if(phase_num is not None):
                self.phase_methods.append((phase_num, val))

        # NLP SC Args
        self.func_dictionary = func_dictionary
        self.fact_types = kwargs.get('fact_types', self.agent.fact_types if (self.agent) else [])

        self.conversion_funcs = []
        if(self.fact_types):
            for ft in self.fact_types:
                for attr, attr_spec in ft.filter_spec('semantic').items():
                    if('conversions' in attr_spec):
                        for f in attr_spec['conversions'].values():
                            self.conversion_funcs.append(f)


        # Lazy import to prevent hard spacy/torch dependency
        from apprentice.agents.cre_agents.how.nlp.parser import TextToPolicyParser

        self.t2p_parser = TextToPolicyParser(func_dictionary, special_patterns,
            spacy_model=spacy_model, use_func_key=False, display_parse=display_parse)

        # SC Args
        self.search_depth = search_depth
        self.min_stop_depth = min_stop_depth
        self.float_to_str = float_to_str
        self.function_set = self.funcs
        self.sc_planner = SetChaining(agent, 
            fact_types=self.fact_types,
            search_depth=search_depth, min_stop_depth=min_stop_depth,
            float_to_str=float_to_str, function_set=self.function_set
            )
        
        self.declared = {}
        self.identifiers = {}
        self.return_phase = None
        
        self.verbosity = verbosity
        if(isinstance(max_phase_expls, int)):
            self.max_phase_expls = {n : max_phase_expls for n, _ in self.phase_methods}
            # Don't limit the number of explanations in the last stage
            n, _ = self.phase_methods[-1]
            del self.max_phase_expls[n]

    @property
    def funcs(self):
        if(not hasattr(self,'_funcs')):
            self._funcs = set(self.func_dictionary.values())
        return self._funcs 

    def clear(self):
        self.declared = {}
        self.identifiers = {}
        self.return_phase = None

    def declare(self, val, var=None):
        if(hasattr(val,'_fact_type')):
            for attr, attr_spec in val._fact_type.filter_spec("semantic").items():
                attr_val = getattr(val, attr, None)
                # TODO: Currently forcing to be float, but should probably 
                #    check for attr_spec['conversions']
                self.declared[attr_val] = var
                try:
                    attr_val = float(attr_val)
                except:
                    pass
                try:
                    self.identifiers[str(int(attr_val))] = attr_val
                except:
                    self.identifiers[str(attr_val)] = attr_val
                self.identifiers[attr_val] = attr_val

        else:
            
            self.declared[val] = var
            # TODO should probably set this up in a way that generalizes better
            #  than string comparison
            try:
                self.identifiers[str(int(val))] = val
            except:
                self.identifiers[str(val)] = val
            self.identifiers[val] = val
    def _policy_preprocessing(self, policy):
        # If an op with the appropriate number of args happens
        #  to be at the end then move it to the beginning.
        for i in range(1,len(policy)):
            dp = policy[i]
            new_dp = []
            for j, (func, args) in enumerate([*dp]):
                if(not isinstance(func, CREFunc)):
                    continue
                if(len(args) >= func.n_args):
                    policy[0].append((func,args))
                else:
                    new_dp.append((func,args))
            policy[i] = new_dp

        # If an op at depth 1 has too few arguments, add it at depth 2 as well.
        if(len(policy) > 0):
            new_dp = []
            for j, (func, args) in enumerate(policy[0]):
                if(len(args) < func.n_args):
                    if(len(policy) < 2): policy.append([])
                    policy[1].append((func,[]))
                else:
                    new_dp.append((func,args))
            policy[0] = new_dp

        # Remove any empty depths caused by the above rearranging
        #  and remove redundant entries
        policy = self._clean_policy(policy)
        return policy

    def _score_expl(self, policy, func, args):
        # print(policy)
        if(len(policy) == 0): return 0.0
        from cre.func import CREFunc, Var

        # print("INP ARGS", args)

        func_policy = func_to_policy(func, args, self.conversion_funcs)

        # print(func, args, func_policy)
        # print("FP", func_policy)
        
        expl_op_matches = 0
        expl_arg_matches = 0
        score = 0
        cnt = 0

        # print("------------")
        # print(op_comp)
        for k, depth_policy in enumerate(policy):
            # print(k)

            depth_expls = func_policy[k] if(k < len(func_policy)) else []
            
            # policy_ops = args.
            if(len(depth_policy) > 0 and len(depth_expls) > 0):
                # Build an alignment matrix between the policies and arugments for the formula
                #  and policy at this depth
                scores = np.zeros((len(depth_policy), len(depth_expls)),dtype=np.float32)
                for i, (p_op, p_args) in enumerate(depth_policy):
                    for j, (e_op, e_args) in enumerate(depth_expls):
                        # print(p_args, e_args)
                        # Add 1 for matching func part
                        scores[i][j] = (p_op.unique_str == e_op.unique_str)

                        # Add 0-.5 for matching number of args
                        # pna, ena = p_op.n_args, e_op.n_args
                        # print(f"n_ARGS:", pna, ena, min(pna, ena)/max(pna, ena))
                        # scores[i][j] += .5*(min(pna, ena)/max(pna, ena))
                        if(len(p_args) > 0):
                            # Add 1 for matching arg content
                            p_cnts, e_cnts = Counter(p_args), Counter(e_args)
                            arg_scr = 0
                            for arg, p_cnt in p_cnts.items():
                                arg_scr += p_cnt-abs(p_cnt-e_cnts.get(arg, 0))
                            # print(f"A_SCR: {arg_scr / max(len(p_args), len(e_args)):.2}", e_args, p_args )
                            scores[i][j] += arg_scr / max(len(p_args), len(e_args)) 

                        # print(k, (i,j), f"{scores[i][j]:.2f}", p_op.name, e_op.name, "" if ~len(p_args) else f"{p_args} {e_args}")
                            
                # Score is the max mapping that projects into the smaller set at this depth.
                if(len(depth_policy) < len(depth_expls)):
                    score += np.sum(np.amax(scores, axis=1))
                else:
                    score += np.sum(np.amax(scores, axis=0))
            cnt += len(depth_policy)

        # If the func_policy is longer than the policy reduce score in proportion
        #  the the remaining bits
        for k in range(k+1, len(func_policy)):
            cnt += len(func_policy[k])
        print(score / cnt, str(func), func_policy)
        return score / cnt

    def _clean_policy(self, policy):
        # Remove any trailing empty depths caused by the above rearranging
        #  and remove redundant entries
        policy = [dp for dp in policy if len(dp) !=0]
        # L = len(np.trim_zeros([len(dp) for dp in policy],trim='b'))
        # policy = policy[:L]
        policy = [list(set( [(op,tuple(args)) for op,args in dp] )) for dp in policy]
        return policy

    def _spread_policy(self, policy, n=1):
        #Copy and extend
        new_policy = [*[[*dp] for dp in policy]]+([[] for _ in range(n)])
        for depth, dp in enumerate(policy):
            for d in range(depth, depth+n+1):
                if(d == depth or d < 0): continue
                for op, args in dp:
                    if(len(args) > 0): continue
                    new_pair = (op,[])
                    new_policy[d].append(new_pair)
        return self._clean_policy(new_policy)

    def _remove_policy_operands(self, policy):
        policy = [[(t[0],[]) if isinstance(t,tuple) else t for t in dp] for dp in policy]
        policy = self._clean_policy(policy)
        return policy

    def _valueify_policy(self, policy):
        # Change the induced policy to be in terms of actual declared
        #  values instead of their string equivalents.
        new_policy = []
        for depth_policy in policy:
            new_depth_policy = []
            for op, operands in depth_policy:
                new_operands = []
                for operand in operands:
                    if(operand in self.identifiers):
                        new_operands.append(self.identifiers[operand])
                    else:
                        try:
                            flt = float(operand)
                        except:
                            continue
                        if(flt in self.identifiers):
                            new_operands.append(self.identifiers[flt])

                new_depth_policy.append((op, new_operands))
            new_policy.append(new_depth_policy)
        policy = new_policy
        return new_policy

    def _score_sort_expls(self, expls, policy):
        sorted_expls = sorted([(self._score_expl(policy, op_comp, args),
                         op_comp, args) for op_comp, args in expls],
                        key=lambda x:-x[0])
        return sorted_expls

    # def apply_sc(self, **kwargs):
    #     kwargs['search_depth'] = kwargs.get('search_depth', self.search_depth)
    #     kwargs['min_stop_depth'] = kwargs.get('min_stop_depth', self.min_stop_depth)
    #     expls = self.sc_planner.get_explanations(self.state, goal, **kwargs,
    #         policy=policy, search_depth=len(policy))

    # -----------------
    # : Search Phases        

    def _remove_scores(self, scored_expls, prune=True):
        if(scored_expls is None or len(scored_expls) == 0):
            return None

        max_score = scored_expls[0][0]

        out_expls = []
        for score, func, args in scored_expls:
            # print(score, func, args)
            if(prune and score == max_score):
                out_expls.append((func, args))
            else:
                break
        return out_expls



    @search_phase(1)
    def phase1(self, goal, policy, search_depth=IGNORE, **kwargs):
        '''Policy w/ ordered operators + operands'''        
        # Run search
        # print(search_depth, kwargs)
        print("P1 Policy:", policy)
        expls = self.sc_planner.get_explanations(self.state, goal, **kwargs,
            policy=policy, search_depth=len(policy))
        scored_expls = self._score_sort_expls(expls, policy)
        return self._remove_scores(scored_expls)

    @search_phase(2)
    def phase2(self, goal, policy, search_depth=IGNORE, **kwargs):
        '''Policy w/o operands'''
        policy = self._remove_policy_operands(policy)        
        expls = self.sc_planner.get_explanations(self.state, goal, **kwargs,
            policy=policy, search_depth=len(policy))

        print("P2 Policy:", policy)     
        return expls

    @search_phase(3)
    def phase3(self, goal, policy, search_depth=IGNORE,
               min_stop_depth=IGNORE, **kwargs):
        '''Policy w/o operands spread by one'''
        orig_length = len(policy)
        policy = self._spread_policy(policy)
        print("P3 Policy:", policy)     

        expls = self.sc_planner.get_explanations(self.state, goal, **kwargs,
            policy=policy, search_depth=len(policy), min_stop_depth=orig_length)
        return expls

    @search_phase(4)
    def phase4(self, goal, policy, min_stop_depth=IGNORE, **kwargs):
        '''Search w/ all mentioned operators'''
        policy = [[t[0] if isinstance(t,tuple) else t for t in dp] for dp in policy]
        func_subset = list(set(chain(*policy)))
        func_subset = [t[0] if isinstance(t, tuple) else t for t in func_subset]

        print("P4 func_subset:", func_subset)     

        expls = self.sc_planner.get_explanations(self.state, goal, **kwargs, 
            function_set=func_subset, min_stop_depth=1)

        return expls

    @search_phase(5)
    def phase5(self, goal, policy, **kwargs):
        '''Search w/ all operators'''
        min_stop_depth = kwargs.get('min_stop_depth', 1)
        search_depth = kwargs.get('search_depth', self.search_depth)

        print("P5 policy", policy, search_depth)
        while(min_stop_depth <= search_depth):
            kwargs['min_stop_depth'] = min_stop_depth
            expls = self.sc_planner.get_explanations(self.state, goal, **kwargs,
                function_set=self.funcs)
            min_stop_depth += 1
            
            print("Min Stop Depth:", min_stop_depth)
            sorted_expls = self._score_sort_expls(expls, policy)
            if(len(sorted_expls) > 0 and sorted_expls[0][0] != 0.0):
                break

        # if(expls is not None):
        #     print(self.funcs)
        #     print("-- P5 EXPLS --", len(expls) if expls else None)
        #     for func, match in expls:
        #         print(func.depth, func, match, func(*match))



        return self._remove_scores(sorted_expls)

        # return expls

    def _search_for_explanations(self, goal, how_help, 
             func_dictionary=None, min_phase=-1, **sc_kwargs):

        if(func_dictionary is None): func_dictionary = self.func_dictionary

        policy = self.t2p_parser(how_help)
        policy = self._policy_preprocessing(policy)
        policy = self._valueify_policy(policy)

        self.policy = policy

        if(self.verbosity > 0):
            print("POLICY", policy)

        # Run all phases
        for phase_num, phase_method in self.phase_methods:
            if(min_phase <= phase_num):
                max_phase_expls = self.max_phase_expls.get(phase_num)

                expls = phase_method(self, goal, policy, **sc_kwargs)
                okay = True
                if(expls is None or 
                   len(expls) == 0 or 
                   (max_phase_expls and len(expls) > max_phase_expls)):
                    okay = False

                if(okay):
                    self.return_phase = phase_num
                    self.num_forward_inferences = self.sc_planner.num_forward_inferences
                    return expls


    def get_explanations(self, state, goal, how_help=None, 
         func_dictionary=None, min_phase=-1, **sc_kwargs):

        print("NLP:", goal, how_help)

        # func_dictionary replaces function_set
        if('function_set' in sc_kwargs):
            del sc_kwargs['function_set'] 

        if(how_help is None or how_help == ""):
            # Use regular SetChaining when no natural language help provided.
            expls = self.sc_planner.get_explanations(state, goal, **sc_kwargs)    
        else:
            # Otherwise clear self and redeclare (Note: SCPlanner will redeclare on own)
            self.clear()

            if sc_kwargs.get('arg_foci') is not None:
                facts = sc_kwargs['arg_foci']
            else:
                wm = state.get("working_memory")
                facts = wm.get_facts()

            self.state = state
            for fact in facts:
                self.declare(fact)

            expls = self._search_for_explanations(goal, how_help, 
                func_dictionary, min_phase, **sc_kwargs)

        if(not isinstance(expls, ExplanationSet)):
            expls = ExplanationSet(expls, sc_kwargs.get('arg_foci'))             
        
        return expls

    def new_explanation_set(self, explanations, *args, **kwargs):
        '''Takes a list of explanations i.e. (func, match) and yields an ExplanationSet object'''
        return ExplanationSet(explanations, *args, **kwargs)



if __name__ == "__main__":
    def test_arg_info_policy():
        from cre.default_funcs import Add, Subtract, Multiply, Divide
        from cre import define_fact
        from numba.types import unicode_type,f8

        BOOP = define_fact("BOOP", {"A" :unicode_type, "B" :f8})

        Add = Add(f8,f8)
        Subtract = Subtract(f8,f8)
        Multiply = Multiply(f8,f8)
        Divide = Divide(f8,f8)

        # 1
        a, b = Var(f8,'a'), Var(f8,'b')
        cf =  Add(Subtract(a,7.0),b)

        policy = func_to_policy(cf, [1,3])
        for i, depth_policy in enumerate(policy):
            print(i, ":", depth_policy)

        # 2
        cf = Add(a,a)
        policy = func_to_policy(cf, [1])
        print(policy)

        # 3
        v = Var(BOOP, 'v')
        cf = Add(v.B,v.B)
        policy = func_to_policy(cf, [BOOP("1", 1)])
        print(policy)


        from apprentice.agents.cre_agents.funcs import Add, Add3, OnesDigit, TensDigit, CastFloat
        a, b = Var(BOOP,'a'), Var(BOOP,'b')
        f = TensDigit((CastFloat(a.A) + CastFloat(b.A)))
        policy = func_to_policy(cf, [BOOP("7", 7), BOOP("6", 6)])
        print(policy)








    from numba.types import f8
    from cre.default_funcs import Add, Subtract, Multiply, Divide
    test_arg_info_policy()
    raise ValueError()

    Add = Add(f8,f8)
    Subtract = Subtract(f8,f8)
    Multiply = Multiply(f8,f8)
    Divide = Divide(f8,f8)

    func_dictionary = {
        "sum" : Add,
        "add" : Add,
        "product" : Multiply,
        "multiply" : Multiply
    }

    planner = NLPSetChaining(func_dictionary=func_dictionary)    
    for n in range(5):
        planner.declare(float(n))

    print("IDENTIFIERS", planner.identifiers)

    text = "Multiply the sum of 1 and 2 with the product of 3 and 4."

    expls = planner._search_for_explanations(36.0, text, search_depth=2)

    print(f"Ended at Phase {planner.return_phase}.")
    for expl in expls:
        print(expl)





