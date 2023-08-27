import numpy as np
from numba import literal_unroll
from cre import define_fact, Fact, Var, BaseFact
from cre.obj import cre_obj_get_item
from cre.var import VarType

from apprentice.agents.cre_agents.extending import new_register_decorator, new_register_all

def insert_feature_factory(registry, obj, name=None, *args, **kwargs):
    registry[name.lower().replace("_","")] = (obj,kwargs.get('level',None))

register_feature_factory = new_register_decorator("feature_factory",
                             insert_func=insert_feature_factory,
                             full_descr="feature factory")

# --------------------------------------------------------------
# : SkillCandidates

from numba import njit, types
from numba.types import Tuple, i8, TypeRef, unicode_type
from numba.typed import Dict, List
from cre import TF
from cre.gval import new_gval
from cre.utils import _struct_tuple_from_pointer_arr, _func_from_address, PrintElapse, _struct_from_ptr
from cre.obj import CREObjType
from cre.func import CREFuncType
from cre.utils import cast, decode_idrec
from cre.memset import MemSetType
from numba.core.typing.typeof import typeof
from cre.transform.enumerizer import EnumerizerType
from cre.transform.flattener import GenericFlattenerType

# @njit(cache=True)
@njit(cache=True, locals={"match_ptr_set" : i8[::1]})
def declare_skill_cands(memset, enumerizer, flattener, _how_part,
            uid, match_ptrs, tuple_type, var_tuple_type, cre_func_type):
    how_part = cast(_how_part, cre_func_type)
    # print("--------------")
    # if(len(match_ptrs) > 10):
    #     raise ValueError()
    val_counts = Dict.empty(unicode_type, i8)

    for i in range(len(match_ptrs)):
        match_ptr_set = match_ptrs[i].copy()
        match = _struct_tuple_from_pointer_arr(tuple_type, match_ptr_set)

        try:
            val = how_part(*match[1:])
        except:
            continue
        val_counts[val] = val_counts.get(val, 0)+1

        # Turn match into equivalent base vars
        var_ptrs = np.empty(len(match), dtype=np.int64)
        j = 0
        for m in literal_unroll(match):
            t_id, _, _ = decode_idrec(m.idrec)
            var = flattener.base_var_map[(t_id, cre_obj_get_item(m, unicode_type,0))]
            var_ptrs[j] = cast(var, i8)
            j += 1

        _vars = _struct_tuple_from_pointer_arr(var_tuple_type, var_ptrs)

        # Declare Each Skill Candidate
        head = TF("SkillCand:", how_part, uid, *_vars)
        nom  = enumerizer.to_enum(val)
        gval = new_gval(head, val, nom=nom)
        memset.declare(gval)

    # Declare the counts of the values of each skill candidate 
    for val, count in val_counts.items():
        head = TF("SkillValueCount:", cast(how_part, CREFuncType), uid)
        nom  = enumerizer.to_enum(count)
        gval = new_gval(head, count, nom=nom)
        memset.declare(gval)



@njit(cache=True, locals={"match_ptr_set" : i8[::1]})
def declare_skill_const(memset, flattener, how_part, uid, match_ptrs, tuple_type, var_tuple_type):
    for i in range(len(match_ptrs)):
        match_ptr_set = match_ptrs[i]
        match = _struct_tuple_from_pointer_arr(tuple_type, match_ptr_set)

        var_ptrs = np.empty(len(match), dtype=np.int64)
        j = 0
        for m in literal_unroll(match):
            t_id, _, _ = decode_idrec(m.idrec)
            var = flattener.get_base_var(t_id, cre_obj_get_item(m, unicode_type,0))
            var_ptrs[j] = cast(var, i8)
            j += 1

        _vars = _struct_tuple_from_pointer_arr(var_tuple_type, var_ptrs)

        val = how_part
        head = TF("SkillCand:", how_part, uid, *_vars)
        gval = new_gval(head, val)
        memset.declare(gval)


_declare_skill_cands_cache={}
def get_declare_skill_cands_impl(how_part):
    sig = getattr(how_part, "signature", None)
    if(sig is None):
        return_type = typeof(how_part)
        if(return_type not in _declare_skill_cands_cache):
            tuple_type = Tuple(tuple([TypeRef(CREObjType)]))
            var_tuple_type = Tuple(tuple([TypeRef(VarType)]))
            @njit(types.void(MemSetType, EnumerizerType, GenericFlattenerType, return_type, unicode_type, i8[:,::1]), cache=True)
            def _declare_skill_const(memset, enumerizer, flattener, how_part, uid, match_ptrs):
                declare_skill_const(memset, flattener, how_part, uid, match_ptrs, tuple_type, var_tuple_type)
            _declare_skill_cands_cache[return_type] = _declare_skill_const
        sig = return_type
    elif(sig not in _declare_skill_cands_cache):
        tuple_type = Tuple(tuple([TypeRef(CREObjType),*[TypeRef(x) for x in sig.args]]))
        var_tuple_type = Tuple(tuple([TypeRef(VarType)]*(1+len(sig.args))))
        cre_func_type = how_part.precise_type#CREFuncTypeClass(return_type=sig.return_type, arg_types=sig.args, is_composed=True)
        # call_type = types.FunctionType(sig)#.return_type(i8[::1]))
        # check_type = types.FunctionType(types.boolean(*sig.args))#.return_type(i8[::1]))
        
        @njit(types.void(MemSetType, EnumerizerType, GenericFlattenerType, CREFuncType, unicode_type, i8[:,::1]), cache=True)
        def _declare_skill_cands(memset, enumerizer, flattener, how_part, uid, match_ptrs):
            declare_skill_cands(memset, enumerizer, flattener, how_part, uid,
             match_ptrs, tuple_type, var_tuple_type, cre_func_type)
        
        _declare_skill_cands_cache[sig] = _declare_skill_cands
    return _declare_skill_cands_cache[sig]


@register_feature_factory(level='agent')
def SkillCandidates(agent, state, feat_state):

    # print("*********************")
    # print(state.get("working_memory"))
    # print(feat_state)
    # print("*********************")
    # with PrintElapse("\t\t\tSkillCandidates"):
    # print("N SKILLS:", len(agent.skills))
    for uid, skill in agent.skills.items():
        # print(">>", skill)
        # Turn the matches into a 2d array of pointers
        # with PrintElapse("\t\t\tget_matches"):
        matches = list(skill.where_lrn_mech.get_matches(state))

        n = len(matches) 
        if(n > 0):
            m = len(matches[0])
            match_ptrs = np.empty((n,m),dtype=np.int64)
            for i, match in enumerate(matches):
                # print("<<", [m.id for m in match])
                for j, fact in enumerate(match):
                    match_ptrs[i][j] = fact.get_ptr()

            # with PrintElapse("\t\t\tget_declare"):
            msc = get_declare_skill_cands_impl(skill.how_part)
            # with PrintElapse("\t\t\tcall_declare"):
            # print(":::", skill.how_part, skill.uid)
            msc(feat_state, agent.enumerizer, agent.flattener,
                    skill.how_part, skill.uid, match_ptrs)
        # else:
            # print(";;;", skill.how_part)
    return feat_state



# -----------------------------------------------------------------
# : Match

@njit(types.void(MemSetType,GenericFlattenerType, i8[::1]), cache=True, locals={"match_ptr_set" : i8[::1]})
def declare_match(memset, flattener, match_ptrs):
    if(len(match_ptrs) > 1):
        for i, m_ptr in enumerate(match_ptrs[1:]):
            m = _struct_from_ptr(CREObjType, m_ptr)
            t_id, _, _ = decode_idrec(m.idrec)
            var = flattener.get_base_var(t_id, cre_obj_get_item(m, unicode_type,0))
            tup =  TF(f"Arg{i}:", var)
            arg_tup = new_gval(tup,"")
            memset.declare(arg_tup)



@register_feature_factory(level='when')
def Match(when_mech, state, feat_state, match):
    match_ptrs = np.empty((len(match)),dtype=np.int64)
    for i, m in enumerate(match):
        match_ptrs[i] = m.get_ptr()

    declare_match(feat_state, when_mech.agent.flattener, match_ptrs)
    return feat_state
    

    

