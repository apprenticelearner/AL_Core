import numpy as np
from cre import define_fact, Fact
from apprentice.agents.cre_agents.extending import new_register_decorator, new_register_all

def insert_feature_factory(registry, obj, name=None, *args, **kwargs):
    registry[name.lower().replace("_","")] = (obj,kwargs.get('level',None))

register_feature_factory = new_register_decorator("feature_factory",
                             insert_func=insert_feature_factory,
                             full_descr="feature factory")

# --------------------------------------------------------------
# : SkillCandidates

from numba import njit, types
from numba.types import Tuple, i8, TypeRef
from cre import TF
from cre.gval import new_gval
from cre.utils import _struct_tuple_from_pointer_arr, _func_from_address, PrintElapse, _struct_from_ptr
from cre.obj import CREObjType
from cre.func import CREFuncType
from cre.utils import cast
from cre.memset import MemSetType
from numba.core.typing.typeof import typeof
from cre.transform.enumerizer import EnumerizerType

# @njit(cache=True)
@njit(cache=True, locals={"match_ptr_set" : i8[::1]})
def declare_skill_cands(memset, enumerizer, _how_part,
            id_num, match_ptrs, tuple_type, cre_func_type):
    how_part = cast(_how_part, cre_func_type)
    # print("--------------")
    # if(len(match_ptrs) > 10):
    #     raise ValueError()
    for i in range(len(match_ptrs)):
        match_ptr_set = match_ptrs[i].copy()
        match = _struct_tuple_from_pointer_arr(tuple_type, match_ptr_set)

        try:
            val = how_part(*match[1:])
        except:
            continue
        # print("VAL@", match[1:], val)
        head = TF("SkillCand:", cast(how_part, CREFuncType), id_num, *match)
        # print(head)
        nom  = enumerizer.enumerize(val)
        # TODO: Should also try to make float
        gval = new_gval(head, val, nom=nom)
        memset.declare(gval)

@njit(cache=True, locals={"match_ptr_set" : i8[::1]})
def declare_skill_const(memset, how_part, id_num, match_ptrs, tuple_type):
    for i in range(len(match_ptrs)):
        match_ptr_set = match_ptrs[i]
        match = _struct_tuple_from_pointer_arr(tuple_type, match_ptr_set)
        val = how_part
        head = TF("SkillCand:", how_part, id_num, *match)
        gval = new_gval(head, val)
        memset.declare(gval)


_declare_skill_cands_cache={}
def get_declare_skill_cands_impl(how_part):
    sig = getattr(how_part, "signature", None)
    if(sig is None):
        return_type = typeof(how_part)
        if(return_type not in _declare_skill_cands_cache):
            tuple_type = Tuple(tuple([TypeRef(CREObjType)]))
            @njit(types.void(MemSetType, EnumerizerType, return_type, i8, i8[:,::1]), cache=True)
            def _declare_skill_const(memset, enumerizer, how_part, id_num, match_ptrs):
                declare_skill_const(memset, how_part, id_num, match_ptrs, tuple_type)
            _declare_skill_cands_cache[return_type] = _declare_skill_const
        sig = return_type
    elif(sig not in _declare_skill_cands_cache):
        tuple_type = Tuple(tuple([TypeRef(CREObjType),*[TypeRef(x) for x in sig.args]]))
        cre_func_type = how_part.precise_type#CREFuncTypeClass(return_type=sig.return_type, arg_types=sig.args, is_composed=True)
        # call_type = types.FunctionType(sig)#.return_type(i8[::1]))
        # check_type = types.FunctionType(types.boolean(*sig.args))#.return_type(i8[::1]))
        
        @njit(types.void(MemSetType, EnumerizerType, CREFuncType, i8, i8[:,::1]), cache=True)
        def _declare_skill_cands(memset, enumerizer, how_part, id_num, match_ptrs):
            declare_skill_cands(memset, enumerizer, how_part, id_num,
             match_ptrs, tuple_type, cre_func_type)
        
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
    for skill in agent.skills:
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
            # print(":::", skill.how_part, skill.id_num)
            msc(feat_state, agent.enumerizer, skill.how_part, skill.id_num, match_ptrs)
        # else:
            # print(";;;", skill.how_part)
    return feat_state



# -----------------------------------------------------------------
# : Match

@njit(types.void(MemSetType,i8[::1]), cache=True, locals={"match_ptr_set" : i8[::1]})
def declare_match(memset, match_ptrs):
    if(len(match_ptrs) > 1):
        for i, m_ptr in enumerate(match_ptrs[1:]):
            m = _struct_from_ptr(CREObjType, m_ptr)
            tup =  TF(f"Arg{i}:", m)
            arg_tup = new_gval(tup,"")
            memset.declare(arg_tup)


@register_feature_factory(level='when')
def Match(when_mech, state, feat_state, match):
    match_ptrs = np.empty((len(match)),dtype=np.int64)
    for i, m in enumerate(match):
        match_ptrs[i] = m.get_ptr()

    declare_match(feat_state, match_ptrs)
    return feat_state
    

    

