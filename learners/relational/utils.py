"""
Standard functions used to support relational learning
"""
from planners.fo_planner import Operator
from planners.fo_planner import build_index
from planners.fo_planner import subst
from planners.fo_planner import extract_strings
from planners.fo_planner import is_variable


def covers(h, x, initial_mapping):
    """
    Returns true if h covers x
    """
    index = build_index(x)
    operator = Operator(tuple(['Rule']), h, [])
    for m in operator.match(index, initial_mapping=initial_mapping):
        return True
    return False


def rename(mapping, literal):
    return tuple(mapping[ele] if ele in mapping else rename(mapping, ele) if
                 isinstance(ele, tuple) else ele for ele in literal)


def generalize_literal(literal, gensym):
    return (literal[0],) + tuple(ele if is_variable(ele) else
                                 # '?gen%s' % hash(ele)
                                 gensym()
                                 for ele in literal[1:])

def remove_vars(literal):
    return tuple('XXX' + ele if is_variable(ele) else remove_vars(ele) if
                 isinstance(ele, tuple) else ele for ele in literal)
