from random import choices as sample_w_replacement
import string
from abc import ABC, ABCMeta, abstractmethod

# -------------------------------------------------------------------------
# : SAI
class SAI(object):
    ''' 
    A basic SAI type. Useful for standardizing many possible SAI formats.
    This type does not define an API requirement. Agents do not need to 
    take/return this precise type, although doing so is recommended. 
    WARNING: NEVER directly edit or monkey-patch this class, if changes are needed
    make a subclass and use that directly within your agent implementation.
    Subclasses are free to change the datatypes of seleciton, action_type, 
    and inputs but must overload as_tuple() so it returns (str, str, dict).
    (e.g. see CREAgent)
    '''

    slots = ('selection', 'action_type', 'inputs')
    def __new__(cls, *args):
        if(len(args) == 1):
            # Translates from object, list, tuple, or dict.
            inp = args[0]
            if(isinstance(inp, SAI)):
                return inp

            elif(hasattr(inp, 'selection')):
                selection = inp.selection
                action_type = getattr(inp, 'action_type', None)
                if(action_type is None):
                    action_type = getattr(inp, 'action', None)
                inputs = inp.inputs
            elif(isinstance(inp, (list,tuple))):
                selection, action_type, inputs = inp
            elif(isinstance(inp, dict)):
                selection = inp['selection']
                action_type = inp.get('action_type', inp.get('action'))
                if(action_type is None):
                    raise KeyError("'action_type' | 'action'")
                inputs = inp['inputs']
            else:
                raise ValueError(f"Unable to translate {inp} to SAI.")
        else:
            selection, action_type, inputs = args

        self = super().__new__(cls)
        self.selection = selection
        self.action_type = action_type
        self.inputs = inputs
        return self
    
    def __iter__(self):
        return iter(self.as_tuple())

    def as_tuple(self):
        return (self.selection, self.action_type, self.inputs)

    def __eq__(self, other):
        if(hasattr(other,'as_tuple')): 
            other = other.as_tuple()
        self_tup = self.as_tuple()

        # Note: Subclasses should gaurentee an as_tuple() representation
        #  of the form (str, str, dict). Thus this works in general.
        #print(self_tup, other, self_tup == other)
        return self_tup == other

    def __getitem__(self, item):
        if(isinstance(item, int)):
            return self.as_tuple()[item]
        else:
            if(item == "selection"):
                return self.selection
            elif(item == "action_type" or item == "action"):
                return self.action_type
            elif(item == "inputs"):
                return self.inputs
            else:
                raise KeyError(f"SAI has no such member item: {item!r}.")

    def __hash__(self):
        sel, at, inps = self.as_tuple()
        return hash((sel, at, tuple(sorted(inps.items()))))
    
    def get_info(self):
        sel_str, at_str, inputs = self.as_tuple()
        return {
            'selection' :  sel_str,
            'action_type' :  at_str,
            'inputs' :  self.inputs,
        }

    def __repr__(self):
        sel_str, at_str, inputs = self.as_tuple()
        return f"SAI({sel_str}, {at_str}, {inputs})"

    __str__ = __repr__

# -------------------------------------------------------------------------
# : Random unique ids

alpha_num_chars = string.ascii_letters + string.digits

# Reasoning for choice of length of 30: 62^30 = ~10^53 is more permutations 
#  than a SHA1 hash (2^160 = ~10^49). So very unlikely to produce hash conflicts.
def rand_skill_uid():
    return f"SK_{''.join(sample_w_replacement(alpha_num_chars, k=30))}"

def rand_skill_app_uid():
    return f"A_{''.join(sample_w_replacement(alpha_num_chars, k=30))}"

def rand_state_uid():
    return f"S_{''.join(sample_w_replacement(alpha_num_chars, k=30))}"

def rand_agent_uid():
    return f"AG_{''.join(sample_w_replacement(alpha_num_chars, k=30))}"


# ------------------------------------------------------------------------
# : Time Logging Utils

def reject_outliers(data, m = 2.):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/mdev if mdev else np.zeros(len(d))
    return data[s<m]

import time
import numpy as np
class ElapseLogger():
    def __init__(self, name):
        self.name = name
        self.durations = []

    def __enter__(self):
        self.t0 = time.time_ns()/float(1e6)
    def __exit__(self,*args):
        self.t1 = time.time_ns()/float(1e6)
        self.durations.append(self.t1-self.t0)

    def __str__(self):
        durs = reject_outliers(np.array(self.durations))
        s = f"{self.name}:\n"
        s += f"  Avergage Elapse Time: {np.mean(durs):.3f}\n"
        s += f"  Std Elapse Time: {np.std(durs):.3f}\n"
        return s




from abc import ABC, ABCMeta
from typing import Any


def sai_from_args(args):
    # Case 1: Args has one object
    if(len(args) == 1):
        val = args[0]
        if(hasattr(val, 'as_tuple')):
            return val.as_tuple()

        elif(hasattr(val, 'selection')):
            selection = val.selection
            action_type = getattr(val, 'action_type', None)
            if(action_type is None):
                action_type = getattr(val, 'action', None)
            inp = inp.input
        elif(isinstance(val, (list,tuple))):
            selection, action_type, inp = val
        elif(isinstance(val, dict)):
            selection = val['selection']
            action_type = val.get('action_type', val.get('action'))
            if(action_type is None):
                raise KeyError("'action_type' | 'action'")
            inp = val['input']
        else:
            raise ValueError(f"Unable to translate {val} to Action.")

    # Case 2: Args is a tuple
    else:
        selection, action_type, inp = args

    return selection, action_type, inp


class ActionLike(ABCMeta): 
    selection : str
    action_type : str
    input : Any
    slots = ('selection', 'action_type', 'input')

    def __init__(self, *args):
        sel, at, inp = sai_from_args(args)
        self.selection = sel
        self.action_type = at
        self.input = inp

    @abstractmethod
    def as_tuple(self):
        return (self.selection, self.action_type, self.input)


class Action(metaclass=ActionLike):
    selection : str
    action_type : str
    input : Any
    sel_obj : Any
    action_type_inst : "ActionType"

    slots = ('selection', 'action_type', 'input', 'selection_inst', 'action_type_inst')

    def __init__(self, *args):
        sel, at, inp = sai_from_args(args)

        if(not isinstance(sel, str)):
            self.selection_inst = sel
            sel = sel.id

        if(not isinstance(at, str)):
            self.action_type_inst = at
            at = at.name

        self.selection = sel
        self.action_type = at
        self.input = inp

    def as_tuple(self):
        return (self.selection, self.action_type, self.input)

    @property
    def sai(self):
        return self.as_tuple()

    def long_hash():
        return unique_hash(self.as_tuple())

    def __str__(self):
        return f"{self.action_type}({self.selection}, {self.input})"

    __repr__  = __str__
        




    

