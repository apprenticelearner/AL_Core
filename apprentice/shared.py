from random import choices as sample_w_replacement
import string

# -------------------------------------------------------------------------
# : SAI
class SAI(object):
    ''' 
    A basic SAI type. Useful for standardizing many possible SAI formats.
    This type does not define an API requirement. Agents do not need to 
    take/return this precise type, although doing so is recommended. 
    WARNING: NEVER directly edit or monkey-patch this class, if changes are needed
    make a subclass and use that directly within your agent implementation.
    (e.g. see CREAgent)
    '''

    slots = ('selection', 'action_type', 'inputs')
    def __init__(self, *args):
        if(len(args) == 1):
            # Translates from object, list, tuple, or dict.
            inp = args[0]
            if(hasattr(inp, 'selection')):
                selection = inp.selection
                action_type = getattr(inp, 'action_type', inp.action)
                inputs = inp.inputs
            elif(isinstance(inp, (list,tuple))):
                selection, action_type, inputs = inp
            elif(isinstance(inp, dict)):
                print(inp)
                selection = inp['selection']

                action_type = inp.get('action_type', inp.get('action'))
                if(action_type is None):
                    raise KeyError("'action_type' | 'action'")
                inputs = inp['inputs']
            else:
                raise ValueError(f"Unable to translate {inp} to SAI.")
        else:
            selection, action_type, inputs = args

        self.selection = selection
        self.action_type = action_type
        self.inputs = inputs
    
    def __iter__(self):
        return iter(self.as_tuple())

    def as_tuple(self):
        return (self.selection, self.action_type, self.inputs)

    def __eq__(self, other):
        if(hasattr(other,'as_tuple')): 
            other = other.as_tuple()
        return self.as_tuple() == other

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
