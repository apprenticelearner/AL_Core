

import re
from  colorama import Fore, Back, Style
# def get_value(x):
#     try:
#         return x.value
#     except:
#         return None

# def get_locked(x):
#     try:
#         return x.value
#     except:
#         return None

def style_locked(val,locked):
    if(locked):
        return f'{Back.WHITE}{Fore.BLACK}{val}{Style.RESET_ALL}'
    else:
        return val

def shorten_id(_id):
    num = re.search(r'(\d)',_id)
    num = "" if num is None else num.group(0)
    char = re.search(r'([a-zA-Z])', _id)
    char = "" if char is None else char.group(0)
    short_id = f"{char}{num}"
    return short_id


def shorthand_state_dict(state_dict, do_shorten_id=True):
    vals = []
    for _id, x in state_dict.items():
        val = x.get('value', None)
        val = "" if val is None else val
            
        short_id = shorten_id(_id) if do_shorten_id else _id
        val = f"{short_id}:{val if val else ''}"
        vals.append((val,x.get('locked', False)))
    return " ".join(style_locked(x,l) for x,l in sorted(vals))

def shorthand_state_wm(state):
    d = {}
    for x in state.get("working_memory").get_facts():
        _id = x.id
        d[_id] = {
            'id' : _id,
            'value' : getattr(x,'value',None),
            'locked' : getattr(x,'locked',None)
        }
    return shorthand_state_dict(d)


def shorthand_state_flat(state):
    d = {}
    for x in state.get("flat").get_facts():
        s = str(x)
        _id,rest = s.split(".")
        attr, val = rest.split(" == ")
        dct = d.get(_id,{"id" : _id}) 
        if(attr == 'value'):
            val = val.strip("'")

        if(attr == 'locked'):
            val = True if val == "True" else False

        dct[attr] = val
        d[_id] = dct
    return shorthand_state_dict(d)



def shorthand_state_rel(rel_state):
    d = {}
    for x in rel_state.get_facts():
        s = str(x)
        if(s[0] != "("):
            splt = s.split(".")

            chain, rest = splt[:-1], splt[-1]
            attr, val = rest.split(" == ")
            _id = "".join([shorten_id(y) for y in chain])
            # print("::", _id, s, chain, attr, val)
            dct = d.get(_id, {"id" : _id}) 
            
        elif(s[1:10] == "SkillCand"):

            match = re.match(r"\(SkillCand:, .*, (?P<id>\d+), (?P<rest>.*)\) == (?P<val>.*)", s)
            _id = match.group('id')
            rest = match.group('rest')
            val = match.group('val')

            _id += "?"
            splt = rest.split(", ")
            for i, chain in enumerate(splt):
                _id += "".join([shorten_id(y) for y in chain.split(".")]) 
                _id += ("," if i < len(splt) -1 else "")
            attr = 'value'

        if(attr == 'value'):
            val = val.strip("'")

        if(attr == 'locked'):
            val = True if val == "True" else False

        dct[attr] = val
        d[_id] = dct
    # print(d)
    return shorthand_state_dict(d, do_shorten_id=False)
