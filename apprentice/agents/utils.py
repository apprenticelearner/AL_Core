import inspect
from itertools import product
from random import uniform

def weighted_choice(choices):
   total = sum(w for c, w in choices)
   r = uniform(0, total)
   upto = 0
   for c, w in choices:
      if upto + w >= r:
         return c, w
      upto += w
   assert False, "Shouldn't get here"

def gen_varnames(start=0, end=float('inf')):
    while start < end:
        var = ""
        val = start
        while val > 25:
            r = val % 26
            val = val // 26
            var = chr(r + ord('A')) + var
        if var == "":
            var = chr(val + ord('A')) + var
        else:
            var = chr(val-1 + ord('A')) + var
        yield var
        start += 1

def tup_sai(selection,action,inputs):
    sai = ['sai']
    sai.append(action)
    sai.append(selection)

    if inputs is None:
        pass
    elif isinstance(inputs, list):
        sai.extend(inputs)
    else:
        sai.append(inputs)

    return tuple(sai)

def compute_features(state, features):
    original_state = {a: state[a] for a in state}
    for feature in features:
        num_args = len(inspect.getargspec(features[feature]).args)
        if num_args < 1:
            raise Exception("Features must accept at least 1 argument")

        possible_args = [attr for attr in original_state]

        for tupled_args in product(possible_args, repeat=num_args):
            new_feature = (feature,) + tupled_args
            values = [state[attr] for attr in tupled_args]
            try:
                yield new_feature, features[feature](*values)
            except Exception as e:
                pass

def parse_foas(foas):
    return [{'name':foa.split('|')[1], 'value':foa.split('|')[2]} for foa in foas]
