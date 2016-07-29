import inspect
from itertools import product


def tup_sai(selection,action,inputs):
    sai = ['sai']
    sai.append(action)
    sai.append(selection)

    if inputs is None:
        pass
    elif isinstance(inputs,list):
        sai.extend(inputs)
    else :
        sai.append(inputs)

    return tuple(sai)

def compute_features(state, features):
    for feature in features:
        num_args = len(inspect.getargspec(features[feature]).args)
        if num_args < 1:
            raise Exception("Features must accept at least 1 argument")
        possible_args = [attr for attr in state]

        for tupled_args in product(possible_args, repeat=num_args):
            new_feature = (feature,) + tupled_args
            values = [state[attr] for attr in tupled_args]
            yield new_feature, features[feature](*values)

def parse_foas(foas):
    return [{'name':foa.split('|')[1],'value':foa.split('|')[2]} for foa in foas]
