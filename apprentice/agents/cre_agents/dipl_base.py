from .extending import new_register_decorator, registries
from apprentice.agents.cre_agents import registries, register_when


def _config_get(config, registry, covered, names, default):
    if(not isinstance(names,list)): names = [names]
    for name in names:
        covered.add(name)

    obj = None
    for name in names:
        if(name in config): 
            obj = config[name]
            break

    if(obj is None): obj = default

    if(isinstance(obj,str) and registry is not None):
        name = obj.lower().replace("_","")
        return registry[name]
    else:
        return obj

    return default

def resolve_op_set(op_set):
    out = []
    op_registry = registries['op']
    for op in op_set:
        if(isinstance(op, str)):
            name = op.lower().replace("_","")
            op = op_registry[name]
        out.append(op)
    return out


class BaseDIPLAgent(object):
# ------------------------------------------------------------------------
# : __init__
    def standardize_config(self, config):
        print(config)
        covered = set()
        config_get = lambda names, default, registry=None : _config_get(config, registry, covered, names, default)

        # Get learning mechanism classes.
        self.how_cls = config_get(['how_type','how','how_cls','how_learner', 'planner'], 
            default='setchaining', registry=registries["how"])
        self.where_cls = config_get(['where_type','where_cls','where','where_learner'], 
            default='antiunify', registry=registries["where"])
        self.when_cls = config_get(['when_type','where_cls','when','when_learner'], 
            default='sklearndecisiontree', registry=registries["when"])
        self.which_cls = config_get(['which_type','which_cls','which','which_learner'], 
            default='proportion_correct', registry=registries["which"])
        
        print(self.where_cls)
        # Standardize arguments for mechanisms.
        self.when_args = config_get(['when_args'], {})
        self.where_args = config_get(['where_args'], {})
        self.how_args = config_get(['how_args','planner_args'], {})
        self.which_args = config_get(['which_args'], {})

        self.function_set = resolve_op_set(config.get("function_set",[]))
        self.feature_set = resolve_op_set(config.get("feature_set",[]))
        self.should_encode_neighbors = config_get(['encode_neighbors', 'should_encode_neighbors'], True)

        # Reroute config options that user might define at the agent level
        #  but belong at the learning mechanism level.
        if('extra_features' not in self.when_args):
            self.when_args['extra_features'] = config_get(['extra_features'], 
                default=[])

        if('search_depth' not in self.how_args):
            self.how_args['search_depth'] = config_get(['search_depth'], 
                default=2)

        if('function_set' not in self.how_args):
            self.how_args['function_set'] = self.function_set


        self.fact_types = config_get('fact_types', default='html',
            registry=registries.get('fact_set',[]))

        self.action_chooser = config_get("action_chooser",
            default='max_which_utility', registry=registries['skill_app_chooser'])

        self.explanation_chooser = config_get("explanation_chooser",
            default='max_which_utility', registry=registries['skill_app_chooser'])

        self.config = {k:v for k,v in config.items() if k not in covered}

    def __init__(self, **config):
        self.standardize_config(config)

    def request(self, *args, **kwargs):
        ''' Legacy method name : pipe into act ''' 
        self.act(*args, **kwargs)


# -------------------------------------------------------------------------
# : SkillApp Choosers

register_skill_app_chooser = new_register_decorator("skill_app_chooser", full_descr="skill application chooser")

def _get_which_utility(state, skill_app):
    return skill_app.skill.which_lrn_mech.get_utility(state, skill_app.match)

def _sort_on_utility(state, skill_apps):
    return sorted(skill_apps,key=lambda sa : _get_which_utility(state,sa))

@register_skill_app_chooser
def max_which_utility(state, skill_apps):
    return _sort_on_utility(state, skill_apps)[-1]

@register_skill_app_chooser
def min_which_utility(state, skill_apps):
    return _sort_on_utility(state, skill_apps)[0]

@register_skill_app_chooser
def random(state, skill_apps):
    return choice(skill_apps)




