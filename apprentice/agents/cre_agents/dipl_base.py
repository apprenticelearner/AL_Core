from .extending import new_register_decorator, registries
from apprentice.agents.cre_agents import registries, register_when
from apprentice.shared import rand_agent_uid


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

def resolve_list(lst, registry):
    out = []
    for obj in lst:
        if(isinstance(obj, str)):
            name = obj.lower().replace("_","")
            obj = registry[name]
        out.append(obj)
    return out


class BaseDIPLAgent(object):
# ------------------------------------------------------------------------
# : __init__
    def standardize_config(self, config):
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
        
        
        # Standardize arguments for mechanisms.
        self.when_args = config_get(['when_args'], {})
        self.where_args = config_get(['where_args'], {})
        self.how_args = config_get(['how_args','planner_args'], {})
        self.which_args = config_get(['which_args'], {})

        # print(registries['feature_factory'])

        # Handle extra features
        self.extra_features = []
        for extra_feature, level in resolve_list(config.get('extra_features',[]), registries['feature_factory']):
            # print("<<", extra_feature, level)
            if(level == 'agent' or level is None):
                self.extra_features.append(extra_feature)
            elif(level == 'when'):
                ef = self.when_args.get('extra_features', [])
                ef.append(extra_feature)
                self.when_args['extra_features'] = ef
        
        

        self.function_set = resolve_list(config.get("function_set",[]), registries['func'])
        self.feature_set = resolve_list(config.get("feature_set",[]), registries['func'])

        self.should_find_neighbors = config_get(['find_neighbors', 'should_find_neighbors'],
         default=False)

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


        self.fact_types = config_get(['fact_types','environment', 'env'], default='html',
            registry=registries.get('fact_set',[]))

        self.action_types = config_get(['action_types'], default='html',
            registry=registries.get('action_type_set',{}))

        self.action_chooser = config_get("action_chooser",
            default='max_which_utility', registry=registries['skill_app_chooser'])

        self.explanation_chooser = config_get("explanation_chooser",
            default='max_which_utility', registry=registries['skill_app_chooser'])

        self.constraints = config_get(['constraints','environment', 'env'], default='html',
            registry=registries.get('constraint',[]))

        self.conversions = list(registries.get('conversion',[]))

        self.config = {k:v for k,v in config.items() if k not in covered}

    def __init__(self, **config):
        self.uid = rand_agent_uid()
        self.standardize_config(config)

    def request(self, *args, **kwargs):
        ''' Legacy method name : pipe into act ''' 
        return self.act(*args, **kwargs)


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


