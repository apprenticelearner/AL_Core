from random import choice
from .extending import new_register_decorator, registries
from apprentice.agents.cre_agents import registries, register_when
from apprentice.shared import rand_agent_uid

# Registries for the optional mechanisms used when acting without a tutor.
#  Their entries are defined in explore.py and learning_mechs/.
register_explorer = new_register_decorator("explorer", full_descr="explorer")
register_effect = new_register_decorator("effect", full_descr="effect-learning mechanism")
register_evaluator = new_register_decorator("evaluator", full_descr="evaluation mechanism")


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
    if(lst is None): return None
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
        self.process_cls = config_get(['process_type','process_cls','process','process_learner'], 
            default=None, registry=registries["process"])
        
        
        # Standardize arguments for mechanisms.
        self.when_args = config_get(['when_args'], {})
        self.where_args = config_get(['where_args'], {})
        self.how_args = config_get(['how_args','planner_args'], {})
        self.which_args = config_get(['which_args'], {})
        self.process_args = config_get(['process_args'], {})

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
        
        

        self.function_set = resolve_list(config.get("function_set", []), registries['func'])
        self.feature_set = resolve_list(config.get("feature_set", []), registries['func'])

        self.should_find_neighbors = config_get(['find_neighbors', 'should_find_neighbors'],
         default=False)
        self.suggest_uncert_neg = config_get(['suggest_uncert_neg', 'suggest_uncertain_negatives'], default=False)
        self.track_rollout_preseqs = config_get(['track_rollout_preseqs'], default=False)
        self.implicit_reward_kinds = set(config_get(['implicit_reward_kinds'], default=[]))

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

        self.action_filter = config_get("action_filter",
            default='thresholds', registry=registries['skill_app_filter'])        

        self.action_filter_args = config_get("action_filter_args", {})

        self.constraints = config_get(['constraints','environment', 'env'], default='html',
            registry=registries.get('constraint',[]))

        self.conversions = list(registries.get('conversion',[]))

        self.config = {k:v for k,v in config.items() if k not in covered}

        self.error_on_bottom_out = config_get("error_on_bottom_out",
            default=False)

        self.one_skill_per_match = config_get("one_skill_per_match",
            default=False)

        self.bottomout_exceptions = config_get("bottomout_exceptions",
            default=[('done', None, None)])

        self.sep_in_proc_when = config_get("sep_in_proc_when",
            default=True)

        # ---- Options for acting without a tutor (all default to the tutor behaviour) ----

        # When explaining an action from existing skills, only consider skills
        #  learned for the same action type. Off by default: tutors mostly use one
        #  action type, sometimes under several names, and rely on the looser match.
        self.strict_action_types = config_get(["strict_action_types"],
            default=False)

        # Induce a skill from an action even when its first example is negative,
        #  and fit where a skill applies from every example whatever its outcome.
        #  Off by default: with a tutor a skill is something that has worked once.
        self.induce_from_negative = config_get(["induce_from_negative", "learn_from_failure"],
            default=False)

        # What act() falls back on when no skill applies and the caller has said
        #  which actions are available. None keeps the tutor behaviour: return
        #  nothing and wait to be shown.
        self.explorer = config_get(["explorer"],
            default=None, registry=registries['explorer'])
        self.explorer_args = config_get("explorer_args", {})

        # Optional mechanism that learns what actions do (see learning_mechs/effect.py).
        self.effect_cls = config_get(["effect_type", "effect", "effect_cls", "effect_learner"],
            default=None, registry=registries['effect'])
        self.effect_args = config_get(["effect_args"], {})

        # Optional mechanism that turns evidence about how an action went into the
        #  agent's own evaluation of it (see learning_mechs/evaluate.py). None means
        #  the reward given to train() is taken at face value, as with a tutor.
        self.evaluator_cls = config_get(["evaluator_type", "evaluator", "evaluator_cls"],
            default=None, registry=registries['evaluator'])
        self.evaluator_args = config_get(["evaluator_args"], {})

    def __init__(self, **config):
        self.uid = rand_agent_uid()
        self.standardize_config(config)

    def request(self, *args, **kwargs):
        ''' Legacy method name : pipe into act ''' 
        return self.act(*args, **kwargs)

    def is_bottom_out_exception(self, action):
        s, a, i = action.as_tuple()
        for (ex_s, ex_a, ex_i) in self.bottomout_exceptions:            
            if((ex_s is None or s == ex_s) and
               (ex_a is None or a == ex_a) and
               (ex_i is None or tuple(ex_i.items()) == tuple(i.items()))):
                return True
        return False

# -------------------------------------------------------------------------
# : SkillApp Filters
register_skill_app_filter = new_register_decorator("skill_app_filter", full_descr="skill application filter")

@register_skill_app_filter
def thresholds(state, skill_apps, thresholds=[.5, 0, -.5, -.999]):
    n_pos = 0
    thresh_groups = [[] for _ in range(len(thresholds))]
    for skill_app in skill_apps:
        when_pred = getattr(skill_app,'when_pred', None)
        if(when_pred is None): when_pred = 1
        for i, thresh in enumerate(thresholds):
            if(when_pred >= thresh):
                thresh_groups[i].append(skill_app)
                break
    for i, tg in enumerate(thresh_groups):
        if(len(tg) > 0):
            return tg
    return []

@register_skill_app_filter
def positive_only(state, skill_apps, min_pred=0.0):
    ''' Keep only skill applications that are not expected to go badly.
        'thresholds' falls back to its best non-empty group even when every
        prediction in it is negative, which suits a tutor author who wants
        to see what the agent would try. An agent that acts on what it is
        handed should not be handed what it predicts will fail.'''
    out = []
    for skill_app in skill_apps:
        when_pred = getattr(skill_app, 'when_pred', None)
        if(when_pred is None or when_pred > min_pred):
            out.append(skill_app)
    return out
# -------------------------------------------------------------------------
# : SkillApp Choosers

register_skill_app_chooser = new_register_decorator("skill_app_chooser", full_descr="skill application chooser")

def _get_which_utility(state, skill_app):
    return skill_app.skill.which_lrn_mech.get_utility(state, skill_app)

def _sort_on_utility(state, skill_apps):
    return sorted(skill_apps,key=lambda sa : _get_which_utility(state, sa))

@register_skill_app_chooser
def max_which_utility(state, skill_apps):
    return _sort_on_utility(state, skill_apps)[-1]

@register_skill_app_chooser
def min_which_utility(state, skill_apps):
    return _sort_on_utility(state, skill_apps)[0]

@register_skill_app_chooser
def random(state, skill_apps):
    return choice(skill_apps)


