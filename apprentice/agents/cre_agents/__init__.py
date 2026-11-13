from .extending import new_register_all, new_register_decorator, registries
from .environment import register_fact, register_all_facts
from .funcs import register_func, register_all_funcs

from .learning_mechs.registers import (
    register_how,
    register_where,
    register_when,
    register_which,
    register_process
)

from .learning_mechs.how import how
from .learning_mechs import where 
from .learning_mechs import when 
from .learning_mechs import which
from .learning_mechs.process import process

from .cre_agent import CREAgent
from .feature_factory import register_feature_factory

# Optional mechanisms for acting without a tutor. Importing them fills their
#  registries; none is used unless an agent is configured to.
from .dipl_base import register_explorer, register_effect, register_evaluator
from . import explore
from .learning_mechs import effect
from .learning_mechs import evaluate
