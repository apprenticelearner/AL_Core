from numba.types import f8, string, boolean
from apprentice.agents.cre_agents.extending import registries, new_register_decorator, new_register_all
from cre import CREFunc
import numpy as np

# --------------
# : Conversion Functions float/str

register_conversion = new_register_decorator("conversion", full_descr="Conversions between types")

@register_conversion(name="CastFloat")
@CREFunc(shorthand = 'f8({0})')
def CastFloat(a):
    return float(a)

@register_conversion(name="CastStr")
@CREFunc(shorthand = 'str({0})')
def CastStr(a):
    return str(a)

##### Define all CREFuncs above this line #####
