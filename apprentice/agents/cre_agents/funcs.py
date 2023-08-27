from numba.types import f8, string, boolean
from apprentice.agents.cre_agents.extending import registries, new_register_decorator, new_register_all
from apprentice.agents.cre_agents.environment import TextField
from cre import CREFunc
import numpy as np

register_func = new_register_decorator("func", full_descr="CREFunc")
register_all_funcs = new_register_all("func", types=[CREFunc], full_descr="CREFunc")

@CREFunc(signature=boolean(string,string),
    shorthand = '{0} == {1}',
    commutes=True)
def Equals(a, b):
    return a == b

@CREFunc(signature=f8(f8,f8),
    shorthand = '{0} + {1}',
    commutes=True)
def Add(a, b):
    return a + b

@CREFunc(signature=f8(f8,f8))
def AddPositive(a, b):
    if(not (a >= 0 and b >= 0)):
        raise Exception
    return a + b

@CREFunc(signature=f8(f8,f8,f8),
    shorthand = '{0} + {1} + {2}',
    commutes=True)
def Add3(a, b, c):
    return a + b + c

@CREFunc(signature=f8(f8,f8),
    shorthand = '{0} - {1}')
def Subtract(a, b):
    return a - b

@CREFunc(signature=f8(f8,f8),
    shorthand = '{0} * {1}',
    commutes=True)
def Multiply(a, b):
    return a * b

@CREFunc(signature=f8(f8,f8),
    shorthand = '{0} / {1}'
    )
def Divide(a, b):
    return a / b

@CREFunc(signature=f8(f8,f8),
    shorthand = '{0} // {1}')
def FloorDivide(a, b):
    return a // b

# @CREFunc(signature=f8(f8,f8),
#     shorthand = '{0} ** {1}')
# def Power(a, b):
#     return a ** b

@CREFunc(signature=f8(f8,f8),
    shorthand = '{0} % {1}')
def Modulus(a, b):
    return a % b


@CREFunc(signature=f8(f8), shorthand = '{0}^2')
def Square(a):
    return a * a

@CREFunc(signature=f8(f8, f8), shorthand = '{0}^{1}')
def Power(a, b):
    return a ** b

@CREFunc(signature=f8(f8), shorthand = '{0}+1')
def Increment(a):
    return a + 1

@CREFunc(signature=f8(f8), shorthand = '{0}-1')
def Decrement(a):
    return a - 1

@CREFunc(signature=f8(f8), shorthand = 'log2({0})')
def Log2(a):
    return np.log2(a)

@CREFunc(signature=f8(f8), shorthand = 'cos({0})')
def Cos(a):
    return np.cos(a)

@CREFunc(signature=f8(f8), shorthand = 'sin({0})')
def Sin(a):
    return np.sin(a)

@CREFunc(signature=f8(f8),
    shorthand = '{0} % 10')
def Mod10(a):
    return a % 10

@CREFunc(signature=f8(f8),
    shorthand = '{0} // 10')
def Div10(a):
    return a // 10

@CREFunc(signature=string(string),
    shorthand = '{0}')
def Copy(a):
    return a

@CREFunc(signature = string(string,string),
    shorthand = '{0} + {1}', 
    commutes=False)
def Concatenate(a, b):
    return a + b


@CREFunc(signature=f8(f8), shorthand = '{0}/2')
def Half(a):
    return a / 2

@CREFunc(signature=f8(f8), shorthand = '{0}*2')
def Double(a):
    return a * 2

@CREFunc(signature=f8(f8), shorthand = 'OnesDigit({0})')
def OnesDigit(a):
    return a % 10

@CREFunc(signature=f8(f8), shorthand = 'TensDigit({0})')
def TensDigit(a):
    return (a // 10) % 10


### Special Functions for Fractions 
###  --typically can be replaced with Multiply

@CREFunc(signature=f8(f8,f8,f8),
    shorthand='({0} / {1}) * {2}')
def ConvertNumerator(a, b, c):
    return (a / b) * c

@CREFunc(signature=f8(TextField, TextField),
    shorthand='Cross({0} * {1})')
def CrossMultiply(a, b):
    if('den' in a.id and 'den' in b.id):
        raise ValueError()
    if('num' in a.id and 'num' in b.id):
        raise ValueError()
    return (float(a.value) * float(b.value))

@CREFunc(signature=f8(TextField, TextField),
    shorthand='Across({0} * {1})')
def AcrossMultiply(a, b):
    if('den' in a.id and 'den' not in b.id):
        raise ValueError()
    if('num' in a.id and 'num' not in b.id):
        raise ValueError()

    return (float(a.value) * float(b.value))


##### Define all CREFuncs above this line #####

register_all_funcs()

