from numba.types import f8, string, boolean
from apprentice.agents.cre_agents.extending import registries, new_register_decorator, new_register_all
from cre import Op

register_op = new_register_decorator("op", full_descr="Op")
register_all_ops = new_register_all("op", types=[Op], full_descr="Op")


@Op(signature=boolean(string,string),
    shorthand = '({0} == {1})',
    commutes=True)
def Equals(a, b):
    return a == b

@Op(signature=f8(f8,f8),
    shorthand = '({0} + {1})',
    commutes=True)
def Add(a, b):
    return a + b

@Op(signature=f8(f8,f8,f8),
    shorthand = '({0} + {1} + {2})',
    commutes=True)
def Add3(a, b, c):
    return a + b + c

@Op(signature=f8(f8,f8),
    shorthand = '({0} - {1})')
def Subtract(a, b):
    return a - b

@Op(signature=f8(f8,f8),
    shorthand = '({0} * {1})',
    commutes=True)
def Multiply(a, b):
    return a * b

def denom_not_zero(a,b):
    return b != 0

@Op(signature=f8(f8,f8),
    shorthand = '({0} / {1})', 
    check=denom_not_zero,
    )
def Divide(a, b):
    return a / b

@Op(signature=f8(f8,f8),
    check=denom_not_zero,
    shorthand = '({0} // {1})')
def FloorDivide(a, b):
    return a // b

@Op(signature=f8(f8,f8),
    shorthand = '({0} ** {1})')
def Power(a, b):
    return a ** b

@Op(signature=f8(f8,f8),
    shorthand = '({0} % {1})')
def Modulus(a, b):
    return a % b

@Op(signature=f8(f8),
    shorthand = '({0} % 10)')
def Mod10(a):
    return a % 10

@Op(signature=f8(f8),
    shorthand = '({0} // 10)')
def Div10(a):
    return a // 10

@Op(signature=string(string),
    shorthand = '{0}')
def Copy(a):
    return a

@Op(signature = string(string,string),
    shorthand = '({0} + {1})', 
    commutes=False)
def Concatenate(a, b):
    return a + b

def b_not_zero(a,b,c):
    return b != 0

@Op(signature=f8(f8,f8,f8),
    check=b_not_zero,
    shorthand='(({0} / {1}) * {2})')
def ConvertNumerator(a, b, c):
    return (a / b) * c


# --------------
# : Cast float/str

def check_cast_float(a):
    if(a == ""): return False
    try:
        float(a)
    except:
        return False
    return True

@Op(shorthand = '{0}', check=check_cast_float)
def CastFloat(a):
    return float(a)

def check_cast_str(a):
    try:
        str(a)
    except:
        return False
    return True

@Op(shorthand = '{0}', check=check_cast_str)
def CastStr(a):
    return str(a)



##### Define all Ops above this line #####

register_all_ops()

