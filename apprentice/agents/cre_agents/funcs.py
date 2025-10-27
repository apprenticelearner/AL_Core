from numba.types import f8, string, boolean
from apprentice.agents.cre_agents.extending import registries, new_register_decorator, new_register_all
from apprentice.agents.cre_agents.environment import TextField
from cre import CREFunc
import numpy as np

register_func = new_register_decorator("func", full_descr="CREFunc")
register_all_funcs = new_register_all("func", types=[CREFunc], full_descr="CREFunc")

@CREFunc(signature=string(string), shorthand='exp_mult_pow({0})')
def MultiplyExponents(s):
    i_close = s.index(')')       # position of ')'
    j_open  = s.index('^{', i_close + 1)  # must be right after ')'
    k_end   = s.index('}', j_open + 2)
    e2 = s[j_open + 2 : k_end]

    inner = s[1 : i_close]       # "base^{e1}"
    i_pow  = inner.index('^{')
    k1_end = inner.index('}', i_pow + 2)
    base = inner[:i_pow]
    e1   = inner[i_pow + 2 : k1_end]

    out = base + "^{" + e1 + " \\cdot " + e2 + "}"
    return out

@CREFunc(signature=string(string), shorthand='exp_power_rule({0})')
def PowerRule(s):
    s = s.rsplit("/", 1)[-1].strip()   # e.g., "675^{6 \cdot 6}"

    i_pow = s.index('^{')              # start of exponent
    j_open = i_pow + 2                 # first char inside '{'
    k_end = s.index('}', j_open)       # closing '}'

    base = s[:i_pow]                   # "675"
    exp_str = s[j_open:k_end]          # "6 \cdot 6"

    a_str, b_str = exp_str.split(r'\cdot')  # exactly two factors
    a = int(a_str.strip())
    b = int(b_str.strip())

    out = base + "^{" + str(a * b) + "}"
    return out

# --- Product rule: a^{m} * a^{n}  ->  a^{m + n}
@CREFunc(signature=string(string), shorthand='exp_product_rule({0})')
def ProductRule(s: str) -> str:
    # make robust like PowerRule
    s = s.rsplit("/", 1)[-1].strip()

    # split on either "\cdot" or "*"
    if r'\cdot' in s:
        left, right = [t.strip() for t in s.split(r'\cdot', 1)]
    else:
        left, right = [t.strip() for t in s.split('*', 1)]

    # Parse left: a^{m}
    iL = left.index('^{'); jL = iL + 2; kL = left.index('}', jL)
    baseL = left[:iL]
    m = left[jL:kL].strip()

    # Parse right: a^{n}
    iR = right.index('^{'); jR = iR + 2; kR = right.index('}', jR)
    baseR = right[:iR]
    n = right[jR:kR].strip()

    # optional safety
    if baseL != baseR:
        raise ValueError("ProductRule expects matching bases.")

    return f"{baseL}^{{{m} + {n}}}"


@CREFunc(signature=string(string), shorthand='exp_product_simplify({0})')
def SimplifyProduct(s: str) -> str:
    s = s.rsplit("/", 1)[-1].strip()
    i = s.index('^{'); j = i + 2; k = s.index('}', j)
    base = s[:i]
    m_str, n_str = [t.strip() for t in s[j:k].split('+', 1)]
    val = int(m_str) + int(n_str)
    return f"{base}^{{{val}}}"



# --- Quotient rule: a^{m} / a^{n}  ->  a^{m - n}
@CREFunc(signature=string(string), shorthand='exp_quotient_rule({0})')
def QuotientRule(s: str) -> str:
    s = s.rsplit("/", 1)[-1].strip()
    left, right = [t.strip() for t in s.split('/', 1)]

    iL = left.index('^{'); jL = iL + 2; kL = left.index('}', jL)
    baseL = left[:iL]
    m = left[jL:kL].strip()

    iR = right.index('^{'); jR = iR + 2; kR = right.index('}', jR)
    baseR = right[:iR]
    n = right[jR:kR].strip()

    if baseL != baseR:
        raise ValueError("QuotientRule expects matching bases.")

    return f"{baseL}^{{{m} - {n}}}"


@CREFunc(signature=string(string), shorthand='exp_quotient_simplify({0})')
def SimplifyQuotient(s: str) -> str:
    s = s.rsplit("/", 1)[-1].strip()
    i = s.index('^{'); j = i + 2; k = s.index('}', j)
    base = s[:i]
    m_str, n_str = [t.strip() for t in s[j:k].split('-', 1)]
    val = int(m_str) - int(n_str)
    return f"{base}^{{{val}}}"


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

# === Prior Knowledge & Cross-Domain Functions ===

@CREFunc(signature=f8(TextField), shorthand='Num({0})')
def Num(tf):
    """Convert TextField value to a float."""
    return float(str(tf.value).replace(',', '').strip())

@CREFunc(signature=string(TextField), shorthand='Str({0})')
def StrTF(tf):
    """Extract string from a TextField."""
    return str(tf.value)

@CREFunc(signature=boolean(TextField), shorthand='IsDen({0})')
def IsDen(tf):
    """Return True if this TextField looks like a denominator."""
    return 'den' in tf.id

@CREFunc(signature=boolean(TextField), shorthand='IsNum({0})')
def IsNum(tf):
    """Return True if this TextField looks like a numerator."""
    return 'num' in tf.id

@CREFunc(signature=boolean(string, string), shorthand='lower({0}) == lower({1})', commutes=True)
def EqualsIgnoreCase(a, b):
    """Case-insensitive equality for strings."""
    return a.lower() == b.lower()

@CREFunc(signature=f8(string), shorthand='parse({0})')
def ParseFloat(s):
    """Parse numeric string into float."""
    return float(s.replace(',', '').strip())

@CREFunc(signature=string(string, string), shorthand='join({0},{1})', commutes=False)
def JoinWithSpace(a, b):
    """Join two strings with a space."""
    return f"{a} {b}"

@CREFunc(signature=f8(f8), shorthand='sq({0})')
def Sq(a):
    """Square a number."""
    return a * a

@CREFunc(signature=f8(f8, f8), shorthand='sq({0}) + {1}')
def SqPlus(a, b):
    """Square a number and add another."""
    return (a * a) + b

@CREFunc(signature=f8(f8, f8), shorthand='Reuse({0},{1})', commutes=True)
def ReusePrior(a, b):
    """Prefer previously known value a; otherwise use b."""
    return a if a != 0 else b

@CREFunc(signature=f8(f8), shorthand='Recall({0})')
def RecallValue(a):
    """Surface a previously computed value."""
    return a

@CREFunc(signature=f8(f8, f8), shorthand='AvgPrior({0},{1})', commutes=True)
def AveragePrior(a, b):
    """Combine earlier results (simple average)."""
    return (a + b) / 2

##### Define all CREFuncs above this line #####

register_all_funcs()

