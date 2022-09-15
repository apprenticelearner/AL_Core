from numbert.operator import BaseOperator
import math
from numba import njit
from .representation import numbalizer

textfield = {
    "id" : "string",
    # "dom_class" : "string",
    # "offsetParent" : "string",
    "value" : "string",
    "contentEditable" : "number",
    # "above" : "string",
    # "below" : "string",
    # "to_right" : "string",
    # "to_left" : "string",
}

button = {
    "id": "string",
    "dom_class":"string",
    "label":"string",
    "above" : "string",
    "below" : "string",
    "to_right" : "string",
    "to_left" : "string",
}

checkbox = {
    "id": "string",
    "dom_class":"string",
    "label":"string",
    "above" : "string",
    "below" : "string",
    "to_right" : "string",
    "to_left" : "string",
    "groupName":"string",  
}


component = {
    "id" : "string",
    "dom_class" : "string",
    # "offsetParent" : "string",
    "above" : "string",
    "below" : "string",
    "to_right" : "string",
    "to_left" : "string",
}

symbol = {
    "id" : "string",
    "value" : "string",
    "filled" : "number",
    "above" : "string",
    "below" : "string",
    "to_right" : "string",
    "to_left" : "string",
}

overlay_button = {
    "id" : "string",
}


numbalizer.register_specification("TextField",textfield)
numbalizer.register_specification("TextArea",textfield)
numbalizer.register_specification("Button", button)
numbalizer.register_specification("Checkbox", checkbox)
numbalizer.register_specification("RadioButton", checkbox)
numbalizer.register_specification("Component",component)
numbalizer.register_specification("Symbol",symbol)
numbalizer.register_specification("OverlayButton",overlay_button)


@njit(cache=True)
def is_prime(n):
    if n % 2 == 0 and n > 2:
        return False
    for i in range(3, int(math.sqrt(n)) + 1, 2):
        if n % i == 0:
            return False
    return True


class SquaresOfPrimes(BaseOperator):
    signature = 'float(float)'

    def condition(x):
        out = is_prime(x)
        return out

    def forward(x):
        return x**2


class EvenPowersOfPrimes(BaseOperator):
    signature = 'float(float,float)'

    def condition(x, y):
        b = is_prime(x)
        a = (y % 2 == 0) and (y > 0) and (y == int(y))
        return a and b

    def forward(x, y):
        return x**y


class Add(BaseOperator):
    commutes = True
    signature = 'float(float,float)'

    def forward(x, y):
        return x + y


class AddOne(BaseOperator):
    commutes = True
    signature = 'float(float)'

    def forward(x):
        return x + 1


class Subtract(BaseOperator):
    commutes = False
    signature = 'float(float,float)'

    def forward(x, y):
        return x - y


class Multiply(BaseOperator):
    commutes = True
    signature = 'float(float,float)'

    def forward(x, y):
        return x * y


class Divide(BaseOperator):
    commutes = False
    signature = 'float(float,float)'

    def condition(x, y):
        return y != 0

    def forward(x, y):
        return x / y


class Equals(BaseOperator):
    commutes = False
    signature = 'float(float,float)'

    def forward(x, y):
        return x == y


class Add3(BaseOperator):
    commutes = True
    signature = 'float(float,float,float)'

    def forward(x, y, z):
        return x + y + z


class Mod10(BaseOperator):
    commutes = True
    signature = 'float(float)'

    def forward(x):
        return x % 10


class Div10(BaseOperator):
    commutes = True
    signature = 'float(float)'

    def forward(x):
        return x // 10


class Concatenate(BaseOperator):
    signature = 'string(string,string)'

    def forward(x, y):
        return x + y

class Append25(BaseOperator):
    signature = 'string(string)'

    def forward(x):
        return x + "25"


class StrToFloat(BaseOperator):
    signature = 'float(string)'
    muted_exceptions = [ValueError]
    nopython = False

    def forward(x):
        return float(x)


class FloatToStr(BaseOperator):
    signature = 'string(float)'
    muted_exceptions = [ValueError]
    nopython = False

    def forward(x):
        # if(int(x) == x):
        #   return str(int(x))
        return str(x)

class RipStrValue(BaseOperator):
    signature = 'string(TextField)'
    template = "{}.v"
    nopython=False
    muted_exceptions = [ValueError]
    def forward(x):
        return str(x.value)

class RipFloatValue(BaseOperator):
    signature = 'float(TextField)'
    template = "{}.v"
    nopython=False
    muted_exceptions = [ValueError]
    def forward(x): 
        return float(x.value)

class RipFloatValueSymbol(BaseOperator):
    signature = 'float(Symbol)'
    template = "{}.v"
    nopython=False
    muted_exceptions = [ValueError]
    def forward(x): 
        return float(x.value)


class Numerator_Multiply(BaseOperator):
    signature = 'float(TextField,TextField)'
    template = "Numerator_Multiply({}.v,{}.v)"
    nopython=False
    muted_exceptions = [ValueError]
    def condition(x,y): 
        return x.id.split(".R")[1] == y.id.split(".R")[1]
    def forward(x,y): 
        return float(x.value) * float(y.value)

class Cross_Multiply(BaseOperator):
    signature = 'float(TextField,TextField)'
    template = "Cross_Multiply({}.v,{}.v)"
    nopython=False
    muted_exceptions = [ValueError]
    def condition(x,y): 
        return x.id.split(".R")[1] != y.id.split(".R")[1]
    def forward(x,y): 
        return float(x.value) * float(y.value)

class Diamond(BaseOperator):
    signature = 'float(float,float)'
    nopython=False
    muted_exceptions = [ValueError, OverflowError]
    def forward(base, exp):
        if not(0 <= exp <= 5 and 0 <= base <= 25):
            return -1
        return int(str(int(base ** exp))[::-1].lstrip("0"))

class Sun(BaseOperator):
    signature = 'float(float,float,float,float)'
    nopython=False
    muted_exceptions = [ValueError]
    def forward(a, b, c, d):
        if b == 0 or d == 0:
            return -1
        l = int(a) // int(b)
        r = int(c) // int(d)
        if r == 0:
            return -1
        return l // r

class Command(BaseOperator):
    signature = 'float(float,float,float)'
    nopython=False
    muted_exceptions = [ValueError]
    def forward(a, b, c):
        l = int(a)
        r = int(b) * int(c)
        if r == 0:
            return -1
        return l // r

class Bullseye(BaseOperator):
    signature = 'float(float,float)'
    nopython=False
    muted_exceptions = [ValueError]
    def forward(a, b):
        if not(abs(int(a) - a) < 1e-6 and abs(int(b) - b) < 1e-6 and 0 < a < 6):
            return -1
        l = int(b) // int(a)
        r = int(b) ** int(a)
        return l + r

class Pow(BaseOperator):
    commutes = False
    signature = 'float(float,float)'
    # muted_exceptions = [OverflowError]

    def condition(x,y): 
        return x < 10 and y < 10
        # return not(x > 15 and y > 15)

    def forward(x, y):
        return x ** y

class Inverse(BaseOperator):
    signature = 'string(float)'
    nopython = False

    # signature = 'float(TextField)'
    #     return float(x.value)
    def forward(x):
        s = str(x)
        return s[::-1]

        # x = int(x / 10) + ((x % 10) * 10)
        # return x