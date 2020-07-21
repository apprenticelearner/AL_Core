from numbert.operator import BaseOperator

from numba import njit

textfield = {
    "id" : "string",
    "dom_class" : "string",
    "offsetParent" : "string",
    "value" : "string",
    "contentEditable" : "number",
    "above" : "string",
    "below" : "string",
    "to_right" : "string",
    "to_left" : "string",
}
component = {
    "id" : "string",
    "dom_class" : "string",
    "offsetParent" : "string",
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


numbalizer.register_specification("TextField",textfield)
numbalizer.register_specification("Component",component)
numbalizer.register_specification("Symbol",symbol)


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
        # 	return str(int(x))
        return str(x)

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


