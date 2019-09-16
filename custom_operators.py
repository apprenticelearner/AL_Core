from planners.fo_planner import Operator
# from planners.VectorizedPlanner import BaseOperator

'''							USAGE INSTRUCTIONS
FO Operator Structure: Operator(<header>, [<conditions>...], [<effects>...])

	<header> : ('<name>', '?<var_1>', ... , '?<var_n>')
			example : ('Add', '?x', '?y')
	<conditions> : [(('<attribute>', '?<var_1>'),'?<value_1>'), ... ,
					 (<func>, '?<value_1>', ...), ...
				   ]
			example : [ (('value', '?x'), '?xv'),
	                  (('value', '?y'), '?yv'),
	                  (lambda x, y: x <= y, '?x', '?y')
	                  ]
	<effects> : [(<out_attribute>,
				 	('<name>', ('<in_attribute1>', '?<var_1>'), ...),
				  	(<func>, '?<value_1>', ...)
			     ), ...]
			example :[(('value', ('Add', ('value', '?x'), ('value', '?y'))),
	                     (int_float_add, '?xv', '?yv'))])
	Full Example:
	def int_float_add(x, y):
	    z = float(x) + float(y)
	    if z.is_integer():
	        z = int(z)
	    return str(z)

	add_rule = Operator(('Add', '?x', '?y'),
			            [(('value', '?x'), '?xv'),
			             (('value', '?y'), '?yv'),
			             (lambda x, y: x <= y, '?x', '?y')
			             ],
			            [(('value', ('Add', ('value', '?x'), ('value', '?y'))),
			              (int_float_add, '?xv', '?yv'))])

	Note: You should explicitly register your operators so you can
			 refer to them in your training.json, otherwise the name will
			 be the same as the local variable
			example: Operator.register("Add")

vvvvvvvvvvvvvvvvvvvv WRITE YOUR OPERATORS BELOW vvvvvvvvvvvvvvvvvvvvvvv '''


def int_float_equals(x, y):
    return float(x) == float(y)


equals = Operator(('Equals', '?x', '?y'),
               [(('value', '?x'), '?xv'),
                (('value', '?y'), '?yv'),
                (lambda x, y: x <= y, '?x', '?y')
                ],
               [(('value', ('Equals', ('value', '?x'), ('value', '?y'))),
                 (int_float_equals, '?xv', '?yv'))])


def int_float_add(x, y):
    z = float(x) + float(y)
    if z.is_integer():
        z = int(z)
        return str(z)


add = Operator(('Add', '?x', '?y'),
               [(('value', '?x'), '?xv'),
                (('value', '?y'), '?yv'),
                (lambda x, y: x <= y, '?x', '?y')
                ],
               [(('value', ('Add', ('value', '?x'), ('value', '?y'))),
                 (int_float_add, '?xv', '?yv'))])


def int_float_substract(x, y):
    z = float(x) - float(y)
    if z.is_integer():
        z = int(z)
        return str(z)


substract = Operator(('Substract', '?x', '?y'),
               [(('value', '?x'), '?xv'),
                (('value', '?y'), '?yv'),
                (lambda x, y: x <= y, '?x', '?y')
                ],
               [(('value', ('Substract', ('value', '?x'), ('value', '?y'))),
                 (int_float_substract, '?xv', '?yv'))])


def int_float_multiply(x, y):
    z = float(x) * float(y)
    if z.is_integer():
        z = int(z)
        return str(z)


multiply = Operator(('Multiply', '?x', '?y'),
               [(('value', '?x'), '?xv'),
                (('value', '?y'), '?yv'),
                (lambda x, y: x <= y, '?x', '?y')
                ],
               [(('value', ('Multiply', ('value', '?x'), ('value', '?y'))),
                 (int_float_multiply, '?xv', '?yv'))])


def int_float_divide(x, y):
    z = float(x) / float(y) # Do we need to check for 0 denominator.
    if z.is_integer():
        z = int(z)
        return str(z)


divide = Operator(('Divide', '?x', '?y'),
               [(('value', '?x'), '?xv'),
                (('value', '?y'), '?yv'),
                (lambda x, y: x <= y, '?x', '?y')
                ],
               [(('value', ('Divide', ('value', '?x'), ('value', '?y'))),
                 (int_float_divide, '?xv', '?yv'))])


# ^^^^^^^^^^^^^^ DEFINE ALL YOUR OPERATORS ABOVE THIS LINE ^^^^^^^^^^^^^^^^
for name,op in locals().copy().items():
  if(isinstance(op, Operator)):
    Operator.register(name,op)
