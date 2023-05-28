from apprentice.agents.cre_agents.how.nlp.nlp_sc_planner import NLPSetChaining
from numba.types import f8
from cre.default_funcs import Add, Subtract, Multiply, Divide, CastFloat
from cre import CREFunc, define_fact

import numpy as np

# ------------------------------------
# : Definition of Ops

Add_f8 = Add(f8,f8)
Subtract_f8 = Subtract(f8,f8)
Multiply_f8 = Multiply(f8,f8)
Divide_f8 = Divide(f8,f8)

@CREFunc(signature=f8(f8), shorthand = '({0}/2)')
def Half(a):
    return a / 2

@CREFunc(signature=f8(f8), shorthand = '({0}*2)')
def Double(a):
    return a * 2

@CREFunc(signature=f8(f8), shorthand = 'Ones({0})')
def Ones(a):
    return a % 10

@CREFunc(signature=f8(f8), shorthand = 'Tens({0})')
def Tens(a):
    return (a // 10) % 10

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

func_dictionary = {
    "sum" : Add_f8,
    "add" : Add_f8,
    "plus" : Add_f8,

    "product" : Multiply_f8,
    "multiply" : Multiply_f8,
    "times" : Multiply_f8,

    "subtract" : Subtract_f8,
    "minus" : Subtract_f8,
    "difference" : Subtract_f8,

    "divide" : Divide_f8,
    "quotient" : Divide_f8,
    "ratio" : Divide_f8,
    "proportion" : Divide_f8,

    "half" : Half,

    "ones" : Ones,

    "tens" : Tens,

    "square" : Square,

    "power" : Power,

    "double" : Double,
    "twice" : Double,

    "increment" : Increment,
    "decrement" : Decrement,
    "log2" : Log2,

    "sin" : Sin,
    "sine" : Sin,

    "cos" : Cos,
    "cosine" : Cos,
}

special_patterns = {
  r"(\S+)\sdivided\sby\s(\S+)" : Divide_f8,
  r"(\S+)\sover\s(\S+)" : Divide_f8,
  r"(\S+)\stimes\s(\S+)" : Multiply_f8,
  r"(\S+)\sminus\s(\S+)" : Subtract_f8,
  r"(\S+)\splus\s(\S+)" : Add_f8,


  # r"\(*\s*1\s*\/\s*2\s*\)*" : Half,

  # r"(\S+)\s*\/\s*(\S+)" : Divide_f8,
  # r"(\S+)\s*\*\s*(\S+)" : Multiply_f8,
  # r"(\S+)\s*-\s*(\S+)" : Subtract_f8,
  # r"(\S+)\s*\+\s*(\S+)" : Add_f8,
  
  r"ones\s(digit|place)" : Ones,
  r"ones'\s(digit|place)" : Ones,
  r"one's\s(digit|place)" : Ones,
  r"last\s(digit|place)" : Ones,
  r"final\s(digit|place)" : Ones,

  r"tens\s(digit|place)" : Tens,
  r"tens'\s(digit|place)" : Tens,
  r"ten's\s(digit|place)" : Tens,
}


IE = define_fact("IE", {
    "id" : str,
    "value" : {"type" : str, "visible" : True, "semantic" : True,
                'conversions' : {float : CastFloat}},
})
IE._fact_proxy.__str__ = lambda x: f"{x.value}@{x.id})"

# ------------------------------------
# : Config

verbosity=0
# verbosity=1
do_all_levels=True
use_facts=False


# ------------------------------------
# : Testing Funcs
planner = NLPSetChaining(
    func_dictionary=func_dictionary, special_patterns=special_patterns,
    fact_types=(IE,), verbosity=verbosity, float_to_str=False)    

def float_if_numeric(x):
    if(isinstance(x,int)):
        return float(x)
    return x

# def count_solutions(expls):
#     l1, l2 = 0,0
#     if(expls is not None):
#         expl_list = list(expls)
#         expls_set = set([str(expl) for expl in expl_list])
#         l1, l2 = len(expl_list), len(expls_set)
#     return l1,l2

class SolutionsProfile(object):
    def __init__(self, return_phase, goal, hint):
        self.data = {}
        self.return_phase = return_phase
        self.goal = goal
        self.hint = hint
        self.is_null = False

    def add_phase_data(self, phase, expls, planner):
        # l1, l2 = 0,0
        # if(expls is not None):
        expl_list = list(expls)
        expls_set = set([str(expl) for expl in expl_list])
            # l1, l2 = len(expl_list), len(expls_set)
        # return l1,l2

        # n_sol, n_unq_sol, **kwargs
        self.data[phase] = {"n_sol":len(expl_list),
                            "n_unq_sol":len(expls_set),
                            "expls" : expl_list,
                            "unq_expls" : expls_set,
                            "num_fwd_infs" : planner.num_forward_inferences}
    def add_null_phase_data(self, phase=1, total_fail=True):
        self.data[phase] = None
        self.is_null = total_fail

    def add_corr_strs(self, corr_strs):
        self.corr_strs = corr_strs

        phases = list(self.data.keys())#[self.return_phase] if not do_all_levels else list(range(self.return_phase,6))
        for p in phases:
            has_correct = False
            num_incorrect = 0
            for expl in self.get_unq_expls(p):
                if(str(expl) in corr_strs):
                    has_correct = True
                else:
                    num_incorrect += 1
            data = self.data.get(p, {})
            if(data is not None):
                data.update({
                    "num_incorrect" : num_incorrect,
                    "has_correct" : has_correct
                })
                self.data[p] = data


    @property
    def has_correct(self):
        return self.data[self.return_phase]['has_correct']

    @property
    def num_incorrect(self):
        return self.data[self.return_phase]['num_incorrect']

    @property
    def num_forward_inferences(self):
        return self.data[self.return_phase]['num_fwd_infs']


    def summary(self, all_phases=True, verbosity=0):
        s = f'-- Goal {self.goal} : "{self.hint}"'
        if(self.is_null):
            s += "-> No solutions." 
        else:
            for phase, data in self.data.items():
                if(data is not None):
                    hc = data.get('has_correct',False)
                    if('num_incorrect' in data):
                        ni = data['num_incorrect']
                        cr = f"{'✔' if hc else '✘'} ({1 if hc else 0}/{ni+hc}) : "
                    else:
                        cr = {"??(?/?)"}
                    s += f"\n{cr} {data['n_sol']}, {data['n_unq_sol']} unique solutions at phase {phase}.{' *' if self.return_phase==phase else ''}"
                else:
                    s += f"No Solution at phase {phase}."
                if(verbosity > 0):
                    for expl in self.expls:
                        s += f"\n{expl}"

                if(not all_phases): break
        return s

    def get_expls(self, phase=None):
        if(phase is None): phase = self.return_phase
        return self.data[phase].get("expls",None)

    @property
    def expls(self):
        return self.get_expls()

    def get_unq_expls(self, phase=None):
        if(phase is None): phase = self.return_phase
        data = self.data.get(phase,{})
        return data.get("unq_expls",None) if data is not None else []

    @property
    def unq_expls(self):
        return self.get_unq_expls()

    # @property
    # def is_null(self):
    #     return self.data.get(1, None) is not None

    def __str__(self):
        return self.summary(all_phases=verbosity > 0, verbosity=verbosity)



def do_search(values, goal, hint, all_levels=None, search_depth=3, display_parse=False):
    all_levels = do_all_levels if all_levels is None else all_levels 
    planner.clear()
    planner.t2p_parser.display_parse = display_parse

    planner.state = []
    for i, value in enumerate(values):
        v = IE(str(i), str(value)) if use_facts else float_if_numeric(value)
        planner.declare(v)
        planner.state.append(v)


    expls = planner._search_for_explanations(float_if_numeric(goal), hint, search_depth=search_depth, float_to_str=False)

    profile = SolutionsProfile(planner.return_phase, goal, hint)
    profile.policy = planner.policy

    if(expls is None or len(expls) == 0):
        profile.add_null_phase_data()
        profile.return_phase = 1
        return profile

    # l1, l2 = count_solutions(expls)

    
    profile.add_phase_data(planner.return_phase, expls, planner)

    # print(f"\n-- Goal {goal} --")
    # print(f"{l1}, {l2} unique, solutions at phase {planner.return_phase}. *")
    # if(verbosity > 0):
    #     for expl in expls:
    #         print(expl)
    planner.t2p_parser.display_parse = False
    if(not all_levels): return profile
    for i in range(planner.return_phase+1, 6):
        # print("RETURN PHASE ", i)
        expls = planner._search_for_explanations(float_if_numeric(goal), hint,
                   min_phase=i, search_depth=search_depth)


        # l1, l2 = count_solutions(expls)
        # print("add phase", i, planner.return_phase, len(expls) if expls else None)
        if(expls is None or len(expls) == 0):
            
            profile.add_null_phase_data(i, False)
        else:
            profile.add_phase_data(i, expls, planner)


        # print(f"{l1}, {l2} unique, solutions at phase {planner.return_phase}.")
        # if(verbosity > 0):
        #     for expl in expls:
        #         print(expl)
    return profile

def search_summary(values,goal, hint, *args, **kwargs):
    p = do_search(values, goal, hint, *args, **kwargs)
    print(p.summary())
    return p

# ------------------------------------
# : Examples

if __name__ == "__main__":

    print("**TEST**")
    p = search_summary([3, 4, 2, 5], 23, "Using foil and multiply the inside and outside values (4 times 2x, which is 8x) and (3x times 5 which is 15x) and then adding these together we get 23x so the value for b is 23.")

    raise ValueError()
    p = search_summary([9,3,6,2], 2.5, "Divide 9 times 3 by 6, then subtract 2.")
    print(p)
    # raise ValueError()
    p = search_summary([3,8,2], -4,"Divide 3 minus 8 and twice 2. Then increment.")

    
    print("\n**  What is the Area of the shaded Region? **")
    p = search_summary([20,12,10,6], 90,"The area of the larger is half of 20 times 12. The area of the smaller triangle is half of 6 times 10. Subtract the area of the smaller triangle from the area of the larger triangle.")

    print("\n** Find the slope of the line that passes through (5, 4) and (7, 8). **")
    search_summary([5,4,7,8], 2, "The slope is the quotient of the difference between the y-values 8 and 4, and the difference between the x-values 7 and 5.")

    # search_summary([5,4,7,8], 2, "Subtract the first y-coordinate 4 from the second y-coordinate 8, then subtract the first x-coordinate 5 from the second x-coordinate 7, and then divide the difference of the y-coordinates (4) by the difference of the x-coordinates (2).")
    # p = search_summary([5,4,7,8], 2, "Subtract 8 by 4 and Subtract 7 by 5. Divide the first sum by the second sum.", all_levels=False)
    p = search_summary([10,2,3], 8, "PEMDAS dictates that division takes place before addition. So, 10 should be divided by 2 first. Then, the quotient should be added to 3.", all_levels=False)
    print(p.policy)
    # return
    # raise ValueError()
    print("\n** Expand the Polynomial (3x+4)(2x+5) into ax2+bx+c **")
    search_summary([3, 4, 2, 5], 6, "Multiply 3 and 2")
    search_summary([3, 4, 2, 5, 6], 23, "Take the sum of the product of 3 and 5 and the product of 4 and 2")
    search_summary([3, 4, 2, 5, 6, 23], 20, "Multiply 4 and 5")

    print("\n** Convert the fractions to simplify 3/4 + 2/3 **")
    search_summary([3, 4, 2, 3], 12, "Multiply 4 and 3")
    search_summary([3, 4, 2, 3, 5, 12], 12, "Copy 12")
    search_summary([3, 4, 2, 3, 5, 12, 12], 9, "Multiply 3 and 3")
    search_summary([3, 4, 2, 3, 5, 12, 12, 9], 8, "Multiply 2 and 4")
    search_summary([3, 4, 2, 3, 5, 12, 12, 9, 8], 12, "Copy the converted denominator 12")
    search_summary([3, 4, 2, 3, 5, 12, 12, 9, 8, 12], 17, "Add the left converted numerator 9 and the right converted numerator 8")

    print("\n** What is the Area of the shaded Region? **")
    p = search_summary([20,12,10,6], 90,"Subtract the area of the larger triangle half of 20 times 12 from the area of the smaller triangle which is half of 10 times 6")
    for expl in p.expls:
        print(expl)


    #Multi-Column Addition Examples... not great because of 'ones digit', 'tens digit'

    # do_search([8, 4, 2, 6, 7], 9,"Add 2 and 7")
    # do_search([8, 4, 2, 6, 7, 9], 0, "Add 6 and 4 and take the ones digit")
    # do_search([8, 4, 2, 6, 7, 9, 0], 1, "Add 6 and 4 and take the tens digit")
    # do_search([8, 4, 2, 6, 7, 9, 0, 1], 9, "Add 1 and 8 and take the ones digit")

    # do_search([8, 4, 2, 6, 7], 9,"Add 2 and 7")
    # do_search([8, 4, 2, 6, 7, 9], 0, "Take the ones digit of 6 plus 4.")
    # do_search([8, 4, 2, 6, 7, 9, 0], 1, "Take the tens digit of 6 plus 4.")
    # do_search([8, 4, 2, 6, 7, 9, 0, 1], 9, "Take the ones digit of 1 plus 8.")


    print("** The radius of a circle is 6 centimeters. What is the area of a sector bounded by a 135° arc? **")
    search_summary([135, 360, 6], 13.5, "Divide 135 by 360 and multiply it by the square of 6.")

    print("\n** Find the slope of the line that passes through (5, 4) and (7, 8). **")
    search_summary([5,4,7,8], 2, "Divide the change in y 8 minus 4 by the change in x 7 minus 5. ")

    print("\n** What is the length of side x if the two triangles are similar **")
    search_summary([1,8,2], 4, "Multiply 1 by 8 divided by 2.")

    # planner = NLPSetChaining(func_dictionary=func_dictionary)    
    # for n in range(5):
    #     planner.declare(float(n))

    # print("IDENTIFIERS", planner.identifiers)

    # text = "Multiply the sum of 1 and 2 with the product of 3 and 4"

    # expls = planner._search_for_explanations(360, text, search_depth=2)

    # print(f"Ended at Phase {planner.return_phase}.")
    # for expl in expls:
    #     print(expl)





    # Find a

