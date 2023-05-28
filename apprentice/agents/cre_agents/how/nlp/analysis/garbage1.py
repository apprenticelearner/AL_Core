import pandas as pd
import numpy as np
import sys
import re

'''
CYU : Check your understanding
ANS : The participant's answer to the question
COR : 1 for correct 0 for incorrrect
CON : The participant's conceptual hint
OP :  The participant's grounded operational hint

ALL_VALS: The grounded operational hint includes all argument and constant values. (e.g. "Add 3 and 4". Not "Add the numerators")
ALL_OPS: All steps are stated as explicit executable mathematical operations. And the stated operations are functionally correct. (e.g. "To find x divide 9 from 18",
          as opposed to "Solve for x".) 
'''
df = pd.read_csv("NLP_how_turk_data_cleaned.csv",delimiter=',')
print(df.columns)

from apprentice.agents.cre_agents.how.nlp.examples import do_search as _do_search, func_dictionary
do_search = lambda *args : _do_search(*args, all_levels=False)

N_PROBS = 14

corr_strs1 = ["(Divide(Multiply(a, b), c), [100.0, 3.0, 12.0])",
              "(Multiply(Divide(a, b), c), [100.0, 12.0, 3.0])",
              "(Multiply(Divide(a, b), c), [3.0, 12.0, 100.0])",
              "(Divide(a, Divide(b, c)), [100.0, 12.0, 3.0])",
              "(Divide(a, Divide(b, c)), [3.0, 12.0, 100.0])",
              ]
def do_problem1(hint):
    p = do_search([3,12,100], 25, hint)
    p.add_corr_strs(corr_strs1)
    return p

corr_strs2 = ["(Multiply(a, b), [2.0, 3.0])",
              "(Multiply(a, b), [3.0, 2.0])"]    
def do_problem2(hint):
    p = do_search([3,4,2,5], 6, hint)
    p.add_corr_strs(corr_strs2)
    return p

corr_strs3 = ["(Add(Multiply(a, b), Multiply(c, d)), [3.0, 5.0, 4.0, 2.0])",
              "(Add(Multiply(a, b), Multiply(c, d)), [3.0, 5.0, 2.0, 4.0])",
              "(Add(Multiply(a, b), Multiply(c, d)), [5.0, 3.0, 4.0, 2.0])",
              "(Add(Multiply(a, b), Multiply(c, d)), [5.0, 3.0, 2.0, 4.0])",
              "(Add(Multiply(a, b), Multiply(c, d)), [2.0, 4.0, 5.0, 3.0])",
              "(Add(Multiply(a, b), Multiply(c, d)), [2.0, 4.0, 3.0, 5.0])",
              "(Add(Multiply(a, b), Multiply(c, d)), [4.0, 2.0, 3.0, 5.0])",
              "(Add(Multiply(a, b), Multiply(c, d)), [4.0, 2.0, 5.0, 3.0])",
              ]
def do_problem3(hint):
    p = do_search([3,4,2,5], 23, hint)
    p.add_corr_strs(corr_strs3)
    return p

corr_strs4 = ["(Multiply(a, b), [4.0, 5.0])",
              "(Multiply(a, b), [5.0, 4.0])"]
def do_problem4(hint):
    p = do_search([3,4,2,5], 20, hint)
    p.add_corr_strs(corr_strs4)
    return p 

corr_strs5 = ['(Multiply(a, b), [3.0, 4.0])',
              '(Multiply(a, b), [4.0, 3.0])']
def do_problem5(hint):
    p = do_search([3,4,2,3], 12, hint)
    p.add_corr_strs(corr_strs5)
    return p

corr_strs6 = ['(Multiply(a, a), [3.0])',
              '(Multiply(a, b), [3.0, 3.0])']
def do_problem6(hint):
    p = do_search([3,4,2,3,12], 9, hint)
    p.add_corr_strs(corr_strs6)
    return p

corr_strs7 = ['(Add(a, b), [9.0, 8.0])',
              '(Add(a, b), [8.0, 9.0])']
def do_problem7(hint):
    p = do_search([3,4,2,3,12,9,8], 17, hint)
    p.add_corr_strs(corr_strs7)
    return p

corr_strs8 = ['(Subtract(Half(Multiply(a, b)), Half(Multiply(c, d))), [20.0, 12.0, 10.0, 6.0])',
              '(Subtract(Half(Multiply(a, b)), Half(Multiply(c, d))), [20.0, 12.0, 6.0, 10.0])',
              '(Subtract(Half(Multiply(a, b)), Half(Multiply(c, d))), [12.0, 20.0, 10.0, 6.0])',
              '(Subtract(Half(Multiply(a, b)), Half(Multiply(c, d))), [12.0, 20.0, 6.0, 10.0])']
def do_problem8(hint):
    p = do_search([20,12,10,6], 90, hint)
    p.add_corr_strs(corr_strs8)
    return p

corr_strs9 = ['(Multiply(Divide(a, b), Square(c)), [135.0, 360.0, 6.0])',
              '(Multiply(Square(a), Divide(b, c)), [6.0, 135.0, 360.0])',
              '(Multiply(Multiply(a, a), Divide(b, c)), [6.0, 135.0, 360.0])',
              '(Multiply(Divide(a, b), Multiply(c, c)), [135.0, 360.0, 6.0])',
              ]
def do_problem9(hint):
    p = do_search([135,6,360], 13.5, hint)
    p.add_corr_strs(corr_strs9)
    return p

corr_strs10 = ['(Multiply(Divide(a, b), c), [12.0, 3.0, 2.0])', 
               '(Multiply(Divide(a, b), c), [2.0, 3.0, 12.0])',
               '(Divide(Multiply(a, b), c), [12.0, 2.0, 3.0])',
               '(Divide(Multiply(a, b), c), [2.0, 12.0, 3.0])', 
               '(Divide(a, Divide(b, c)), [12.0, 3.0, 2.0])',
               '(Divide(a, Divide(b, c)), [2.0, 3.0, 12.0])'
              ]
def do_problem10(hint):
    p = do_search([2,12,3], 8, hint)
    p.add_corr_strs(corr_strs10)
    return p

corr_strs11 = ['(Divide(a, Add(b, c)), [144.0, 0.2, 1.0])',
               '(Divide(a, b), [144.0, 1.2])']
def do_problem11(hint):
    p = do_search([20,144,1.2,1,.2], 120, hint)
    p.add_corr_strs(corr_strs11)
    return p

corr_strs12 = ['(Add(Divide(a, b), c), [10.0, 2.0, 3.0])']
def do_problem12(hint):
    p =  do_search([3,10,2], 8, hint)
    p.add_corr_strs(corr_strs12)
    return p
    
corr_strs13 = ['(Ones(Add(a, b)), [8.0, 4.0])',
               '(Ones(Add(a, b)), [4.0, 8.0])']
def do_problem13(hint):
    p = do_search([3,8,2,4,7], 2, hint)
    p.add_corr_strs(corr_strs13)
    return p

corr_strs14 = ['(Divide(Subtract(a, b), Subtract(c, d)), [4.0, 8.0, 5.0, 7.0])',
               '(Divide(Subtract(a, b), Subtract(c, d)), [8.0, 4.0, 7.0, 5.0])']
def do_problem14(hint):
    p = do_search([5,4,7,8], 2, hint)
    p.add_corr_strs(corr_strs14)
    return p
    
# Make a data frame for the parsing statistics
from itertools import chain
columns = list(chain(*[[
    f"ALL_VALS_{i}",
    f"ALL_OPS_{i}",

    f"HAS_COR_{i}",
    f"N_INCOR_{i}",
    f"N_INFS_{i}",
    
    f"P4_HAS_COR_{i}",
    f"P4_N_INCOR_{i}",
    f"P4_N_INFS_{i}",
    
    ] for i in range(1,N_PROBS+1)]))
# parse_stats_df = pd.DataFrame(columns=columns,index=range(10))
# for i in range(1,N_PROBS+1):
#     parse_stats_df[f"ALL_VALS_{i}"] = df[f"ALL_VALS_{i}"]
#     parse_stats_df[f"ALL_OPS_{i}"] = df[f"ALL_OPS_{i}"]

#Grab just the grounded operational hints
op_hints_df = df.filter(regex="OP_\d")

def run():
    total_n_correct = 0
    total_n = 0
    n_incorrects = []
    for name, col in list(op_hints_df.items()):#[5:6]:
        prob_num = re.findall(r'\d+',name)[0]
        #if(int(prob_num) != 14): continue
            
        do_problem = globals()[f'do_problem{prob_num}']
        #print(f'\n{name}:')
        print(f'** {prob_num} **')
        prob_n_correct = 0
        num_col = 0
        for i, op_hint in enumerate(col):
            # Don't bother if they got the problem wrong 
            if(not df[f'COR_{prob_num}'][i]):
                continue 
            p = do_problem(op_hint)
            
            print(p.summary(all_phases=True))
            print(p.policy)
            #print(p.data[4].keys())
            # parse_stats_df.at[i,f"HAS_COR_{prob_num}"] = p.has_correct
            # parse_stats_df.at[i,f"N_INCOR_{prob_num}"] = p.num_incorrect
            # parse_stats_df.at[i,f"N_INFS_{prob_num}"] = p.num_forward_inferences
            # parse_stats_df.at[i,f"P4_HAS_COR_{prob_num}"] = p.data[5]['has_correct']
            # parse_stats_df.at[i,f"P4_N_INCOR_{prob_num}"] = p.data[5]['num_incorrect']
            # parse_stats_df.at[i,f"P4_N_INFS_{prob_num}"] = p.data[5]['num_fwd_infs']
            # if(p.has_correct):
            #     total_n_correct += 1
            #     prob_n_correct += 1
            #     n_incorrects.append(p.num_incorrect)
            if(not p.is_null):
                for expl in p.unq_expls:
                    print(expl)
            #print("--phase 4--")
            #for expl in p.get_unq_expls(phase=4):
            #    print(expl)
            num_col += 1
            print()
            
        #print("<<", num_col)
        total_n += num_col
        #print("Problem Has Correct:", prob_n_correct / num_col)    
        #print()
    #print("Total Has Correct:", total_n_correct / total_n)
    #print("Average Incorrect:", np.average(n_incorrects))
        

run()
