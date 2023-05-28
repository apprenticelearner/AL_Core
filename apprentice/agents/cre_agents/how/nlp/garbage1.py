from collections import Counter

def foo(p_args, e_args):
    p_cnts, e_cnts = Counter(p_args), Counter(e_args)
    arg_scr = 0
    for arg, p_cnt in p_cnts.items():
        arg_scr += p_cnt-abs(p_cnt-e_cnts.get(arg, 0))
    return arg_scr / max(len(p_args), len(e_args))

# print(f"A_SCR: {arg_scr / max(len(p_args), len(e_args)):.2}", e_args, p_args )

print(foo(['a','a'], ['a']))
