from tutorenvs.fractions_std import FractionArithmetic
from tutorenvs.multicolumn_std import MultiColumnAddition
from cre.transform import MemSetBuilder
from cre import MemSet
from apprentice.agents.cre_agents.state import State, encode_neighbors
from apprentice.agents.cre_agents.environment import TextField

from apprentice.agents.cre_agents.learning_mechs.process.process import *

def newSA(hp, sel, args, short_name=None, state=None, next_state=None, prob_uid=None):
    if(short_name is None):
        short_name = f"{hp}({sel})"
    skill = PrimSkill(how_part=hp)
    match = [sel, *args] if sel else args
    sa  = SkillApp(skill, match, short_name=short_name,
            state=state, next_state=next_state, prob_uid=prob_uid)
    skill.add_app(sa)
    return sa

def make_skill_apps(env, args, name_demo, ind_pref=0):
    memset_builder = MemSetBuilder()
    state_cls = State(None)

    env.set_problem(*args)
    
    def get_state(env):
        state = env.get_state()
        state = encode_neighbors(state)
        wm = memset_builder(state, MemSet())
        state_uid = f"S_{wm.long_hash()}"
        state = state_cls({'__uid__' : state_uid, 'working_memory' : wm})
        state.is_done = env.is_done
        return state, wm, state_uid

    skill_apps = []
    i = 0
    start_uid = None
    while(not env.is_done):
        state, wm, state_uid = get_state(env)

        if(i == 0):
            prob_uid = state_uid
            state.is_start = True

        demos = env.get_all_demos()
        if(ind_pref < len(demos)):
            demo = demos[ind_pref]
        else:
            demo = demos[0]

        op, name = name_demo(demo, env)
        sel = wm.get_fact(id=demo.sai[0])
        args = [wm.get_fact(id=a) for a in demo.args] if demo.args else []
        
        # print(sel, args)
        env.apply(demo)
        next_state, _, _ = get_state(env)
        
        sa = newSA(op, sel, args, short_name=name, state=state, next_state=next_state, prob_uid=prob_uid)
        skill_apps.append(sa)

        i += 1
    return skill_apps

def name_demo_fracs(demo, env):
    sel = demo.sai[0]
    if(env.problem_type == "M"):
        d = {
            "ans_num": ("m", "mn"),
            "ans_den": ("m", "md"),
            "done": ("d", "d"),
        }
    elif(env.problem_type == "AS"):
        d = {
            "ans_num": ("a", "an"),
            "ans_den": ("cp", "cp"),
            "done": ("d", "d"),
        }
    elif(env.problem_type == "AD"):
        d = {
            "check_convert" : ("x", "x"),
            "conv_den1": ("m", "cd1"),
            "conv_den2": ("m", "cd2"),
            "conv_num1": ("m", "cn1"),
            "conv_num2": ("m", "cn2"),
            "ans_num": ("a", "an"),
            "ans_den": ("cp", "ad"),
            "done": ("d", "d"),
        }
    return d[sel]

def frac_skill_apps(op, lst, ind_pref=0, **kwargs):
    env = FractionArithmetic(n_fracs=2, demo_args=True, demo_how=True)
    return make_skill_apps(env, (op, lst), name_demo_fracs, ind_pref)

def name_demo_mc(demo, env):
    sel = demo.sai[0]
    
    # print(demo, demo.how_str)
    if('done' in sel):
        return 'd', 'd'
    elif('carry' in sel):
        if("a,b,c" in demo.how_str):
            op = "c3"
        else:
            op = "c2"
    elif('out' in sel):
        if("Add" in demo.how_str):
            pre = 'a'
            if("a,b,c" in demo.how_str):
                op = "a3"
            else:
                op = "a2"
        else:
            op = 'cp'
    return (op, op+sel[-1])
            
def mc_skill_apps(a, b, ind_pref=0, **kwargs):
    env = MultiColumnAddition(demo_args=True, demo_how=True, **kwargs)
    return make_skill_apps(env, (a, b), name_demo_mc, ind_pref)


def test_mc():
    # seq1 = [a21, c21, a32, c32, a33, c33, cp4, d] = mc_skill_apps("777", "777")
    # seq2 = [a21, a22, a23, d] = mc_skill_apps("333", "333")
    # seq3 = mc_skill_apps("333", "373")
    # seq4 = mc_skill_apps("333", "63")

    probs = [("777", "777"),
             ("333", "333"),
             ("128", "456"),
             ("345", "567"),
             ("173", "163"),
             ("633", "933")
             ]

    seqs = []
    for prob in probs:
        seqs.append(mc_skill_apps(*prob, ind_pref=0))
        seqs.append(mc_skill_apps(*prob, ind_pref=1))
    # seq1 = mc_skill_apps("777", "777")
    # seq2 = mc_skill_apps("333", "333")
    # seq3 = mc_skill_apps("128", "456")
    # seq4 = mc_skill_apps("345", "567")
    # seq5 = mc_skill_apps("173", "163")
    # seq6 = mc_skill_apps("633", "933")
    # seqs = [seq1, seq2, seq3, seq4, seq5, seq6]
    # sequences = 
    


    
    order = np.arange(len(seqs))
    np.random.shuffle(order)


    # order = [0, 2, 5, 4, 3, 1]
    # order = [3, 0, 1, 5, 2, 4]
    # order = [7, 2, 11, 10, 3, 0, 8, 6, 1, 9, 5, 4]
    # order = [11  1  4 10  6  3  5  2  9  0  7  8]
    # order = [7, 9, 2, 11, 10, 8, 1, 3, 0, 5, 6, 4]
    # order = [4, 1, 3, 7, 0, 6, 10, 8, 2, 5, 9, 11]
    # order = [5, 8, 4, 0, 1, 3, 10, 11, 7, 2, 9, 6]
    # order = [5, 10, 3, 1, 9, 8, 4, 0, 11, 2, 7, 6]
    # order = [0,3]   
    order = [7, 8, 4, 11, 3, 0, 10, 6, 5, 1, 2, 9] 


    print("SEQs:", order)
    for i in order:
        print(", ".join([str(x) for x in seqs[i]]))
    

    # with PrintElapse("Moo"):
    #     lrn = HTNLearner()
    #     for i in order:
    #         lrn.ifit(seqs[i])

    # with PrintElapse("Moo"):
    #     lrn = HTNLearner()
    #     for i in order:
    #         lrn.ifit(seqs[i])

    # with PrintElapse("Moo"):
    lrn = HTNLearner()
    for i in order:
        seq = seqs[i]
        # print()
        # print("START: ", ", ".join([str(x) for x in seq]))
        o = np.arange(len(seq))
        # np.random.shuffle(o)
        for j in o:
            sa = seq[j]
        # for j, sa in enumerate(seq):
            lrn.ifit(sa)

            # if(hasattr(lrn, 'grammar')):
            #     ok, pps = parse_subseq(lrn.grammar, seq[:j+1])
            #     print(sa, pps)
            #     if(ok):
            #         grps = groups_items_by_depends(pps)
            #         print(','.join([
            #             f"[{','.join([str(sk.skill_apps[0]) for sk in group])}]" for group in grps
            #         ]))
            # groups_items_by_depends()
        print("+".join(probs[i//2]))
        print("SEQ:", ", ".join([str(x) for x in seq]))
        print(lrn.grammar)
        print()




    # print("-------")
    # for seq in lrn.sequences:
    #     print(", ".join([str(x) for x in seq]))
    # print("-------")


    print("SEQs:", order)
    for i in order:
        # parse_subseq(lrn.grammar, seqs[i][:-1])
        print(", ".join([str(x) for x in seqs[i]]))
    
    # print(", ".join([str(x) for x in seq6]))
    print("Skills:")
    print(", ".join([str(sk.skill_apps[0]) for sk in lrn.item_skills]))

    # seq = mc_skill_apps("909","492")
    seq = mc_skill_apps("773","668")
    print(parse_subseq(lrn.grammar, seq))



def test_frac():
    print("\n"*10, "START FRAC:")
    probs = [("+", [(1, 2), (2, 2)]),
             ("x", [(3, 4), (9, 8)]),
             ("+", [(1, 3), (3, 4)]),
            ]

    seqs = []
    for prob in probs:
        seqs.append(frac_skill_apps(*prob, ind_pref=0))
        seqs.append(frac_skill_apps(*prob, ind_pref=-1))

    order = np.arange(len(seqs))
    np.random.shuffle(order)
    print("SEQs ORDER:", order)

    lrn = HTNLearner()
    for i in order:
        seq = seqs[i]
        lrn.ifit_seq(seq)

        op, vals = probs[i//2]
        print(op.join(f"{t[0]}/{t[1]}" for t in vals))
        print("SEQ:", ", ".join([str(x) for x in seq]))
        print(lrn.grammar)
        print()

    print("SEQs ORDER:", order)
    for i in order:
        print(", ".join([str(x) for x in seqs[i]]))

    print("Skills:")
    print(", ".join([str(sk.skill_apps[0]) for sk in lrn.item_skills]))

    seq = frac_skill_apps("+", [(3, 4), (9, 8)], ind_pref=-1)
    print(parse_subseq(lrn.grammar, seq))



import re
def make_simple_seq(syms, order=None):
    if(isinstance(syms, int)):
        syms = [chr(97+i) for i in range(syms)]
    state_cls = State(None)
    
    state_obj = {}
    wm = MemSet()

    loc_order = sorted(syms)
    prev = None
    covered_alpha = set()
    print(loc_order)
    for i, sym in enumerate(loc_order):
        sym_alpha = re.sub(r'\d+', '', sym)

        if(sym_alpha not in covered_alpha):
            state_obj[f"arg_{sym_alpha}"] = {
                "id": f"arg_{sym_alpha}", "type" : "TextField", "value" : "", "locked": True,
                "x" : 2*i *110, "y" : 0, "width" : 100, "height" : 100,
                "left" : prev, "right" : sym,
            }
            if(prev):
                state_obj[prev]['right'] = f"arg_{sym_alpha}"
            prev = f"arg_{sym_alpha}" 
            covered_alpha.add(sym_alpha)
        elif(prev):
            state_obj[prev]['right'] = sym
        # nxt = loc_order[i+1] if i < len(loc_order)-1 else None
        # print(prev, sym, nxt)
        state_obj[sym] = {
            "id": sym, "type" : "TextField", "value" : "", "locked": False,
            "x" : 2*(i+1) *110, "y" : 100, "width" : 100, "height" : 100,
            "left" : prev, "right" : None,
        }
        prev = sym
    print([(d['left'], d['id'], d['right']) for d in state_obj.values()])
    memset_builder = MemSetBuilder()

    if(order is None):
        order = syms

    skill_apps = []
    for i, sym in enumerate(order):  
        sym_alpha = re.sub(r'\d+', '', sym)  
        wm = memset_builder(state_obj, MemSet())
        state_uid = f"S_{wm.long_hash()}"
        if(i == 0):
            prob_uid = state_uid
        state = state_cls({'__uid__' : state_uid, 'working_memory' : wm})

        sa = newSA(sym, wm.get_fact(id=sym), [wm.get_fact(id=f"arg_{sym_alpha}")], short_name=sym, state=state, prob_uid=prob_uid)
        skill_apps.append(sa)

        state_obj[sym]['value'] = "x"
        state_obj[sym]['locked'] = True

    return skill_apps


def test_parse_subseq():

    [a,b,c,t,u,v,x,y,z,q,r,s,l,m,n] = \
        make_simple_seq(['a','b','c','t','u','v','x','y','z','q','r','s', 'l','m','n'])

    A0 = MethodSkill([a,b,c], optionals=[1,2])
    A = MacroSkill("A", [A0])
    C0 = MethodSkill([x,y,z])
    C = MacroSkill("C", [C0])
    D0 = MethodSkill([q,r,s])
    D = MacroSkill("D",[D0])
    B0 = MethodSkill([D], optionals=[0])
    B1 = MethodSkill([t,u,v])
    B2 = MethodSkill([l,m,n], optionals=[0,1,2], unordered=True)
    B = MacroSkill("B",[B0, B1, B2])
    S0 = MethodSkill([A,B,C])
    S = MacroSkill("S", [S0])

    grammar = SkillGrammar()
    grammar.add(S)
    grammar.add(A)
    grammar.add(B)
    grammar.add(C)
    grammar.add(D)

    print(grammar)

    ok, pp = parse_subseq(grammar, [a])

    assert str(groups_items_by_depends(pp)) == "[[b], [c], [n, m, l, t, q], [x]]"
    print()

    # parse_subseq(grammar, [a, b, t])
    # parse_subseq(grammar, [a, l])
    # parse_subseq(grammar, [a, n, x])

def test_htn_lrn():
    seq1 = mc_skill_apps("333", "333", ind_pref=0)
    seq2 = mc_skill_apps("777", "777", ind_pref=0)

    fit_seq = lambda lrn, _seq, r : [lrn.ifit(sa,reward=r) for sa in _seq]

    lrn = HTNLearner()
    fit_seq(lrn, seq1, 1)
    fit_seq(lrn, seq2, 1)


    for sa in seq1:
        print(lrn.get_next_skill_apps(sa.state))

    print("--------------")
    for sa in seq2:
        print(lrn.get_next_skill_apps(sa.state))

    # fit_seq(lrn, seq2, -1)
    print("AFTER")
    print(lrn.grammar)
    fit_seq(lrn, seq1, -1)
    print(lrn.grammar)







if(__name__ == "__main__"):
    # test_mc()
    test_frac()
    # test_parse_subseq()
    # test_htn_lrn()

