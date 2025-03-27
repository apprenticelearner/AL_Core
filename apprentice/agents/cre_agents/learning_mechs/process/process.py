
from .base import PrimSkill, MacroSkill, SkillApp, MethodSkill, SkillGrammar, SymbolFactory
from .parse import RHSChange, get_best_changes, best_alignment, parse_w_changes, apply_grammar_changes#, propose_child_apps
import numpy as np
from numba import njit
from cre.utils import PrintElapse
from ..registers import register_process
from apprentice.agents.cre_agents.cre_agent import SkillApplication
from copy import copy

class DummySkill:
    def __init__(self,):
        self.uid = "dummy"
    def __call__(self, *rest):
        return rest[0].id
    def __str__(self):
        return "???"


######################################

def best_order(T):
    L = len(T)
    in_weight = np.sum(T,axis=0)
    min_starts = (in_weight==np.min(in_weight)).nonzero()[0]
    for i_start in min_starts:
        alignment = -np.ones(L, dtype=np.int64)

        # for i in range(L):
        #     np.argmax()

        #     alignment[i] = T[]
        #     T1 = np.zeros((L, L), dtype=np.float64)
            
    print(in_weight)

    # with np.printoptions(precision=4, linewidth=10000, suppress=True):
    #     print(T1, order)
    #     print(T2, order)
    return T1, order

def _best_order(T):
    T = T+0.1
    L = T.shape[0]
    T1 = np.zeros((L, L), dtype=np.float64)
    T2 = np.zeros((L, L), dtype=np.int64)

    T1[0, 0] = 1
    T2[0, 0] = 0

    for i in range(1, L):
        T1[i, :] = np.max(T1[i - 1, :] * T.T, 1)
        T2[i, :] = np.argmax(T1[i - 1, :] * T.T, 1)

    order = np.empty(L, dtype=np.int64)
    order[-1] = np.argmax(T1[L - 1])
    for i in reversed(range(1, L)):
        order[i - 1] = T2[i, order[i]]

    with np.printoptions(precision=4, linewidth=10000, suppress=True):
        print(T1, order)
        print(T2, order)
    return T1, order

def connected_components(T):
    covered_inds = set()
    groups = []
    
    for i in range(len(T)):
        if(i in covered_inds):
            continue
        group = []
        queue = [i]
        while(len(queue) > 0):
            new_queue = []
            for i in queue:
                if(i not in covered_inds):
                    group.append(i)
                    covered_inds.add(i)
                    for j in T[i].nonzero()[0]:
                        if(j not in covered_inds):
                            new_queue.append(j)
            queue = new_queue
        groups.append(group)
    return groups


@njit(cache=True)
def reorder_matrix(M, order):
    N = np.empty_like(M)
    L = M.shape[0]
    for i in range(L):
        for j in range(L):
            N[i,j] = M[order[i],order[j]]
    return N

@njit(cache=True)
def score_next(order, i, A_min, A_max, C, O, D):
    # print()
    L = len(O)

    # Find oL the first item left of i not disjoint with i
    nL = -1
    for k in order[::-1]:
        if(not D[i,k]):
            nL = k
            break

    nL_adj = (nL == order[-1]) * .05

    prev = order[-1]
    

    # Count order violations from choosing i next
    in_seq = np.zeros(L,dtype=np.int8)
    in_seq[order] = 1
    Ord = 0
    for k in range(L):
        if(i == k):
            continue

        if(in_seq[k]):
            if(A_max[k,i] < 0):
                Ord -= 1
        else:
            if(A_max[i,k] < 0):
                Ord -= 1

    Adj = 0
    SAn = 0
    WAn = 0
    DAn = 0
    Ov = 0
    maxD = 0
    if(nL != -1):
        in_bound = A_min[nL,i] <= 1 and A_max[nL,i] >= 1
        maxD = max(A_max[nL,i], 1)
        # print("THIS", nL, i, A_min[nL,i] <= 1, A_max[nL,i] >= 1, C[i,nL],  C[nL,i], O[nL, i])
        # SAn = in_bound and C[i,nL] and C[nL, i] and ((1+nL_adj)*(.5+O[nL, i]) / maxD)
        # WAn = in_bound and (C[i,nL] | C[nL,i]) and ((1+nL_adj)*(.5+O[nL,i]) / maxD)
        SAn = in_bound and C[i,nL] and C[nL, i] and ((.5+O[nL, i]) / maxD)
        WAn = in_bound and (C[i,nL] | C[nL,i]) and ((.5+O[nL,i]) / maxD)
        DAn = D[prev,i]*O[nL,i]
        Ov = O[prev,i]
        Adj = max(WAn, DAn)

    # print(f"i={i}", f"nL={nL}", f"prev={prev}", maxD)

    # distN = 0.0
    # for v, k in enumerate(order):
    #     if(C[i,k] and A_max[k,i]):
    #         d = 0
    #         for j in order[v+1:]:
    #             if(not D[i,j]):
    #                 d += 1
    #         distN -= max(d - A_max[k,i], 0) / abs(A_max[k,i])
    # return 100*Ord + 10*SAn + (WAn+DAn+Ov)
    return (Ord, SAn, WAn, Ov, DAn), Ov






@njit(cache=True)
def get_swap_tup(i, j, order, A_min, A_max, C, O, D, calc_old):
    # print()
    # print("calc", i, j )
    oi = i
    oj = j
    L = len(O)
    oL, oR = i-1, i+1
    nL, nR = j-1, j

    i = order[i]

    # Find oL the first item left of i not disjoint with i
    oL = -1
    for k in range(oi-1,-1,-1):
        if(not D[i,order[k]]):
            oL = k
            break
    oL_adj = (oi-oL == 1) * .05

    # Find oR the first item right of i not disjoint with i
    oR = L
    for k in range(oi+1,L):
        if(not D[i,order[k]]):
            oR = k
            break
    oR_adj = (oR-oi == 1) * .05

    # Find nL the first item left of j not disjoint with i
    nL = -1
    for k in range(oj-1,-1,-1):
        if(not D[i,order[k]]):
            nL = k
            break
    nL_adj = (oj-nL == 1) * .05

    # Find nR the first item right of j not disjoint with i
    nR = L
    for k in range(oj,L):
        if(not D[i,order[k]]):
            nR = k
            break
    nR_adj = (nR-oj == 0) * .05

    oL = -1 if(oL < 0) else order[oL]
    oR = -1 if(oR >= L) else order[oR]
    nL = -1 if(nL < 0) else order[nL]
    nR = -1 if(nR >= L) else order[nR]

    # Total out-of-order issues that would be resolved / created
    Ord = 0
    for k in range(oj,oi):
        k = order[k]
        if(A_min[k,i] < 0):
            Ord += 1
        elif(A_max[k,i] > 0):
            Ord -= 1
        
    # Strong Adjacencies old
    SAo = 0.0
    if(oR != -1 and calc_old): 
        SAo += A_min[i,oR] <= 1 and A_max[i,oR] >= 1 and C[i,oR] and C[oR,i] and (.5+(1+oR_adj)*O[i,oR])
    if(oL != -1 and calc_old):
        SAo += A_min[oL,i] <= 1 and A_max[oL,i] >= 1 and C[i,oL] and C[oL,i] and (.5+(1+oL_adj)+O[oL,i])
    if(nL != -1 and nR != -1):
        # print("OLD", A_min[nL,nR], A_max[nL,nR], C[nL,nR], C[nR, nL], (.5+O[nR, nL]))
        SAo += A_min[nL,nR] <= 1 and A_max[nL,nR] >= 1 and C[nL,nR] and C[nR, nL] and (.5+(1+nL_adj+nR_adj)*O[nR, nL])

    # Strong Adjacencies New
    SAn = 0.0
    if(nR != -1):
        SAn += A_min[i,nR] <= 1 and A_max[i,nR] >= 1 and C[i,nR] and C[nR, i] and (.5+(1+nR_adj)*O[i,nR])
    if(nL != -1):
        SAn += A_min[nL,i] <= 1 and A_max[nL,i] >= 1 and C[i,nL] and C[nL, i] and (.5+(1+nL_adj)*O[nL, i])
    if(oL != -1 and oR != -1):
        SAn += A_min[oL,oR] <= 1 and A_max[oL,oR] >= 1 and C[oL,oR] and C[oR,oL] and (.5+(1+oL_adj+oR_adj)*O[oL,oR])

    # Weak Adjacencies oLd
    WAo = 0.0
    if(oR != -1 and calc_old):
        WAo += A_min[i,oR] <= 1 and A_max[i,oR] >= 1 and (C[i,oR] | C[oR,i]) and (.5+(1+oR_adj)*O[i,oR])
    if(oL != -1 and calc_old):
        WAo += A_min[oL,i] <= 1 and A_max[oL,i] >= 1 and (C[i,oL] | C[oL,i]) and (.5+(1+oL_adj)*O[oL,i])
    if(nL != -1 and nR != -1):
        WAo += A_min[nL,nR] <= 1 and A_max[nL,nR] >= 1 and (C[nL,nR] | C[nR, nL]) and (.5+(1+nL_adj+nR_adj)*O[nL,nR])

    # Weak Adjacencies New
    WAn = 0.0
    if(nR != -1):
        WAn += A_min[i,nR] <= 1 and A_max[i,nR] >= 1 and (C[i,nR] | C[nR,i]) and (.5+(1+nR_adj)*O[i,nR])
    if(nL != -1):
        WAn += A_min[nL,i] <= 1 and A_max[nL,i] >= 1 and (C[i,nL] | C[nL,i]) and (.5+(1+nL_adj)*O[nL,i])
    if(oL != -1 and oR != -1):
        WAn += A_min[oL,oR] <= 1 and A_max[oL,oR] >= 1 and (C[oL,oR] | C[oR, oL]) and (.5+(1+oL_adj+oR_adj)*O[oL,oR])

    # WAo = min(WAo, 1)
    # WAn = min(WAn, 1)

    # Overlaps old
    Oo = 0.0
    if(oR != -1 and calc_old):
        Oo += O[i,oR]
    if(oL != -1 and calc_old):
        Oo += O[oL,i]
    if(nL != -1 and nR != -1):
        Oo += O[nL,nR]

    # Overlaps New
    On = 0.0
    if(nR != -1):
        On += O[i,nR]
    if(nL != -1):
        On += O[nL,i]
    if(oL != -1 and oR != -1):
        On += O[oL,oR]

    distO = 0.0
    for v in range(0,i):
        if(C[i,v] and A_max[v,i]):
            distO -= max((i-v) - A_max[v,i], 0) / abs(A_max[v,i])
    for v in range(i,L):
        if(C[i,v] and A_max[i,v]):
            distO -= max((v-i) - A_max[i,v], 0) / abs(A_max[i,v])

    distN = 0.0
    for v in range(0,nR):
        if(C[i,v] and A_max[v,i]):
            distN -= max((nR-v) - A_max[v,i], 0) / abs(A_max[v,i])
    for v in range(nR,L):
        if(C[i,v] and A_max[i,v]):
            distN -= max((nR-i) - A_max[i,v], 0) / abs(A_max[i,v])

    # Oo /= ((oR != -1) + (oL != -1) + (nL != -1 and nR != -1) + .01)
    # On /= ((nR != -1) + (nL != -1) + (oL != -1 and oR != -1) + .01)

    # print(":", f"{oi}->{i}", f"{oj}->{nR}")
    
    # print(Ord, "o", SAo, WAo, Oo, oL, oR)
    # print(Ord, "n", SAn, WAn, On, nL, nR)
    # print(f"distON ", distN-distO)
    return (Ord, (WAn-WAo)+(SAn-SAo), On-Oo, distN-distO)

@njit(cache=True)
def reinsert(order, i,j):
    if(i == j):
        return order.copy()

    new = np.zeros_like(order)
    mi = min(i,j)
    ma = max(i,j)
    d0 = i > j 
    d1 = j > i 
    new[:mi] = order[:mi]
    new[j:j+1] = order[i:i+1]
    new[mi+d0:ma+d0] = order[mi+d1:ma+d1]               
    new[ma+1:] = order[ma+1:]
    return new

def _ensure_convert_skill_app(sa):
    if(type(sa) is SkillApplication):
        # TODO: (FIX HACKYNESS) Find way to merge cre_agent.SkillApplication
        #   with SkillApp and Skill with PrimSkill

        
        try:
            conv_funcs = sa.skill.agent.conversions
            how_str = sa.skill.how_part.minimal_str(ignore_funcs=conv_funcs)
        except:
            how_str = str(sa.skill.how_part)
        prim_skill = PrimSkill(how_str, sa.skill)

        sa = SkillApp(prim_skill, sa.match, sa.state, 
                      getattr(sa, 'next_state', None),
                      getattr(sa, 'prob_uid', None),
                      in_process=getattr(sa, 'in_process', False)) 
        prim_skill.add_app(sa)
    return sa

class PreseqTracker:
    # For each added SkillApp keeps track of all ordered sequences
    #  of SkillApps which can lead to each state. Requires that each
    #  SkillApp has .next_state defined. 
    def __init__(self):
        self.pre_seqs = {}
        self.done_uids = {}
        self.states = {}
        self.actions = {}

    def add_skill_app(self, sa, is_start=None, do_update=True):
        sa = _ensure_convert_skill_app(sa)
        s_info, n_info = self._get_sa_state_infos(sa)

        # Add sa to 'outs' of state and 'ins' of next_state 
        any_change = False
        if(sa not in s_info['outs']):
            any_change = True
            s_info['outs'].add(sa)
        if(sa not in n_info['ins']):
            any_change = True
            n_info['ins'].add(sa)

        if(any_change and do_update):
            try:
                prob_uid = self.resolve_prob_uid(sa, is_start=is_start)
                self.update_subseqs(prob_uid)
            except:
                pass
            

    def remove_skill_app(self, sa, is_start=None, do_update=True):
        s_info, n_info = self._get_sa_state_infos(sa)

        # Remove sa from 'outs' of state and 'ins' of next_state
        any_change = False
        if(sa in s_info['outs']):
            any_change = True
            s_info['outs'].remove(sa)
        if(sa in n_info['ins']):
            any_change = True
            n_info['ins'].remove(sa) 

        if(any_change and do_update):
            prob_uid = self.resolve_prob_uid(sa, is_start=is_start)
            self.update_subseqs(prob_uid)

    def _get_sa_state_infos(self, sa):
        next_state = getattr(sa,'next_state', None)
        assert next_state is not None, "next_state info required."
        # print("NEXT STATE", next_state)

        s_uid, n_uid = sa.state.get('__uid__'), next_state.get('__uid__')
        # print("SUID", s_uid[:5],  "NUID", n_uid[:5], id(self))

        if(s_uid not in self.states):
            self.states[s_uid] = {'outs' : set(), 'ins' : set()}

        if(n_uid not in self.states):
            self.states[n_uid] = {'outs' : set(), 'ins' : set()}

        s_info, n_info = self.states[s_uid], self.states[n_uid]
        return s_info, n_info

    def resolve_prob_uid(self, x, is_start=None):
        prob_uid = None

        if(isinstance(x, (SkillApp, SkillApplication))):
            sa = x
            sa = _ensure_convert_skill_app(sa)

            prob_uid = getattr(sa, 'prob_uid',None)
            state = sa.state
            s_uid = state.get('__uid__')

            if(prob_uid is None):
                if(is_start == True):
                    prob_uid = sa.prob_uid = s_uid

        # Otherwise is state
        else:
            state = x
            s_uid = state.get('__uid__')
            if(getattr(state,'is_start', is_start)):
                prob_uid = s_uid

        # Go through ancestors and see if they have prob_uid
        if(prob_uid is None):
            s_info = self.states.get(s_uid,{})
            # print("s_info", s_uid[:5], s_info, id(self))
            ins = s_info.get('ins',[])
            # print("INS", ins, is_start)
            for ancestor in ins:
                prob_uid = getattr(ancestor, "prob_uid", None)
                if(prob_uid is None and ancestor.state.is_start):
                    prob_uid = ancestor.state.get("__uid__")
                # print("PUID", prob_uid)
                if(isinstance(x, (SkillApp, SkillApplication)) and prob_uid):
                    x.prob_uid = prob_uid

                if(prob_uid is not None):
                    break

        if(is_start):
            state.is_start = True

        if(prob_uid is None):
            print(f"FAIL PROB UID is_start={is_start} uid={state.get('__uid__')} ins={ins}")
            raise RuntimeError(f"Could not resolve 'prob_uid' for {type(x).__name__} {x}")
        return prob_uid

    def update_subseqs(self, prob_uid):
        # Do a breadth first search from a problem start state
        #  indicated by 'prob_uid' to build all subsequences 
        seqs = []
        
        if(prob_uid not in self.states):
            return None

        pre_seqs = {}#self.pre_seqs.get(prob_uid,{})    
        next_sas = list(self.states[prob_uid]['outs'])
        done_uid = None

        while(len(next_sas) > 0):
            new_next_sas = []
            for sa in next_sas:
                next_state = sa.next_state
                s_uid, n_uid = sa.state.get('__uid__'), next_state.get('__uid__')
                # if(s_uid == n_uid):
                #     raise RuntimeError()

                if(s_uid in pre_seqs):
                    if(n_uid not in pre_seqs):
                        pre_seqs[n_uid] = set()

                    new_pre_seqs = pre_seqs[n_uid]
                    for seq in pre_seqs[s_uid]:
                        new_pre_seqs.add((*seq,sa))
                else:
                    new_pre_seqs = {(sa,)}
                pre_seqs[n_uid] = new_pre_seqs

                if(n_uid in self.states):
                    new_next_sas += list(self.states[n_uid]['outs'])

                if(next_state.is_done):
                    done_uid = n_uid
            next_sas = new_next_sas

        self.pre_seqs[prob_uid] = pre_seqs
        self.done_uids[prob_uid] = done_uid

    def get_preseqs(self, state, prob_uid=None):
        s_uid = state.get('__uid__')
        if(prob_uid is None):
            prob_uid = self.resolve_prob_uid(state)

        if(s_uid == prob_uid):
            return {tuple()}

        pre_seqs = self.pre_seqs.get(prob_uid, None)
        # print(s_uid, prob_uid, pre_seqs)
        if(pre_seqs is not None):
            return pre_seqs.get(s_uid, set())
        return set()

    def get_good_preseq(self, state, prob_uid=None):        
        pre_seqs = self.get_preseqs(state, prob_uid)
        if(len(pre_seqs) == 0):
            return None 

        first = None
        for preseq in pre_seqs:
            if(first is None):
                first = preseq
            if(all([getattr(x,'in_process', False) for x in preseq])):
                print("IS GOOD")
                return preseq
        print("NOT GOOD")
        return first



@register_process
class HTNLearner:
    def __init__(self, agent=None):
        self.reset()
        # self.grammar = None
        # self.item_skills = []
        # self.sequences = set()
        # self.seq_orders = []
        # self.O = np.empty((0,0), dtype=np.float32)
        # # self.A = np.empty((0,0), dtype=[('min', 'i4'), ('max', 'i4')])
        # self.A_min = np.empty((0,0), dtype=np.int32)
        # self.A_max = np.empty((0,0), dtype=np.int32)
        # self.C = np.empty((0,0), dtype=np.int8)
        # self.D = np.empty((0,0), dtype=np.int8)
        # self.P = np.empty((0,), dtype=np.int8)
        # self.pre_seqs = {}
        # self.states = {}
        # self.actions = {}
        self.agent = agent
        self.preseq_tracker = PreseqTracker()
        
        # self.agent_skill_to_prim_skill = {}

    def reset(self):
        self.item_skills = []
        self.sequences = set()
        self.seq_orders = []
        self.O = np.empty((0,0), dtype=np.float32)
        # self.A = np.empty((0,0), dtype=[('min', 'i4'), ('max', 'i4')])
        self.A_min = np.empty((0,0), dtype=np.int32)
        self.A_max = np.empty((0,0), dtype=np.int32)
        self.C = np.empty((0,0), dtype=np.int8)
        self.D = np.empty((0,0), dtype=np.int8)
        self.P = np.empty((0,), dtype=np.int8)
        self.grammar = None

# -------------------------------
# : ifit() and remove()

    def _prep_fit(self, state, skill_app, is_start, prob_uid):
        if(type(skill_app) is SkillApplication):
            # Ensure that skill_app has next_state
            if(getattr(skill_app, 'next_state', None) is None):
                wm = state.get("working_memory")
                next_state = skill_app.skill.agent.predict_next_state(wm, skill_app.action)
                skill_app.next_state = next_state

        # Convert into 
        skill_app = _ensure_convert_skill_app(skill_app)
        pst = self.preseq_tracker

        if(prob_uid is None):
            try:
                prob_uid = pst.resolve_prob_uid(skill_app, is_start)
            except:
                pass

        return skill_app, prob_uid

    def _add_and_refit_seqs(self, skill_app, is_start, prob_uid):
        pst = self.preseq_tracker
        pst.add_skill_app(skill_app, is_start)
        done_uid = pst.done_uids.get(prob_uid, None)
        if(done_uid is not None):
            for seq in pst.pre_seqs[prob_uid][done_uid]:
                seq = (*seq,)
                if(seq not in self.sequences):
                    self.ifit_seq(seq)

    def _remove_and_refit_seqs(self, skill_app, is_start, prob_uid):
        pst = self.preseq_tracker
        old_seqs = set()    
        done_uid = pst.done_uids.get(prob_uid, None)
        if(done_uid is not None):
            old_seqs = pst.pre_seqs.get(prob_uid, {}).get(done_uid, set())
        
        pst.remove_skill_app(skill_app, is_start)

        curr_seqs = pst.pre_seqs.get(prob_uid,{}).get(done_uid,set())
        removed_seqs = old_seqs.difference(curr_seqs)
        old_seqs = self.sequences.copy()
        # print("O R",)
        # print(old_seqs)
        # print(removed_seqs)

        # Clear sequences and refit everything, delay method pass until end.
        if(len(removed_seqs) > 0):
            self.reset()
            for seq in old_seqs:
                if(seq not in removed_seqs):
                    self.ifit_seq(seq, skip_method_pass=True)
                self.method_pass()

    def remove(self, state, skill_app, is_start=None, prob_uid=None):
        skill_app, prob_uid = self._prep_fit(state, skill_app, is_start, prob_uid)
        if(not prob_uid): return
        self._remove_and_refit_seqs(skill_app, is_start, prob_uid)

    
    def ifit(self, state, skill_app, is_start=None, prob_uid=None, reward=1):
        skill_app, prob_uid = self._prep_fit(state, skill_app, is_start, prob_uid)
        if(not prob_uid): return

        if(reward is not None and reward > 0):
            self._add_and_refit_seqs(skill_app, is_start, prob_uid)
            # pst.add_skill_app(skill_app, is_start)
            # done_uid = pst.done_uids.get(prob_uid, None)
        else:
            self._remove_and_refit_seqs(skill_app, is_start, prob_uid)


            # old_seqs = None    
            # done_uid = pst.done_uids.get(prob_uid, None)
            # if(done_uid):
            #     old_seqs = pst.pre_seqs.get(prob_uid, {}).get(done_uid, set())
            
            # pst.remove_skill_app(skill_app, is_start)
        
        # print(pst.done_uids)
        # if(done_uid is not None):
        #     # print("NOT NONE", prob_uid)
        #     if(reward is not None and reward > 0):
        #         for seq in pst.pre_seqs[prob_uid][done_uid]:
        #             seq = (*seq,)
        #             if(seq not in self.sequences):
        #                 self.ifit_seq(seq)

        #     elif((reward is None or reward <= 0) and old_seqs):
        #         curr_seqs = pst.pre_seqs.get(prob_uid,{}).get(done_uid,set())
        #         removed_seqs = old_seqs.difference(curr_seqs)
        #         old_seqs = self.sequences.copy()
        #         # print("O R",)
        #         # print(old_seqs)
        #         # print(removed_seqs)
        #         self.reset()
        #         for seq in old_seqs:
        #             if(seq not in removed_seqs):
        #                 self.ifit_seq(seq, skip_method_pass=True)
        #             self.method_pass()
        # else:
        #     print("NONE", prob_uid)

        # if(hasattr(self, "grammar")):
        #     print("--------------")
        #     print(self.grammar)
        #     print("--------------")

# -------------------------------
# : ifit() and remove()

    def ifit_seq(self, seq, skip_method_pass=False):
        seq = (*seq,)
        if(seq in self.sequences):
            return False

        new_item_skills = []
        prev_ov_ind = -1
        item_inds = []
        for seq_ind, sa in enumerate(seq):
            any_overlap = False
            for ind, item in enumerate(self.item_skills):
                if(item.overlap(sa) == 1.0):
                    prev_ov_ind = ind
                    any_overlap = True
                    self.item_skills[ind] = item.merge(sa)
                    # print(sa.prob_uid[:5], "MERGE", item, sa)
                    item_inds.append(ind)
                    # item.seq_inds.append(seq_ind)
                    break

            if(not any_overlap):
                item = sa.skill
                # print("NEW", item, len(self.item_skills)+len(new_item_skills))
                item_inds.append(len(self.item_skills)+len(new_item_skills))
                new_item_skills.append((prev_ov_ind+1, item))

        self.sequences.add(seq)
        self.seq_orders.append(item_inds)

        for ind, sa in new_item_skills:
            self.item_skills.append(sa)
        L0, L1 = self._expand()

        # with PrintElapse("Matrix Pass"):
        m_change = self.update_matricies_pass(seq, item_inds)
        did_change = m_change or L0 != L1

        # with np.printoptions(precision=2, linewidth=10000):
            # print("O")
            # print(self.O)
            # print("A_max")
            # print(self.A_max)
            # print("C")
            # print(self.C)
            # print("D")
            # print(self.D)
        # with PrintElapse("Reorder Pass"):
        if(did_change):
            self.reorder_pass(L0)
            # with PrintElapse("Method Pass"):
            if(not skip_method_pass):
                self.method_pass()

        return did_change

    def _expand(self):
        L0, L1 = len(self.O), len(self.item_skills)
        # print(L0, "->", L1)
        # A = np.empty((L,L,2), dtype=np)
        self.O = np.pad(self.O, ((0,L1-L0),(0,L1-L0)), 'constant', constant_values=-1)
        self.A_min = np.pad(self.A_min, ((0,L1-L0),(0,L1-L0)), 'constant', constant_values=0)
        self.A_max = np.pad(self.A_max, ((0,L1-L0),(0,L1-L0)), 'constant', constant_values=0)
        self.C = np.pad(self.C, ((0,L1-L0),(0,L1-L0)), 'constant', constant_values=1)
        self.D = np.pad(self.D, ((0,L1-L0),(0,L1-L0)), 'constant', constant_values=1)
        self.P = np.pad(self.P, (0,L1-L0), 'constant', constant_values=1)
        if(L0 > 0):
            self.C[0:L0, L0:L1] = 0
            self.P[L0:L1] = 0
        
        # print(self.C.shape)
        return L0, L1
            

    def update_matricies_pass(self, seq, item_inds):
        
        old_A_min = self.A_min.copy()
        old_A_max = self.A_max.copy()
        old_O = self.O.copy()
        old_C = self.C.copy()
        old_D = self.D.copy()
        old_P = self.P.copy()

        # start with adjacency
        for i, (ind_i, sa_i) in enumerate(zip(item_inds, seq)):
            for j, (ind_j, sa_j) in enumerate(zip(item_inds, seq)):
                if(i != j):                
                    delta = j-i
                    mi = self.A_min[ind_i, ind_j]
                    ma = self.A_max[ind_i, ind_j]

                    # print("delta", ind_i, ind_j, delta, mi == 0 and ma == 0)
                    if(mi == 0 and ma == 0):
                        self.A_min[ind_i, ind_j] = delta
                        self.A_max[ind_i, ind_j] = delta
                    else:
                        self.A_min[ind_i, ind_j] = min(mi,delta)
                        self.A_max[ind_i, ind_j] = max(ma,delta)


            for ind_j, skill_j in enumerate(self.item_skills):
                j_occured = ind_j in item_inds
                if(ind_i != ind_j):
                    self.O[ind_i, ind_j] = self.O[ind_j, ind_i] = skill_j.match_overlap(sa_i)
                    self.C[ind_i, ind_j] &= j_occured
                    self.D[ind_i, ind_j] &= ~j_occured
                    # self.D[ind_j, ind_i] &= ~j_occured
                self.P[ind_j] &= j_occured

        did_change = (
            np.any(old_A_min != self.A_min) or
            np.any(old_A_max != self.A_max) or
            np.any(old_O != self.O) or
            np.any(old_C != self.C) or
            np.any(old_D != self.D) or
            np.any(old_P != self.P)
        )
        return did_change
        # for ind_i, skill_i in enumerate(self.item_skills):
        #     i_occured = ind_i in item_inds
        #     if(i_occured)
        #     for ind_j, skill_j in enumerate(self.item_skills):
        #         j_occured = ind_j in item_inds


        #         self.C[ind_i, ind_j] &= co_occur
        #         self.C[ind_j, ind_i] = self.C[ind_i, ind_j]
        #         self.D[ind_i, ind_j] &= ~co_occur
        #         self.D[ind_j, ind_i] = self.D[ind_i, ind_j]
    def reorder_pass(self, L0):
        L = len(self.O)
        # print(f"BEFORE [{', '.join([str(sk.skill_apps[0]) for sk in self.item_skills])}]")
        C = self.C
        A_min = self.A_min
        A_max = self.A_max
        O = self.O
        D = self.D

        covered = np.zeros(L, dtype=np.int8)
        covered[0] = 1
        order_buff = np.arange(L, dtype=np.int64)

        # with PrintElapse("ORDER PART"):
        # score_matrix = np.zeros((L,L),dtype=np.float64)

        order = order_buff[:1]#np.array([0], dtype=np.int64)
        for i in range(1,L):            
            best_score = (-np.inf,)#(-np.inf,0,0,0)
            best_ov = -np.inf#(-np.inf,0,0,0)
            best_sj = -1
            best_ojs = []
            # print()
            # print("order", ', '.join([str(self.item_skills[k].skill_apps[0]) for k in order]))
            for j in range(L):
                if(not covered[j]):
                    # print()
                    score, ov = score_next(order, j, A_min, A_max, C, O, D)
                    # score_matrix[i,j] = score
                    # print(self.item_skills[j].skill_apps[0], score, ov)
                    if(score > best_score):
                        best_score = score
                        best_sj = j

                    if(ov > best_ov):
                        best_ov = ov
                        best_ojs = [j]
                    elif(ov == best_ov):
                        best_ojs.append(j)

            # print("BEST", self.item_skills[best_sj].skill_apps[0], best_score, best_ov)
            # If the best overlapping next item is not the best
            #   scoring next item then test if choosing the best
            #   overlapping next item first would reduce the score 
            #   of picking the next best scoring afterwards.
            best_j = best_sj
            if(best_sj not in best_ojs):
                for best_oj in best_ojs:
                    order_buff[i] = best_oj
                    order = order_buff[:i+1]
                    ins_score, ins_ov = score_next(order, best_sj, A_min, A_max, C, O, D)

                    if(ins_score >= best_score):
                        # print("SWAP", self.item_skills[best_sj].skill_apps[0])
                        best_j = best_oj                    
                        best_score = ins_score
                        # print("INS", self.item_skills[best_j].skill_apps[0], best_score, ins_ov)
                # print(best_score, ins_score)


            covered[best_j] = 1
            order_buff[i] = best_j
            order = order_buff[:i+1]

        # with np.printoptions(precision=2, linewidth=10000, suppress=True):
        #     print("score_matrix")
        #     print(score_matrix)

        self.A_min = reorder_matrix(self.A_min, order)
        self.A_max = reorder_matrix(self.A_max, order)
        self.O = reorder_matrix(self.O, order)
        self.C = reorder_matrix(self.C, order)
        self.D = reorder_matrix(self.D, order)
        self.P = self.P[order]
        self.item_skills = [self.item_skills[i] for i in order]
        # print(f"AFTER [{', '.join([str(sk.skill_apps[0]) for sk in self.item_skills])}]")

        # print("SEQ ORDERS:")
        seq_orders = []
        for i, seq in enumerate(self.sequences):
            so = self.seq_orders[i]
            seq_orders.append([np.nonzero(order==j)[0][0] for j in so])
            # print(seq_orders[-1])
        self.seq_orders = seq_orders



    def _reorder_pass(self, L0):
        L = len(self.O)
        print(f"BEFORE [{', '.join([str(sk.skill_apps[0]) for sk in self.item_skills])}]")
        C = self.C
        A_min = self.A_min
        A_max = self.A_max
        O = self.O
        D = self.D

        # First take an ordering pass to ensure that the item_seq order 
        #  satisfies A_min constraints
        order = np.arange(L, dtype=np.int64)#[i for i in range(L)]

        # First pass, try to insert any new items as best as possible
        i = L0
        while(i < L):
            best_tup = (0, 0, 0, 0, 0)
            best_j= -1
            for j in range(i-1,-1,-1):
                tup = get_swap_tup(i, j, order, A_min, A_max, C, O, D, False)
                # print(i,j,tup)
                if(tup > best_tup):
                    best_tup = tup
                    best_j = j

            if(best_j != -1):
                # order = reinsert(order, i, best_j)
                # print("insert", i, best_j)
                order = list(order)
                order.insert(best_j, order.pop(i))
                order = np.array(order, dtype=np.int64)
                # print("A_max")
                # print(reorder_matrix(self.A_max, order))
                # print("C")
                # print(reorder_matrix(self.C, order))
                # print(", ".join([str(self.item_skills[i].skill_apps[0]) for i in order]))
            # else:
            i += 1
            # print("LOOP")

        # Second pass reorder 
        i = L-1
        n_passes = 0
        while(i >= 0 and n_passes < L):
            best_tup = (0, 0, 0, 0, 0)
            best_j= -1
            for j in range(i-1,-1,-1):
                tup = get_swap_tup(i, j, order, A_min, A_max, C, O, D, True)
                # print(i,j,tup)
                if(tup > best_tup):
                    best_tup = tup
                    best_j = j

            if(best_j != -1):
                # order = reinsert(order, i, best_j)
                # print("insert", i, best_j)
                order = list(order)
                order.insert(best_j, order.pop(i))
                order = np.array(order, dtype=np.int64)
                # print("A_max")
                # print(reorder_matrix(self.A_max, order))
                # print("C")
                # print(reorder_matrix(self.C, order))
                # print(", ".join([str(self.item_skills[i].skill_apps[0]) for i in order]))
                i = L-1
            else:
                i -= 1
            n_passes += 1

        order = np.array(order, dtype=np.int64)
        self.A_min = reorder_matrix(self.A_min, order)
        self.A_max = reorder_matrix(self.A_max, order)
        self.O = reorder_matrix(self.O, order)
        self.C = reorder_matrix(self.C, order)
        self.D = reorder_matrix(self.D, order)
        self.P = self.P[order]
        self.item_skills = [self.item_skills[i] for i in order]
        print(f"AFTER [{', '.join([str(sk.skill_apps[0]) for sk in self.item_skills])}]")

        # print("SEQ ORDERS:")
        seq_orders = []
        for i, seq in enumerate(self.sequences):
            so = self.seq_orders[i]
            seq_orders.append([np.nonzero(order==j)[0][0] for j in so])
            # print(seq_orders[-1])
        self.seq_orders = seq_orders

    def method_pass(self):
        if(len(self.O) == 0):
            return 

        C = self.C
        O = self.O
        D = self.D
        A_min = self.A_min
        A_max = self.A_max
        P = self.P
        # print("C")
        # print(C)
        L = len(self.O)
        seq_orders = self.seq_orders
        seq_skills = self.item_skills
        symbol_factory = SymbolFactory()

        meth_scores = np.zeros((L,L), dtype=np.float32)
        for n in range(0, L):
            for m in range(n, L):
                val = 0
                c, o = 0.0, 0.0
                size = (m-n)+1
                okay = True
                un0, un1 = np.inf, 0
                for i in range(n, m):
                    for j in range(i+1, m+1):
                        # print(i,j)
                        if(D[i,j] > 0):
                            okay = False
                            break
                        c += (C[i,j] | C[j,i]) + (C[i,j] & C[j,i]) * (m-n)/2
                        o += O[i,j]

                        # Find largest unorder span 
                        lwr, upr = min(i,j), max(i,j)
                        if(A_min[lwr, upr] < 0 and A_max[lwr, upr] > 0):
                            un0 = min(un0, lwr)
                            un1 = max(un0, upr+1)

                    if(not okay):
                        break
                if(okay):
                    u = un0 == n and un1 == m+1
                    den = (size*size)-(0 if size == 1 else size)
                    meth_scores[n,m] = (2 * (u / size) + (c + o) / (2*den))
                    # print((n, m), c, f"{o:.2f}", u, meth_scores[n,m])

        # with np.printoptions(precision=3, linewidth=10000, suppress=True):
        #     print("meth_scores")
        #     print(meth_scores)
        
        meth_spans = []
        i = 0
        z_seqs = [str(sk.skill_apps[0]) for sk in self.item_skills]
        while(i < L):
            j = np.argmax(meth_scores[i,i:])+i
            score = meth_scores[i,j]
            meth_spans.append((i, j+1))
            # print(">>", i,j+1, score, z_seqs[i:j+1])
            i = j+1

        # print("meth_spans")
        # print(meth_spans)
        
        meth_L = len(meth_spans)
        meth_O = np.zeros((meth_L,meth_L), dtype=np.float32)
        meth_C = np.zeros((meth_L,meth_L), dtype=np.int8)

        method_infos = []
        for k0, (s0, e0) in enumerate(meth_spans):
            # Calculate the co-occurance C and overlap O
            #  between method candidates
            for k1 in range(k0+1, meth_L):
                if(k0 == k1):
                    continue
                s1, e1 = meth_spans[k1]    
                any_Cij = False
                any_Cji = False
                max_O = 0.0
                for i in range(s0,e0):
                    for j in range(s1,e1):
                        if(C[i, j]):
                            any_Cij = True
                        if(C[j, i]):
                            any_Cji = True
                        if(O[i,j] > max_O):
                            max_O = O[i,j]
                meth_C[k0, k1] = any_Cij
                meth_C[k1, k0] = any_Cji
                meth_O[k0, k1] = meth_O[k1, k0] = max_O

            # Determine if the method candidate has
            #  unordered sections and/or optional items
            # un0, un1 = np.inf, 0
            optionals = np.zeros(e0-s0, dtype=np.bool_)
            un_spans = []
            full_un = False
            prev_un_end = 0
            for i in range(s0, e0):
                # Check if i always co-occurs with the others
                all_co_occur = True
                for j in range(s0, e0):
                    if(i != j):
                        if(not C[j,i]):
                            all_co_occur = False
                        # lwr, upr = min(i,j), max(i,j)
                        # if(A_min[lwr, upr] < 0 and A_max[lwr, upr] > 0):
                        #     un0 = min(un0, lwr)
                        #     un1 = max(un0, upr+1)

                if(i >= prev_un_end):
                    un_end = i
                    for j in range(i+1, e0):
                        if(A_min[i, j] < 0 and A_max[i, j] > 0):
                            un_end = j+1

                    if(un_end != i):
                        # print("LUN", i-s0, un_end-s0)
                        prev_un_end = un_end
                        if(i==s0 and un_end == e0):
                            full_un = True
                        else:
                            un_spans.append((i-s0, un_end-s0))

                optionals[i-s0] = not all_co_occur

            

            # print(full_un, ">>", ', '.join([str(self.item_skills[i].skill_apps[0]) for i in range(s0,e0)]),len(method_infos))
            method_infos.append(((s0,e0), full_un, un_spans, optionals))
            # print((un0-s0, un1-s0), optionals)

        # print("METH_O")
        # print(meth_O)
        # print("METH_C")
        # print(meth_C)

        disjoint_groups = []
        group = [0]
        for k in range(1,len(meth_spans)):
            # (s0,e0) = meth_spans[k-1] 
            (s1,e1) = meth_spans[k]
            
            all_disjoint = True
            for g in group:
                s0,e0 = meth_spans[g]
                for i in range(s0,e0):
                    for j in range(s1,e1):
                        if(not D[i, j]):
                            all_disjoint = False
                    # print(i,j, D[i, j])
            if(all_disjoint):
                group.append(k)
            else:
                disjoint_groups.append(group)
                group = [k]
        disjoint_groups.append(group)

        # print("disjoint_groups")
        # print(disjoint_groups)


        # regroup_map = np.arange(len(O), dtype=np.int64)
        ind = 0
        new_items = []
        new_macros = []
        new_opts = np.zeros(len(O), dtype=np.bool_)
        for k, grp in enumerate(disjoint_groups):
            
            _, grp_full_un, _, _ = method_infos[grp[0]]
            # if(len(grp) > 1 or full_un or len(un_spans) > 0):
            meths_items_opts = []
            always_occurs = True

            if(len(grp) > 1 or grp_full_un):
                grp_sym = symbol_factory()
                grp_sym_ind = len(new_macros)

            for meth_ind in grp:
                (s,e), full_un, un_spans, opts = method_infos[meth_ind]

                # print("un_spans", un_spans)
                
                if(full_un):
                    # Reorder so that non-optional actions come first
                    order = np.argsort(opts)
                    items_opts = [(seq_skills[s+i],opts[i]) for i in order]
                else:
                    items_opts = []
                    un_span = un_spans[0] if len(un_spans) > 0 else None
                    un_ind = 0
                    i = 0
                    while(i < e-s):
                        if(un_ind < len(un_spans) and i == un_spans[un_ind][0]):
                            un0, un1 = un_spans[un_ind]
                            # print(">>", [s+j for j in range(un0,un1)], [seq_skills[s+j] for j in range(un0,un1)])
                            meth = MethodSkill([seq_skills[s+j] for j in range(un0,un1)],
                                    unordered=True, optionals=opts[un0:un1])
                            macro = MacroSkill(symbol_factory(), [meth])
                            new_macros.append(macro)

                            items_opts.append((macro, False))
                            un_ind += 1
                            i = un1
                        else:
                            # print(">>", [s+i], [seq_skills[s+i]])
                            items_opts.append((seq_skills[s+i], opts[i]))# or not P[s+i]))
                            i += 1

                meths_items_opts.append(items_opts)

            # If multiple candidate methods are grouped 
            #  or the method candidate is unordered 
            #  then add a new macro.
            if(len(grp) > 1 or grp_full_un):
                meths = []
                for grp_ind, items_opts in enumerate(meths_items_opts):
                    (s,e), full_un, un_spans, opts = method_infos[grp[grp_ind]]

                    items = [] 
                    opt_mask = np.zeros(len(items_opts),dtype=np.bool_) 
                    for i, (item, opt) in enumerate(items_opts):
                        items.append(item)
                        opt_mask[i] = opt

                    # print(items, opt_mask)
                    # print(">>", items, full_un, meth_ind)
                    meth = MethodSkill(items, unordered=full_un, optionals=opt_mask)
                    meths.append(meth)
                    # always_occurs &= 
                    # print(">", meth)                    
                    
                macro = MacroSkill(grp_sym, meths)
                new_items.append(macro)
                new_macros.insert(grp_sym_ind, macro)
                ind += 1

            # Otherwise just keep the method cand's items
            #  in the item sequence for this step.
            else:
                (s,e), _, _, _ = method_infos[grp[0]]
                items_opts = meths_items_opts[0]
                # print(items_opts, s, e)
                for item, opt in items_opts:
                    new_opts[ind] = opt 
                    new_items.append(item)
                    ind += 1

        new_L = ind
        new_opts = new_opts[:new_L]

        # print(", ".join([str(it.skill_apps[0]) for it in new_items]), new_opts)
        # print("regroup_map")
        # print(regroup_map)

        grammar = SkillGrammar()
        # print(new_items, new_opts)
        S0 = MethodSkill(new_items, optionals=new_opts, fix_bad_opts=True)
        S = MacroSkill("S", [S0])
        grammar.add(S)
        for macro in new_macros:
            grammar.add(macro)
        # print(grammar)
        self.grammar = grammar

    def get_next_item_paths(self, state, preseq_tracker=None, prob_uid=None, group_by_depends=False):
        if(preseq_tracker is None):
            preseq_tracker = self.preseq_tracker

        # try:
        #     preseqs = preseq_tracker.get_preseqs(state)
        # except RuntimeError as e:
        #     return []

        # if(len(preseqs) == 0):

        #     print("NO PRESEQS")
        #     s_uid = state.get('__uid__')
        #     # print(preseq_tracker.states.get(s_uid,None))

        grammar = getattr(self, 'grammar', None)
        if(grammar is None):# or len(preseqs) == 0):
            return []
        else:
            # print("GOOD PRESEQ LEN", len(preseq))

            # TODO: There is probably a more effecient way to do
            #  This than checking every possible preseq
            # for preseq in preseqs:
                # okay, paths = parse_subseq(grammar, preseq)
            okay, paths = parse_state_from_tracker(grammar, state, preseq_tracker, prob_uid=prob_uid)
            # if(okay):
            #     break

            # if(okay == False):
            #     print("PARSE FAIL")
            if(group_by_depends):
                return group_paths_by_depends(paths)
            else:
                return paths

    def get_next_skill_apps(self, state, preseq_tracker=None, prob_uid=None, group_by_depends=False):
        if(prob_uid is None):
            try:
                prob_uid = preseq_tracker.resolve_prob_uid(state)
            except:
                print("BAD PROB UID")
                return []

        groups = self.get_next_item_paths(state, preseq_tracker, prob_uid, group_by_depends)

        if(not group_by_depends):
            groups = [groups]
            skill_apps = []

        skill_app_grps = []
        skill_apps = []
        wm = state.get("working_memory")
        for group in groups:                
            for path in group:
                # print(path)
                macro, meth_ind, item_ind, cov = path[-1]
                item = macro.methods[meth_ind].items[item_ind]
                # print(item, item.skill_apps[0].match)

                # TODO: This is kind of a hack to get the skill and match... 
                #  fix will go along with unifying SkillApp + SkillApplication  
                #  and Skill and PrimSkill
                sa = item.skill_apps[0]
                match = [wm.get_fact(id=m.id) for m in sa.match]
                skill = item.skill
                if(skill is None):
                    skill = DummySkill()
                sk_app = SkillApplication(skill, match, state, prob_uid=prob_uid)
                # print(">>", type(sk_app))
                if(sk_app):
                    sk_app.path = path
                    skill_apps.append(sk_app)

            if(group_by_depends):
                skill_app_grps.append(skill_apps)
                skill_apps = []

        if(not group_by_depends):
            return skill_apps
        else:
            return skill_app_grps

        # mac_C = np.zeros((new_L, new_L), dtype=np.int8)
        # new_seq_orders = []
        # for seq_order in seq_orders:
        #     nso = []
        #     prev_rg = -1
        #     for j in seq_order:
        #         rg = regroup_map[j]
        #         if(rg != prev_rg):
        #             nso.append(rg)
        #         prev_rg = rg
        #     new_seq_orders.append(nso)

class ParsePath:
    def __init__(self, steps, depends_on=[], dependants=[]):
        self.steps = steps
        self.depends_on = depends_on
        self.dependants = dependants
    
    def __getitem__(self, ind):
        return self.steps[ind]

    def __len__(self):
        return len(self.steps)

    def __str__(self):
        step_strs = []
        for (macro, meth_ind, item_ind, cov) in self.steps:
            method = macro.methods[meth_ind]

            item_strs = []
            is_un = cov is not None
            pre = "\033[4m" if(is_un) else ""
            # post = "\033[0m" if(is_un) else "\033[0m"
            meth_str = ""
            for i, item in enumerate(method.items):

                if(is_un and i in cov):
                    s = f"\033[9m{item}"
                elif(i == item_ind):
                    s = f"\033[1m\033[92m{item}"
                else:
                    s = str(item)

                if(method.optional_mask[i]):
                    s = f"{s}*"
                
                if(i != len(method.items)-1):
                    s += f"\033[0m{pre} "
                meth_str += (f"{pre}{s}\033[0m")

            # meth_str = ' '.join(item_strs)
            if(method.unordered):
                meth_str = f"{meth_str}\033[0m"

            step_strs.append(f"{macro}->{meth_str}")

        return f"[{'; '.join(step_strs)}]"

    __repr__ = __str__

    @property
    def steps_as_tuple(self):
        if(not hasattr(self, '_steps_as_tuple')):
            self._steps_as_tuple = tuple(
                (a,b,c, tuple(d) if d is not None else d) for (a,b,c,d) in self.steps
            )
        return self._steps_as_tuple

    def __hash__(self):
        return hash(tuple(self.steps_as_tuple))

    def __eq__(self, other):
        return tuple(self.steps_as_tuple) == tuple(other.steps_as_tuple)

    def get_item(self):
        macro, meth_ind, item_ind, cov = self.steps[-1]
        return macro.methods[meth_ind].items[item_ind]

    def get_info(self):
        step_infos = []
        for step in self.steps:
            macro, meth_ind, item_ind, cov = step
            step_infos.append({
                "macro" : str(macro),
                "meth_ind" : meth_ind,
                "item_ind" : item_ind,
                "cov" : list(cov) if cov else None
            })
        return step_infos

    @property
    def is_internal_unordered(self):
        if(len(self.steps) == 0):
            return False
        macro, meth_ind, item_ind, cov = self.steps[-1]
        if(cov is not None and len(cov) > 0):
            return True
        return False

    @property
    def is_initial_unordered(self):
        if(len(self.steps) == 0):
            return False
        macro, meth_ind, item_ind, cov = self.steps[-1]
        if(cov is not None and len(cov) == 0):
            return True
        return False

        

def resolve_macro_prim_paths(path):
    # For a path which points to a Macro expand the path 
    #  into the primative paths for each of its methods. 
    if(path is None):
        return []
    macro, meth_ind, item_ind, cov = path[-1]
    method = macro.methods[meth_ind]
    item = method.items[item_ind]
    assert isinstance(item, MacroSkill), ""

    new_paths = []
    for j, meth in enumerate(item.methods):
        # print("NEW METH", meth, meth.unordered)
        if(meth.unordered):
            for k in range(len(meth.items)):
                new_path = ParsePath(path.steps+[(item, j, k, set())],
                            depends_on=path.depends_on)
                new_paths.append(new_path)
        else:
            new_path = ParsePath(path.steps+[(item, j, 0, None)],
                        depends_on=path.depends_on)
            new_paths.append(new_path)

    return resolve_prim_paths(new_paths)

def incr_parent(path, depends_on=[], ignore_opt_parents=False):
    # For a path return the lowest parent path with remaining symbols
    #  optionally ignore parents which are marked as optional

    k = len(path)-2
    lowest_parent = None

    macro, meth_ind, item_ind, cov = path[-1]
    method = macro.methods[meth_ind]

    while(k >= 0):
        macro, meth_ind, item_ind, cov = path[k]
        method = macro.methods[meth_ind]
        L = len(method.items)

        if(ignore_opt_parents and method.is_optional(item_ind)):
            continue

        if(cov is not None):
            new_cov = cov.union({item_ind})
            exh = all([(method.is_optional(c) or c in new_cov) for c in range(L)])
            full_exh = len(new_cov) == L
        else:
            exh = full_exh = item_ind+1 >= len(method.items)
            
        if(not full_exh):
            lowest_parent = ParsePath(path.steps[:k] + [(macro, meth_ind, item_ind+1, cov)],
                            depends_on=depends_on)

            # If not exhausted don't go to next parent
            if(not exh):
                break

        k -= 1
    return lowest_parent

def resolve_prim_paths(paths):
    # Resolve any set of paths pointing to non-terminals 
    #  or optional items into a set of paths which point
    #  to (terminal) primative skills.

    prim_paths = []
    while(len(paths) > 0):
        new_paths = []
            
        # prev_item_prims = []
        for path in paths:
            macro, meth_ind, item_ind, cov = path[-1]
            method = macro.methods[meth_ind]
            item = method.items[item_ind]
            L = len(method.items)

            # Resolve item, into possibly several divergent paths
            if(isinstance(item, MacroSkill)):
                item_prims = resolve_macro_prim_paths(path)
            elif(isinstance(item, PrimSkill)):
                item_prims = [path] 

            # print("item_prims", item, item_prims)
            prim_paths += item_prims

            # If the current path points to an optional item
            could_overflow = False
            if(method.is_optional(item_ind)):
                #  ...and the containing method is not unordered then 
                #  process the next item in the next iteration 
                if(not method.unordered):
                    if(item_ind+1 < len(method.items)):
                        new_path = ParsePath(path.steps[:-1] + [(macro, meth_ind, item_ind+1, None)],
                            depends_on=item_prims)
                        new_paths.append(new_path)
                    else:
                        could_overflow = True

                #  If unordered then the other !=item_ind options
                #   should already be accounted for. However, if
                #   all the remaining items are optional then we
                #   may still need to try incrementing the parent.
                else:
                    exh = all([(method.is_optional(c) or c in cov) for c in range(L)])
                    could_overflow = exh
                    # for k in range(L):
                    #     new_cov = cov.union({k})
                    #     if(k not in cov):
                    #         new_path = ParsePath(path.steps[:-1] + [(macro, meth_ind, k, new_cov)],
                    #             depends_on=item_prims)
                    #         new_paths.append(new_path)

                # If +1 ind would overflow then look for next parent
                if(could_overflow):
                    # TODO: Need to write a test which leverages ignore_opt_parents
                    parent_path = incr_parent(path, depends_on=item_prims)
                        # ignore_opt_parents=True)
                    new_paths.append(parent_path)

        paths = new_paths

    # Collect the unique new paths, and join the depends_on items
    d = {}
    for prim_path in prim_paths:
        if(prim_path not in d):
            d[prim_path] = set(prim_path.depends_on)
        else:
            deps = d[prim_path]
            for dep in prim_path.depends_on:
                deps.add(dep)

    out = []
    for prim_path, deps in d.items():
        prim_path = ParsePath(prim_path.steps, depends_on=list(deps))
        out.append(prim_path)

    return out



def new_paths_for_item(seq_item, prim_paths):
    new_paths = []
    did_cover = False
    for path in prim_paths:
        # print("path", path)
        macro, meth_ind, item_ind, cov = path[-1]
        method = macro.methods[meth_ind]
        L = len(method.items)
        # print("PRIM", method, cov, seq_item)

        path_exhausted = False
        
        if(cov is not None):
            # for k in range(L):
                # print(k, k not in cov, cov)
            if(item_ind not in cov):
                item = method.items[item_ind]
                # print(item, seq_item, item.overlap(seq_item))
                if(item_ind not in cov and item.overlap(seq_item) == 1.0):
                    did_cover = True
                    new_cov = cov.union({item_ind})
                    # print(cov, new_cov, len(new_cov), L)
                    if(len(new_cov) == L):
                        path_exhausted = True

                    for k in range(L):
                        if(k not in new_cov):
                            new_path = ParsePath(path.steps[:-1] + [(macro, meth_ind, k, new_cov)])
                            new_paths.append(new_path)
                    # break
        else:
            item = method.items[item_ind]
            if(item.overlap(seq_item) == 1.0):
                did_cover = True
                if(item_ind + 1 < L):
                    new_path = ParsePath(path.steps[:-1] + [(macro, meth_ind, item_ind+1, None)])
                    new_paths.append(new_path)
                else:
                    path_exhausted = True

        # print("path_exhausted", path_exhausted)
        if(path_exhausted):
            parent = incr_parent(path)
            # print("INCR PARENT", parent)
            if(parent):
                new_paths.append(parent)

    return did_cover, new_paths  

def parse_subseq(grammar, seq):
    root = grammar.root_symbols[0]

    # macro, meth_num, item_num, coverage
    cov = set() if root.methods[0].unordered else None
    path = ParsePath([(root, 0, 0, cov)])
    prim_paths = resolve_prim_paths([path])
    # print("------PARSE:", ", ".join([str(x) for x in seq]))
    seq_ind = 0
    did_cover = True
    while(seq_ind < len(seq) and len(prim_paths) > 0 and did_cover):
        seq_item = seq[seq_ind]

        # Turn all 
        print("---", seq_item, len(prim_paths))

        did_cover, new_paths = new_paths_for_item(seq_item, prim_paths)
        if(not did_cover):
            # print("DID NOT COVER")
            break
        
        prim_paths = resolve_prim_paths(new_paths)
        # print(">>", seq_item, groups_items_by_depends(prim_paths))
        # for pp in prim_paths:
        #     print(pp, pp.depends_on)
        # print()
        seq_ind += 1

    if(seq_ind == len(seq) and did_cover):
        # print("PARSE SUCCEED")
        # print(prim_paths)
        return True, prim_paths
    else:
        # print("PARSE FAILED")#, " ".join([str(x) for x in seq[:seq_ind]]), f"\033[9m{' '.join([str(x) for x in seq[seq_ind:]])}\033[0m")
        return False, []

def parse_state_from_tracker(grammar, state, preseq_tracker, prob_uid=None):

    if(prob_uid is None):
        try:
            prob_uid = preseq_tracker.resolve_prob_uid(state)
        except:
            print(">BAD PROB UID")
            return False, []


    target_uid = state.get("__uid__")

    

    if(getattr(preseq_tracker, 'curr_grammar', None) is grammar):
        # tar_info = getattr(preseq_tracker, "prim_paths", None)#.states[target_uid]
        prim_path_cache = getattr(preseq_tracker, "prim_path_cache", None)
        if(target_uid in prim_path_cache):
            return True, prim_path_cache[target_uid]

    # If grammar changed then reset prim_paths caches
    else:
        # print("NEW GRAMMAR")
        preseq_tracker.curr_grammar = grammar
        preseq_tracker.prim_path_cache = {}

    
    root = grammar.root_symbols[0]
    # macro, meth_num, item_num, coverage
    cov = set() if root.methods[0].unordered else None
    path = ParsePath([(root, 0, 0, cov)])
    prim_paths = resolve_prim_paths([path])
    

    
    if(target_uid == prob_uid):
        preseq_tracker.prim_path_cache[target_uid] = prim_paths        
        return True, prim_paths
    elif(target_uid not in preseq_tracker.states):
        # print("FAIL PARSE")
        return False, []
    
    begin_uid = prob_uid 
    begin_s_info = preseq_tracker.states[prob_uid]

    tar_info = preseq_tracker.states[target_uid]
    targ_ins = tar_info.get('ins', [])    
    if(len(targ_ins) > 0):
        # print("ARE INS")
        # prim_paths = []
        for sa in targ_ins:
            s_uid = sa.state.get("__uid__")
            if(s_uid in preseq_tracker.prim_path_cache):
                begin_uid = s_uid
                begin_s_info = preseq_tracker.states[begin_uid]
                prim_paths = preseq_tracker.prim_path_cache[begin_uid]
                # print(prim_paths)
                # print("BEGIN ", s_uid[:5])
                break
            # else:
            #     print("NOT CACHED")
            # else:
            #     print("NOT BEGIN")
    # else:
    #     print("NO INS")
                    
    next_items_paths = {sa:prim_paths for sa in begin_s_info.get('outs', [])}
    target_prim_paths = set()
    did_cover = True
    # covered_state_uids = {begin_uid}
    while(len(next_items_paths) > 0):
        _new_items_paths = {}
        # print()
        for seq_item, prim_paths in next_items_paths.items():

            # Turn all 
            

            did_cover, new_paths = new_paths_for_item(seq_item, prim_paths)
            prim_paths = resolve_prim_paths(new_paths)

            # print("---", repr(seq_item), did_cover)#, prim_paths)
            # print(prim_paths)

            if(did_cover):
                next_state = getattr(seq_item,'next_state', None)
                if(next_state is None):
                    # print("NEXT STATE NONE")
                    continue
                
                n_uid = next_state.get("__uid__")
                # print(n_uid)
                # if(n_uid in covered_state_uids):
                #     continue

                if(n_uid == target_uid):
                    # print(">>", prim_paths)
                    target_prim_paths = target_prim_paths.union(set(prim_paths))
                    # print(">>", target_prim_paths)
                else:
                    # print("---", n_uid[:5], seq_item)#, prim_paths)
                    n_info = preseq_tracker.states[n_uid]
                    # print("N_OUTS", len(n_info.get('outs', [])))
                    for n_sa in n_info.get('outs', []):
                        # next_state = getattr(sa, 'next_state', None)
                        # if(next_state is not None):
                        #     _n_uid = next_state.get("__uid__")
                        # print(prob_uid[:5], n_uid[:5])#, target_uid[:5], repr(n_sa))
                        
                        if(n_sa in _new_items_paths):
                            # print("OLD", _new_items_paths[n_sa])
                            # print("NEW", prim_paths)
                            _new_items_paths[n_sa] = _new_items_paths[n_sa].union(set(prim_paths))
                            # if(tuple(prim_paths) != tuple(_new_items_paths[n_sa])):
                            #     raise ValueError()
                        else:
                            _new_items_paths[n_sa] = set(prim_paths)

                # covered_state_uids.add(n_uid)

        if(len(_new_items_paths) == 0):
            break
        next_items_paths = _new_items_paths
        
        # for 
            
            # print(">>", seq_item, groups_items_by_depends(prim_paths))
            # for pp in prim_paths:
            #     print(pp, pp.depends_on)
            # print()
            # seq_ind += 1

    if(len(target_prim_paths) > 0):
        # print("PARSE SUCCEED")
        # print(target_prim_paths)
        preseq_tracker.prim_path_cache[target_uid] = list(target_prim_paths)
        return True, list(target_prim_paths)
    else:
        # print("PARSE FAILED")#, " ".join([str(x) for x in seq[:seq_ind]]), f"\033[9m{' '.join([str(x) for x in seq[seq_ind:]])}\033[0m")
        # print(next_items_paths)
        return False, []

    
def group_paths_by_depends(prim_paths):    
    prim_paths = [*prim_paths]
    covered = set()
    L = len(prim_paths)
    groups = []
    while(len(covered) != L):
        group = []
        for i in range(len(prim_paths)-1,-1,-1):
            path = prim_paths[i]
        # for i, path in enumerate(prim_paths):
            all_deps_okay = True
            for dep in path.depends_on:
                if(dep not in covered):
                    all_deps_okay = False

            if(all_deps_okay):
                del prim_paths[i]
                group.append(path)

        for path in group:
            covered.add(path)

        if(len(group) == 0):
            raise RuntimeError("Failed to construct item groups.")
        groups.append(group)

    return groups

def groups_items_by_depends(prim_paths):
    out = []
    groups = group_paths_by_depends(prim_paths)
    for group in groups:
        out.append([p.get_item() for p in group])
    return out
















        # print("new_seq_orders")
        # print(new_seq_orders)

        # mac_O = np.zeros((mac_L, mac_L), dtype=np.float32)        
        # for k0, grp0 in enumerate(disjoint_groups):
        #     for k1 in range(k0+1, mac_L):
        #         if(k0 == k1):
        #             continue
        #         grp1 = disjoint_groups[k1]
        #         any_Cij = False
        #         any_Cji = False
        #         max_O = 0.0
        #         for i in grp0:
        #             for j in grp1:
        #                 if(meth_C[i, j]):
        #                     any_Cij = True
        #                 if(meth_C[j, i]):
        #                     any_Cji = True
        #                 if(meth_O[i,j] > max_O):
        #                     max_O = meth_O[i,j]

        #         mac_C[k0,k1] = any_Cij
        #         mac_C[k1,k0] = any_Cji
        #         mac_O[k0,k1] = max_O
        
        # print("MAC_O")
        # print(mac_O)
        # print("MAC_C")
        # print(mac_C)



        # for i, (s0,e0) in enumerate(meth_spans):
            


        



''' Thinking Thinking 3/4

--Reorder Pass--
This pass should order primative skills such that
1) All primatives that always co-occur and are adjacent
     are contigious
2) All spans of primatives (which will probably make up
     the resulting methods) that are mutually disjoint
     and adjacent are contiguous

Steps:
1 : Produce a matrix Tc which captures elements which 
     have a minimum adjacency of 1 and always co-occur

2 : Find the ordered connected components in Tc, defined as 
    a set of primatives {s_k} = S for which all i,j in S
    C[i,j] | C[j,i] and for all s_k=i, there exists j=s_k+1
    such that A_min[i,j] = 1. If ~C[i,j] or ~C[j,i] then
    sum(C[i, :] > 0) == 1, in other words there are not 
    other primatives that 

3 :


Thinking 3/5

We need to reorder from the back such that 
we maximize:
1) The consistency for adjacency min/max of co-occuring items
2) The consistency for adjacency min/max of one-way co-occuring items
3) The overlap adjacency of contiguous items



'''
