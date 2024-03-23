import numpy as np
from itertools import permutations
from copy import copy
from .base import *


# print("HI")

def _seq_to_initial_grammar(seq):
    g = SkillGrammar()
    rhs = MethodSkill(seq)
    g.add(MacroSkill("S", rhss=[rhs]))
    return g, rhs

# print("HI", _seq_to_initial_grammar)
# ------------------------------------------------
# : Find Edits btw. two sequences 

def _merge_contiguous(edits):
    if(len(edits) == 0):
        return edits

    # print("BEF EDITS", edits)
    merged_edits = []
    contig_ins_dels = []

    pe = edits[0]
    for i in range(1,len(edits)):
        e = edits[i]
        kind, pe_kind = e[0], pe[0]

        # If edit is contigous with prev edit to merge into 
        if((kind != "unorder" and pe_kind != "unorder") # Neither are unorders
           and  pe[2] == e[1] # Contiguous
           and  kind in ("replace", "insert", "delete")
           and  pe_kind in ("replace", "insert", "delete")):

            new_kind = pe_kind
            if(kind != pe_kind):
                new_kind = "replace"
            pe = (new_kind, pe[1], e[2], pe[3]+e[3])
            #    kind == "replace" or pe_kind == "replace" # Either is replace
            # or kind == pe_kind # or are same kind
            # or kind == "delete" and pe_kind == "insert"
            # or kind == "insert" and pe_kind == "delete")): # insert and delete 

            # If delete and contiguous w/ prev (delete/replace)
            # if(kind == "delete"):
                # NOTE: Should probably delete this exception
                # Skip merge of replace, delete
                # if(kind == "delete"):
                #     merged_edits.append(pe) 
                #     pe = e
                # else:
                    # Extend deletion region (MAIN PART)
                # pe = (pe_kind, pe[1], e[2], *pe[3:])

            # If insert and contiguous w/ prev (insert/replace)
            # elif(kind == "insert"):
                # Append 
                # if(pe_kind == "insert"):
                #     # pe = (pe_kind, pe[1], pe[2]+e[2])
                # else: # pe_kind == "replace"
                # pe = (pe_kind, pe[1], pe[2], pe[3]+e[3])

            # If replace and contiguous w/ prev (insert/delete/replace)
            # elif(kind == "replace"):                
            #     ins_vals = [] if pe_kind == "delete" else pe[-1]
            #     pe = ('replace', pe[1], e[2], ins_vals+e[3])
                

        else:
            merged_edits.append(pe) 
            pe = e
    merged_edits.append(pe) 

    return merged_edits


def find_edits(alignment, subseq, item_skills, return_align=False):
    N, M = len(alignment), len(subseq)
    # seq0, seq1 = item_apps, subseq
    eq_matrix = np.zeros((N, M))
    # print(seq0, seq1)
    # if(alignment is None):
    #     for i, c1 in enumerate(seq0):
    #         for j, c2 in enumerate(seq1):
    #             if((hasattr(c1, 'skill_eq') and c1.skill_eq(c2)) or
    #                 c1 == c2):
    #                 eq_matrix[i][j] = 1
    # else:
    for i, j in enumerate(alignment):
        if(j < 0):
            continue
        
        c0, c1 = item_skills[i], subseq[j]
        # print("SKILL EQ", c0, c1, c1.skill_eq(c0))
        if((hasattr(c1, 'skill_eq') and c1.skill_eq(c0)) or
            c0 == c1):
            eq_matrix[i][j] = 1

    # print(seq0, seq1)
    # print("eq_matrix")
    # print(eq_matrix)
    edits = []
    c = 0
    s0 = s1 = 0
    e0 = e1 = 1
    spanning = False
    alignment = -1*np.ones(N, dtype=np.int64)
    for _ in range(max(N,M)):
        # print(f's0={s0} e0={e0} s1={s1} e1={e1}, c={c}')
        # Note: printing a copy with +.1 added to the span of s0:e1,s1:e1
        #  is helpful for ensuring that each iterator is changing correctly
        # eq_copy = eq_matrix.copy()
        # eq_copy[s0:e0,s1:e1] += .1
        # print(eq_copy)
        # print(e0-1, s1, e0 <= len(seq0), eq_matrix[e0-1,s1:])
        # print(s0 ,e1-1, e1 <= len(seq1), eq_matrix[s0:,e1-1])
        # print(eq_copy[s0:e0,s1:e1])

        is_del = e0 <= N and np.sum(eq_matrix[e0-1,s1:]) == 0
        is_ins = e1 <= M and np.sum(eq_matrix[s0:,e1-1]) == 0

        # print(is_del, is_ins)
        if(is_del or is_ins):
            if(is_del and is_ins):
                # print(('replace', e0-1))
                
                edits.append(('replace', e0-1, e0, [subseq[e1-1]]))
                alignment[e0-1:e0] = e1-1
            elif(is_del):
                # print("!>", ('delete', s0))
                edits.append(('delete', e0-1, e0, []))
            else:
                # print(('insert', s0))
                edits.append(('insert', e0-1, e0-1, [subseq[e1-1]]))

            # print("edit", s0, s1)
            

            if(is_del):
                s0 += not spanning
                e0 += 1
                c += spanning
            if(is_ins):
                s1 += not spanning
                e1 += 1            
            continue

        

        if(eq_matrix[s0,s1]):
            # Move on if in-place character
            # print("skip", s0)
            # print("==", s0, e0)
            alignment[s0] = s1
            s0, s1 = e0, e1
            

            if(e0 >= N or e1 >= M):
                break

            e0, e1 = e0+1, e1+1
            continue

        # Otherwise check for out order span
        if(np.sum(eq_matrix[s0:e0, s1:e1]) == e0-s0-c):
            # print(('unorder', s0, e0))
            # print(eq_matrix[s0:e0, s1:e1])
            # print(np.nonzero(eq_matrix[s0:e0, s1:e1]))
            # print(np.nonzero(eq_matrix[s0:e0, s1:e1])[1]+s1)
            # print("unorder", (s0,e0), np.nonzero(eq_matrix[s0:e0, s1:e1])[1]+s1)
            for k, row in enumerate(eq_matrix[s0:e0, s1:e1]):
                nz = np.nonzero(row)[0]
                if(len(nz) > 0):
                    alignment[s0+k] = nz[0]+s1

            edits.append(('unorder', s0, e0))
            s0, s1 = e0, e1
            spanning = False
            c = 0
        else:
            spanning = True

        if(e0 >= N or e1 >= M):
            break

        e0, e1 = e0+1, e1+1

    if(e0 < N):
        # for i in range(e0,len(seq0)):
        edits.append(("delete", e0, N, []))

    if(e1 < M):
        # print("THIS", seq1, s0, seq1[e1:])
        edits.append(("insert", s0, s0, list(subseq[s1:])))

    edits = sorted(edits, key=lambda x: x[1])
    
    # print("E", edits)
    # Merge contiguous replace and insert/delete edits
    filtered_edits = _merge_contiguous(edits)

    # print(seq1, edits)
    # print(alignment)
    # assert -1 not in alignment
    if(return_align):
        return filtered_edits, alignment
    else:
    # print(filtered_edits)
        return filtered_edits


# --------------------------------------------
# : Modify RHS from Edits

def apply_item_edits(meth_items, meth_opt_mask, edits):
    # unorder_edits = [edit for edit in edits if edit[0]=='delete']
    delete_edits = [edit for edit in edits if edit[0]=='delete']
    ins_repl_edits = [edit for edit in edits if edit[0] in ['insert', 'replace','lift']]

    # print("RAP", edits)
    # For each delete edit make the corresponding symbol optional.
    new_opt_mask = meth_opt_mask.copy()
    for edit in delete_edits:
        for i in range(edit[1],edit[2]):
            new_opt_mask[i] = True

    # items = rhs.items
    if(len(ins_repl_edits) > 0):
        items_opt = list(zip(meth_items, new_opt_mask))

        for edit in reversed(ins_repl_edits):
            kind = edit[0]

        # For each insert edit add an optional symbol. 
            if(kind == "insert"):
                items_opt = items_opt[:edit[1]] + [(e,1) for e in edit[3]] + items_opt[edit[1]:]
        #  For each replace make the appropriate replacement with
        #  non-optional symbols. Apply in reversed order (should already
        #   be ordered by index) to prevent index invalidation
            elif(kind == "replace"):
                new_syms = [(symb, 0) for symb in edit[3]]
                items_opt = items_opt[:edit[1]] + new_syms + items_opt[edit[2]:]
            elif(kind == "lift"):
                # print("START LIFT", items_opt)
                methods = items_opt[edit[1]][0].methods
                # print(items_opt[edit[1]][0], methods)
                assert len(methods) == 1
                meth0_items = methods[0].items
                meth0_opt = methods[0].optional_mask
                meth0_items_opt = list(zip(meth0_items, meth0_opt))
                # print(meth0_items_opt)
                items_opt = items_opt[:edit[1]] + meth0_items_opt + items_opt[edit[2]:]
                # print("END LIFT", items_opt)

        new_opt_mask = np.array([x[1] for x in items_opt], dtype=np.bool_)
        return [x[0] for x in items_opt], new_opt_mask
    else:
        return meth_items, new_opt_mask
    # if unorder is None:
    #     unorder = rhs.unordered

    # print("ITEMS", items)
    # new_meth = MethodSkill(items, macro=rhs.macro, unordered=unorder, optionals=optional_mask)
    # if(str(items) == "[a2, A, d]"):
    #     raise ValueError()
    # return new_meth

# --------------------------------------------
# : Grammar Change btw. RHS + sequence

def _subsume_unorder_overlaps(non_unorder_edits, unorder_substs):
    ''' Find edits that overlap with an unorder edit and apply the 
        edits directly on the new symbol associated with the unordering.'''
    i = 0
    filtered = []
    subsumed_edits = {}
    for edit in non_unorder_edits:
        was_subsumed = False
        while(i < len(unorder_substs)):
            (macro, methods, (s,e)) = unorder_substs[i]
            ind = edit[1]

            # Edit Subsumed by Substitution Case
            if(ind >= s and ind < e):
                sub_edits = subsumed_edits.get(i,[])

                # Make the edit relative to the subuming RHS
                e = copy(list(edit))
                e[1] -= s
                if(e[0] == "replace"):
                    e[2] -= s

                sub_edits.append(tuple(e))
                subsumed_edits[i] = sub_edits
                was_subsumed = True
                break # Subsumed so get next edit
            elif(ind >= e): 
                i += 1
                continue # Edit is right of split i get split i+1
            else: 
                break # Edit is left of split i get next edit

        if(not was_subsumed):
            filtered.append(edit)

    # Apply the edits
    for split_ind, edits in subsumed_edits.items():
        (macro, methods, (s,e)) = unorder_substs[split_ind]
        method = methods[-1] # Last method is always the new one

        items, opt_mask = apply_item_edits(
                    method.items, method.optional_mask, edits)

        methods[-1] = MethodSkill(items,
                          macro=macro,
                          unordered=True,
                          optionals=opt_mask)

    # for (sym, rhss, (s,e)) in [*unorder_substs, *disjoin_substs]:
    #     print(sym, '->', " | ".join([str(rhs) for rhs in rhss]))
    return filtered
    # non_unorder_edits = filtered



def organize_edits(meth_items, meth_opt_mask,
                     edits, symbol_factory=None):
    ''' Preprocess a set of edits into a set of method changes which 
        can consist of full unordering, substitutions, edits to the 
        new method generated after substitutions, and disjunctions 
        added to the method's parent macro.
    '''
    # print("--EDITS", edits)

    if(symbol_factory is None):
        symbol_factory = SymbolFactory(ord("T"))

    # Look for an unorder edit that spans the whole Method
    full_unorder = False
    # print(edits)
    has_ins = any([edit[0] in ('insert', 'replace') for edit in edits])
    if(not has_ins):
        for i, edit in enumerate(edits):
            if(edit[0] == 'unorder' and edit[1] == 0 and edit[2] == len(meth_items)):
                full_unorder = True
                edits.pop(i)
                break

    any_new_sym = any([edit[0] in ["unorder", "replace"] for edit in edits])
    prev_end = 0 if any_new_sym else None

    # Make new symbols for each 'unorder' and 'replace' edit
    unorder_substs = [] # [(sym, [rhs,...], span),...]
    replace_substs = [] # [(sym, [rhs,...], span),...]
    disjoins = [] # [(sym, [rhs,...], span),...]
    unorder_edits = []
    non_unorder_edits = []
    new_macros = []
    lifted_macros = []
    for edit in edits:
        kind = edit[0]
        s,e = edit[1], edit[2]
        # For each edit which would introduce a new Macro
        if(kind in ["unorder", "replace"]):
            

            # Fill any holes preceeding this edit
            # if(prev_end is not None and prev_end < s):
            #     new_sym = Sym(symbol_factory())
            #     method0 = MethodSkill(seq1[prev_end:s], new_sym)
            #     disjoin_substs.append((new_sym, [method0], (prev_end,s)))
            #     non_unorder_edits.append(('replace', prev_end, s, [new_sym]))
            # prev_end = e

            

            # UnorderSubst Case 
            if(kind == "unorder"):
                # print("--UNORDER SUBST--")
                new_macro = MacroSkill(symbol_factory())
                method0 = MethodSkill(meth_items[s:e],
                                      macro=new_macro,
                                      unordered=True,
                                      optionals=meth_opt_mask[s:e])
                new_macro.add_method(method0)
                unorder_substs.append((new_macro, [method0], (s,e)))
                unorder_edits.append(('replace', s, e, [new_macro]))

                # print(rhs0)


            # DisjoinSubst Case 
            elif(kind == "replace"):
                # new_seq = edit[3] if isinstance(edit[3], list) else [edit[3]]
                
                # rhs0 = RHS(seq1[s:e], new_sym, unordered=rhs.unordered)
                method1 = MethodSkill(edit[3])
                # new_macro.add_method(method1)
                # print("0:", rhs0)
                # print("1:", rhs1)

                # If replacement spans the RHS then add as disjunction
                if(e-s == len(meth_items)):
                    # print("--ADD--", rhs.symbol, [rhs0, rhs1])
                    disjoins.append(method1) 
                    # method.macro.add_method(method1)

                # Otherwise substitute subsequence with new variable
                else:
                    new_macro = MacroSkill(symbol_factory())
                    # print("HERE")
                    method0 = MethodSkill(meth_items[s:e], macro=new_macro, optionals=meth_opt_mask[s:e])
                    new_macro.add_method(method0)
                    new_macro.add_method(method1)
                    # method0.macro = new_macro
                    # print("--SUBST--", method.symbol, [method0, method1])
                    replace_substs.append((new_macro, [method0, method1], (s,e)))
                    # Swap the new symbol in for the RHS of the replace edit 
                    edit = ('replace', s, e, [new_macro])
                    non_unorder_edits.append(('replace', s, e, [new_macro]))
        else:
            # Insert and Delete Edits
            non_unorder_edits.append(edit)
            if(kind == "lift"):
                for l_macro in meth_items[s:e]:
                    lifted_macros.append(l_macro)

    # Fill any holes after these edits
    # print("**", any_new_sym, prev_end, len(seq1))
    # if(any_new_sym and prev_end < len(seq1)):
    #     print("FILL", seq1[prev_end:len(seq1)])
    #     new_sym = Sym(symbol_factory())
    #     rhs0 = RHS(seq1[prev_end:len(seq1)], new_sym)
    #     s, e = prev_end, len(seq1)
    #     disjoin_substs.append((new_sym, [rhs0], (s, e)))
    #     non_unorder_edits.append(('replace', s, e, [new_sym]))

    
    # Apply any edits subsumed by an unorder substitution
    if(len(unorder_substs) > 0):
        non_unorder_edits =_subsume_unorder_overlaps(non_unorder_edits, unorder_substs)

    # Add a replace edit for each new symbol in RHS
    parent_edits = [*unorder_edits, *non_unorder_edits]
    substs = [*unorder_substs, *replace_substs]
    # for (sym, rhss, (s,e)) in unorder_substs:
    #     parent_edits.append(('replace', s, e, [sym]))
    # for (sym, rhss, (s,e)) in disjoin_substs:
    #     parent_edits.append(('replace', s, e, [sym]))
    parent_edits = sorted(parent_edits, key=lambda x:x[1])
    return (full_unorder, substs, parent_edits, disjoins, lifted_macros)

def generalize_method_items(items, subseq, edits, alignment):
    # print("GENERALIZE ITEMS:", items)
    # print("subseq:", subseq)

    # Alignment with -1 for bit that become part of disjoint RHSs
    merge_alignment = alignment.copy()
    # Alignment with -1 for bit that are replaced with non-terminals
    retain_alignment = alignment.copy()
    # The amount by which other conditions need be shifted to be retained
    shift = np.zeros(len(items), dtype=np.int64)
    for edit in reversed(edits):
        kind = edit[0]
        if(kind == "replace"):
            merge_alignment[edit[1]:edit[2]] = -1
            retain_alignment[edit[1]:edit[2]] = -1
            shift[edit[2]:] += (edit[2]-edit[1])-1
        elif(kind == "unorder"):
            shift[edit[2]:] += (edit[2]-edit[1])-1
            retain_alignment[edit[1]:edit[2]] = -1

    # print("merge_alignment:", merge_alignment)
    # print("retain_alignment:", retain_alignment)
   
    _subseq = subseq.items if isinstance(subseq, MethodSkill) else subseq
    # print("_subseq", _subseq)
    new_items = [*items]
    # print("vv", [(len(x.skill_apps),x) for x in new_items])
    for i, (ind, old_item) in enumerate(zip(merge_alignment, items)):
        # print(i, "##", old_item, _subseq[ind], subseq[ind].prob_uid)#, type(old_item), type(_subseq[ind]), type(subseq[ind]))

        if(isinstance(old_item, SkillBase) and ind >= 0):
            # print("??", _subseq, ind)
            new_items[i] = old_item.merge(_subseq[ind])
    # print("vv", [(len(x.skill_apps),x) for x in new_items])
    return new_items
   

from cre.utils import PrintElapse

class SeqParse:
    def __init__(self, method=None, method_app=None, span=(0,1), subseq=[], alignment=None, gen_penl=0.0):
        if(isinstance(method, SkillApp)):
            self.method_app = method
            self.method = method.skill            
        else:
            self.method = method
            self.method_app = method_app

        self.span = span
        self.subseq = subseq
        self.alignment = np.arange(len(subseq)) if alignment is None else alignment
        # self.lit_matches = lit_matches
        self.gen_penl = gen_penl
        self.cost = gen_penl

    def recalc_cost(self):
        self.cost = self.gen_penl

class RHSChange(SeqParse):
    ''' Instantiates a candidate grammar change and its edit cost and span.
        A single change instance can simultaneously represents multiple kinds
        of changes:
         0. No-Change Corresponds to a Full RHS Match
         1. Insert (corresponds to insert edits... extensions always made optional)
         2. Unorder (occurs if a swap edit spans the whole target)
         3. Optional (corresponds to delete edits)
         4. UnorderSplit (replaces part of a RHS with a new symbol with an unordered RHS. Occurs
                when a subset of a rhs is found to be unordered)
         5. DisjoinSplit (replaces part of a RHS with a new symbol with a disjunction S -> A | B. 
                occurs when a replace edit is encountered)

         6. UnorderMerge (Merges a )
         7. DisjoinMerge (Replaces every instance of a disjunction S-> A | B with a sequence of 
                optionals A*B*. Occurs if multiple options of a disjoint non-terminal occur in sequence)
    '''
    def __init__(self, method=None, method_app=None, span=(0,1),
             subseq=[], edits=[], alignment=None, scores=None, gen_penl=0.0, nomatch=False,
             b_alignment=None, b_span=None):
        
        super().__init__(method, method_app, span, subseq, alignment, gen_penl)


        # self.span = span
        # self.subseq = subseq
        self.edits = edits
        # self.alignment = np.arange(len(subseq)) if alignment is None else alignment
        # self.lit_matches = lit_matches
        # self.gen_penl = gen_penl
        self.nomatch = nomatch
        self.scores = scores
        self.b_alignment = b_alignment
        self.b_span = b_span
        self.recalc_cost()

    def __str__(self):
        meth_str = self.method.__str__(bracket=False)
        return f"{self.macro_str}->{meth_str} {'nomatch' if self.nomatch else self.edits}"

    @property
    def macro_str(self):
        return str(self.method.macro if self.method else None)

    __repr__ = __str__

    def __copy__(self):
        return RHSChange(self.method, self.method_app, self.span, self.subseq, self.edits, self.alignment,
                          self.lit_matches, self.gen_penl, self.nomatch)

    def recalc_cost(self):
        if(self.nomatch):
            cost = self.span[1]-self.span[0]
        else:
            cost = 0.0
            for edit in self.edits:
                kind = edit[0]
                if(kind == "insert"):
                    cost += 1.0 * len(edit[3])
                elif(kind == "delete"):
                    cost += 1.0 * (edit[2]-edit[1])
                elif(kind == "replace"):
                    cost += .5 * (edit[2]-edit[1]) + .5*len(edit[3])
                elif(kind == "unorder"):
                    # print(edit)
                    cost += .2*(edit[2]-edit[1])
            seq_len = self.span[1]-self.span[0]
            # cost /= min(len(self.rhs.items), seq_len)
        self.cost = self.gen_penl + cost

    
    # def ensure_changes_computed(self, symbol_factory=None):
    #     if(not hasattr(self, 'parent_edits')):
    #         full_unorder, unorder_substs, disjoin_substs, parent_edits = \
    #             organize_edits(self.rhs, self.edits, symbol_factory)
    #         self.full_unorder = full_unorder
    #         self.unorder_substs = unorder_substs
    #         self.disjoin_substs = disjoin_substs
    #         self.parent_edits = parent_edits

    def apply(self, grammar=None, symbol_factory=None):
        # print(self.alignment)
        # print("P", self.rhs, self.parent_edits)
        old_meth = self.method #if(old_meth is None) else old_meth
        old_macro = old_meth.macro
        # print("^^",old_meth.items, self.subseq, self.alignment)
        
        old_items = []
        for old_item in old_meth.items:
            if(isinstance(old_item, MacroSkill)):
                old_items.append(grammar.macro_skills.get(old_item._id))
            else:
                old_items.append(old_item)
        # print("LOP", self.subseq)
        # if(len(self.subseq) == 3 and isinstance(self.subseq[1].skill, MacroSkill)):
            # print("MROP", old_items[1].methods)
            # print("ROP", self.subseq[1].skill.methods)
        new_items = generalize_method_items(old_items,#old_meth.items,
                        self.subseq, self.edits, self.alignment)

        # print("EDITS", self.edits)

        methods = [*old_macro.methods]
        meth_ind = methods.index(old_meth)
        # print(methods)
        # print("meth_ind", meth_ind)
        if(any([e[0] == 'erase' for e in self.edits])):
            del methods[meth_ind]
            inserted_macros = []
            new_meth = None
        else:
            full_unorder, substs, parent_edits, disjoins, lifted_macros = \
                organize_edits(new_items, old_meth.optional_mask,
                    self.edits, symbol_factory)


            # self.ensure_changes_computed(symbol_factory)
            # print("NEW RHS", new_meth, self.full_unorder)
            # print("P EDITS", self.parent_edits)

            items, opt_mask = apply_item_edits(new_items,
                     old_meth.optional_mask, parent_edits)

            # for item in items:
            #     item._update()
            # items, opt_mask = apply_item_edits(
            #             method.items, method.optional_mask, edits)
            # print(":", items, opt_mask)
            # print(items)
            new_meth = MethodSkill(items,
                              unordered=full_unorder or old_meth.unordered,
                              optionals=opt_mask)

            methods[meth_ind] = new_meth
            methods += disjoins

            inserted_macros = [x[0] for x in substs]

        new_macro = MacroSkill(old_meth.macro._id, methods)

        if(grammar is None):
            return new_macro, new_meth, inserted_macros
        else:
            # grammar = grammar.__copy__()
            grammar.replace(old_macro, new_macro)
            for macro in lifted_macros:
                grammar.remove(macro)
            for macro in inserted_macros:
                # print("add", macro)
                grammar.add(macro)
            # print("grammar skills", grammar.macro_skills)
            return grammar, new_macro, new_meth, inserted_macros

class MergeChange(SeqParse):
    ''' 
    '''
    def __init__(self, macro_app, methods, change_parts, span=(0,1),
                subseq=[], alignment=None, edits=[], gen_penl=0.0, nomatch=False):

        self.macro_app = macro_app
        self.methods = methods
        self.edits = edits
        # Should be of form [...(seq_part_span, meth_edits)]
        #  where meth_edits has form [...(method_ind, meth_span, alignment, edits)]
        self.change_parts = change_parts
        self.nomatch = nomatch
        super().__init__(None, None, span, subseq, alignment, gen_penl)
        self.recalc_cost()

        self.macro_map = {}
        for k, method in enumerate(self.methods):
            old_macro = method.macro
            arr = self.macro_map.get(old_macro,[])
            arr.append(old_macro.methods.index(method))
            self.macro_map[old_macro] = arr

        # print(unaffected_methods)
        # If there is any reason that the new_macro
        self.should_lift_macro = False
        if(len(self.macro_map) == 1):
            old_macro = list(self.macro_map)[0]
            meth_inds = self.macro_map[old_macro]
            unaffected_methods = [m for i, m in enumerate(old_macro.methods) if i not in meth_inds]    
            if(len(unaffected_methods) == 0):
                self.should_lift_macro = True

        # self._fill_internal_holes()

    # def _fill_internal_holes(self):
    #     # May be unecessary
    #     print("FILL INTERNAL")

    #     # Make a tempory parse trees, one for each partition so we can  
    #     #  resuse _make_seq_and_holes() and _fill_parse_holes() here        

    #     parse_tree_seq = []
    #     for (sp, ep, main_k), meth_mods in self.change_parts:
    #         for k, meth_span, alignment, edits in meth_mods:
    #             if(k == main_k):
    #         # span=(0,1),
    #         #  subseq=[], edits=[], alignment=None,
    #                 rc = RHSChange(None, span=(sp,ep), 
    #                     subseq=self.subseq, edits=edits, alignment=alignment)
    #                 pt = ParseTree(rc)
    #                 parse_tree_seq.append(pt)
    #                 print("||", self.subseq[sp:ep], (sp,ep), edits)


    #     new_trees = _make_seq_and_holes(parse_tree_seq, self.subseq,
    #                      make_new_seq=False)
    #     for pt in new_trees:
    #         print(pt)        
    #     _fill_parse_holes(new_trees)

    #     for pt in new_trees:
    #         if(not isinstance(pt, ParseHole)):
    #             print(pt.change)



    @property
    def macro_str(self):
        return ",".join([str(x) for x in self.macro_map])

    def recalc_cost(self):
        self.cost = 0.0

    def _make_merged_method(self, symbol_factory=None):
        # method_item_sets = [[None]*len(meth)]
        part_macros = []
        part_items_opt = []
        all_methods = []
        all_lifted_macros = []
        for (sp, ep, main_k), meth_mods in self.change_parts:
            part_macro_id = symbol_factory()
            
            # print(self.subseq, (sp, ep), self.subseq[sp:ep])
            part_seq = self.subseq[sp:ep]

            # If the methods being merged did not overlap
            #  with some subseqence then insert it.
            if(len(meth_mods) == 1):
                k, meth_span, alignment, edits = meth_mods[0]
                if(('unorder', 0, ep-sp) not in edits):
                    # print("WOOP", part_seq, len(meth_mods), edits)
                    part_items_opt += [(sa, True) for sa in part_seq]
                    continue

            # Otherwise disjoin subsets of methods associated
            #  with this partition.
            part_methods = []
            for k, meth_span, alignment, edits in meth_mods:
                # print("<", k, meth_span, alignment, edits)
                # if(meth_span is not None):
                (ms,me) = meth_span
                old_meth = self.methods[k]
                old_meth_items = old_meth.items[ms:me]
                alignment[alignment>=0] -= sp

                # print("++", ms,me, old_meth_items, alignment)
                # Only generalize the method in the 
                #  partion's dijunct which aligned with subseq
                if(k == main_k):
                    new_items = generalize_method_items(old_meth_items,
                                    part_seq, edits, alignment)
                else:
                    new_items = old_meth_items

                opt_mask = old_meth.optional_mask[ms:me]
                full_unorder, substs, parent_edits, disjoins, lifted_macros = \
                    organize_edits(new_items, opt_mask,
                        edits, symbol_factory)

                all_lifted_macros += lifted_macros

                items, opt_mask = apply_item_edits(new_items,
                     opt_mask, parent_edits)

                new_meth = MethodSkill(items,
                              unordered=full_unorder or old_meth.unordered,
                              optionals=opt_mask)

                part_methods.append(new_meth)
                # else:
            # print("PART", part_macro_id, part_methods)
            # print(f"########## {part_macro_id}  #############")
            part_macro = MacroSkill(part_macro_id, part_methods)
            # print(part_macro)
            # print("##########################")
            part_macros.append(part_macro)
            part_items_opt.append((part_macro,False))
            all_methods += part_methods

        
        part_items = []
        part_opt_mask = np.zeros(len(part_items_opt), dtype=np.bool_)
        for i, (item, is_opt) in enumerate(part_items_opt):
            part_items.append(item)
            part_opt_mask[i] = is_opt
        # print("part_items", part_items)
        print("&&&&&&&&&&&&&&&&&&&&&&&&&&&")
        print(self.edits)
        part_items, part_opt_mask = apply_item_edits(part_items,
                part_opt_mask, self.edits)
        self.merged_method = MethodSkill(part_items, optionals=part_opt_mask)
        print("&&&&&&&&&&&&&&&&&&&&&&&&&&&")
        self.part_macros = part_macros
        self.all_methods = all_methods
        self.all_lifted_macros = all_lifted_macros
        return self.merged_method

    def _make_calling_macros(self):
        new_macros = []
        for old_macro, meth_inds in self.macro_map.items():
            unaffected_methods = [m for i, m in enumerate(old_macro.methods) if i not in meth_inds]

            # print(unaffected_methods)
            # If there is any reason that the new_macro
            if(len(self.macro_map) > 1 or
               len(unaffected_methods) > 0 or
                self.merged_method.unordered):
                print("SHOULD NOT SIMPLIFY")    
            else:
                print("SHOULD SIMPLIFY")
            new_macro = MacroSkill(old_macro._id,
                 unaffected_methods+[self.merged_method])
            new_macros.append(new_macro)
        self.new_macros = new_macros


    def apply(self, grammar, symbol_factory=None):
        
        merged_method = self._make_merged_method(symbol_factory)
        self._make_calling_macros()
                

        if(grammar is None):
            return self.new_macros, self.all_methods, self.part_macros
        else:
            for old_macro, new_macro in zip(self.macro_map, self.new_macros):
                grammar.replace(old_macro, new_macro)

            for macro in self.all_lifted_macros:
                grammar.remove(macro)
            for macro in self.part_macros:
                # print("add", macro)
                grammar.add(macro)
            # print("grammar skills", grammar.macro_skills)
            # print(grammar)
            return grammar, self.new_macros, self.all_methods, self.part_macros
        # for 

    def __str__(self):
        m_strs = "+".join([str(m) for m in self.methods])
        macro = self.macro_app.skill
        return f"{macro}->{m_strs} edits={self.edits}"
        # for method in self.methods:
        #     s += str(met)



def _align_rels(rel_lits, alignment):
    aligned_rel_lits = []
    for r, o0, o1, a0, a1 in rel_lits:
        o0 = alignment[o0]
        o1 = alignment[o1]
        if(o0 != -1 and o1 != -1):
            aligned_rel_lits.append((r, o0, o1, a0, a1))
        else:
            aligned_rel_lits.append(None)
            # pres[o0] = 1
            # pres[o1] = 1
    return aligned_rel_lits

def rel_lit_match(rel_lit, seq):
    r, o0, o1, a0, a1 = rel_lit
    item0 = seq[o0]
    item1 = seq[o1]
    if(hasattr(item0, 'args') and hasattr(item1, 'args') and
        a0 < len(item0.args) and a1 < len(item1.args)):
        arg0 = item0.args[a0]
        arg1 = item1.args[a1]
        rel = arg0.get_relation(arg1)
        return r == rel
    return False


def _check_rel_lit_matches(rhs, subseq, alignment):
    # print(rhs, rhs.rel_lits)

    # a_lst = list(alignment)
    # inv_align = -np.ones(len(subseq), dtype=np.int64)
    # for i,ind in enumerate(alignment):
    #     inv_align[ind] = i
    # inv_align = np.array(inv_align, dtype=np.int64)
    # print("align", alignment)
    # print("inv_align", inv_align)

    # print([np.where(alignment==ind)[0][0] for ind in range(len(alignment))])
    # alignment = np.array([np.where(alignment==ind)[0] for ind in range(len(alignment))], dtype=np.int64)
    # rel_lits = rhs.rel_lits
    # print("BEF", rhs.rel_lits)
    rel_lits = _align_rels(rhs.rel_lits, alignment)
    # print("AFT", rel_lits)
    match_inds = []
    total = np.ones(len(subseq), dtype=np.int64)
    match_w = np.ones(len(subseq), dtype=np.int64)
    # _a = alignment[np.where(alignment!=-1)]
    # match_w[_a] = 0
    # match_w[np.where(alignment==-1)] = 0
    # print("ALIGN", alignment)
    if(isinstance(subseq, MethodSkill)):
        # print("A")
        for i, rel_lit in enumerate(rel_lits):
            if(rel_lit is None): continue
            r, o0, o1, a0, a1 = rel_lit
            
            # o0, o1 = alignment[o0], alignment[o1]
            total[o0] += 1
            total[o1] += 1    
            if((r, o0, o1, a0, a1) in subseq.rel_lits):
                match_inds.append(i)
                match_w[o0] += 1
                match_w[o1] += 1
    else:
        # print("B")
        for i, rel_lit in enumerate(rel_lits):
            if(rel_lit is None): continue
            r, o0, o1, a0, a1 = rel_lit
            # o0, o1 = alignment[o0], alignment[o1]
            total[o0] += 1
            total[o1] += 1
            if(rel_lit_match(rel_lit, subseq)):
                match_inds.append(i)
                match_w[o0] += 1
                match_w[o1] += 1
    # print(match_w, total)
    # print(alignment, rhs.optional_mask)
    gen_penl = np.sum((alignment == -1) & ~rhs.optional_mask) + len(subseq) - np.sum(match_w / total)
    return match_inds, gen_penl


def find_rhs_change(method_app, subseq, alignment=None, span=None,
                        seq=None, scores=None, seq_trees=None,
                        b_alignment=None, b_span=None,):
    # print(f"changes({rhs.symbol}->{''.join([str(x) for x in rhs.items])}, {seq2})")

    # method_app, item_apps = apps_pair
    method = method_app.skill
    


    item_skills = [x.skill for x in method_app.child_apps]
    # print(seq1, seq2)
    # print("OLD Alignment", alignment)
    # print("S", subseq)
    edits = find_edits(alignment, subseq, item_skills)#, return_align=True)
    # print("EDITS", method_app, alignment, span, edits)

    gen_penl = 0.0
    # lit_matches, gen_penl = _check_rel_lit_matches(rhs, subseq, alignment)
    # print("align", seq1, seq2, edits, alignment)

    # Filter out any edits that are redundant with 
    #  the exiting unorder / optionals of the RHS
    filtered_edits = []
    for edit in edits:
        kind = edit[0]
        if(method.unordered and kind == "unorder"):
            continue
        if(kind == "delete" and method.is_optional(edit[1])):
            continue
        filtered_edits.append(edit)

    # print("SST", seq_trees)
    if(seq_trees):
        for i, pt in enumerate(seq_trees):            
            if(getattr(pt, "change", None) and 
                getattr(pt.change, "should_lift_macro", False)):

                align_inds = np.nonzero(alignment==i)[0]
                # print("MIGHT LIFT", alignment, i, align_inds)
                if(len(align_inds) > 0):
                    # ind = alignment[i
                    ind = align_inds[0]
                    # print("DID LIFT!", alignment, filtered_edits, ind)
                    filtered_edits.append(("lift", ind, ind+1))


    # match = method_match_tree.matches[0]
     # = method_match_tree.matches[0]
    
    # method_app = SkillApp(method, match, state=subseq[0].state)
    # print()
    # print("S2", subseq)
    # print(method_app)
    change = RHSChange(method_app, span=span, subseq=subseq, 
                    edits=filtered_edits, alignment=alignment,
                    scores=scores, gen_penl=gen_penl,
                    b_alignment=b_alignment, b_span=b_span)
    
    return change
    # g = Grammar()
    # g.add_production("S", rhs)
    # new_g, new_meth = change.apply(g)
    # print(g)

    # return new_meth

# def calc_align_scores(pattern, sequence):
#     l0, l1 = len(pattern), len(sequence)
#     scores = np.zeros((l0, l1), dtype=np.float64)    
#     for i, s0 in enumerate(pattern):
#         for j, s1 in enumerate(sequence):
#             if(hasattr(s0,'overlap')):
#                 scores[i][j] = s0.overlap(s1)
#             else:
#                 scores[i][j] = s0 == s1
#     return scores

# def calc_overlap(pattern, sequence, skip_penalty=0.1):
#     print("THIS")
#     scores = calc_align_scores(pattern, sequence)
#     # print(scores)
#     alignment = np.argmax(scores, axis=1)
#     # assert(len(alignment) == len(pattern))

#     mx = np.max(scores, axis=1)
#     # assert(len(mx) == len(pattern))

#     alignment[mx==0.0] = -1
    
#     mx = [-skip_penalty if x == 0 else x for x in mx]
#     overlap = np.sum(mx)
#     if(len(sequence) > len(pattern)):
#         overlap -= skip_penalty*(len(sequence)-len(pattern))
#     return max(overlap,0), alignment#/(l0)#+abs(l0-l1))


def greedy_span_rcs(rcs):
    if(len(rcs) == 0):
        return rcs

    rcs = sorted(rcs, key=lambda x:x.cost)

    best_rcs = [rcs.pop(0)]
    for rc in rcs:
        rs, re = rc.span
        overlaps = False
        for x in best_rcs:
            xs, xe = x.span
            if(not (re <= xs or rs >= xe)):
                overlaps = True
                break
        if(not overlaps):
            best_rcs.append(rc)
    return best_rcs


def get_best_changes(grammar, seq, seq_trees, method_match_trees, score_thresh=.7):
    # if(rhss is None):
    #     rhss = []
    #     for _id, ms in grammar.macro_skills.items():
    #         for method in ms.methods:
    #             rhss.append(method)

    # align_matricies = []
    # exact_matches = [None]*len(seq)
    best_changes = []
    alignment_sets = []
    for j, method_match_tree in enumerate(method_match_trees):
        item_trees = method_match_tree.child_trees[0]
        score_matrix, item_m_skill_apps = calc_align_score_matrix(item_trees, seq)
        scores, alignment, span = align_greedy(score_matrix, .001)
        alignment_sets.append((scores, alignment, span, item_m_skill_apps))

    seq_best_scores = np.zeros(len(seq), dtype=np.float64)
    seq_best_inds = [(-1,-1)]*len(seq)
    # for i in range(len(seq)):
    for j, (scores, alignment, span, _) in enumerate(alignment_sets):
        for k in range(len(alignment)):
            i = alignment[k]
            if(i >= 0):
                score = scores[k]
                if(score > seq_best_scores[i]):
                    seq_best_scores[i] = score
                    seq_best_inds[i] = (j, k)

    print("seq_best_inds",seq_best_inds)
    for j, (scores, alignment, span, item_m_skill_apps) in enumerate(alignment_sets):
        method_match_tree = method_match_trees[j]
        b_alignment, b_span = alignment.copy(), span
        print(method_match_tree.parent.skill, "->", method_match_tree.skill)
        print("PREV", scores, alignment, span)
        for k in range(len(alignment)):
            i = alignment[k]
            (_j, _k) = seq_best_inds[i]
            if( scores[k] < score_thresh or 
                method_match_trees[_j].parent is not method_match_trees[_j].parent and
                seq_best_inds[i] != (j,k)):
                alignment[k] = -1
        seq_inds = alignment[alignment!=-1]
        print(alignment, seq_inds)
        if(len(seq_inds) > 0):
            span = (min(seq_inds), max(seq_inds)+1)
        else:
            continue

        
        # alignment_sets[j] = (scores, alignment, span)

        print("NEW ", scores, alignment, span)



        # align_matricies.append((score_matrix, best_item_skill_apps))

        # print(scores)
        # print(np.nonzero(scores==1.0), len(seq))
        # for k,i in zip(*np.nonzero(score_matrix==1.0)):
        #     exact_matches[i] = (j,k)
    # print("----++++-----")
    # print("$$$$$$$$$$$$$$$")
    # print(seq)
    # print(exact_matches)
    # best_changes = []
    # for j, (score_matrix, item_m_skill_apps) in enumerate(align_matricies):
    #     method_match_tree = method_match_trees[j]
        # print()
        # print(score_matrix)
        # Zero out any partial matches to items which have exact matches
        # for i, tup in enumerate(exact_matches):
        #     if(tup is None):
        #         continue
        #     (_j, k) = tup
        #     if(j != _j):
        #         print("DID THIS", j, i)
                # score_matrix[:, i] = 0

        

        # print(item_trees, ">>", scores, alignment, span)
        # print("OLD SPAN", span)
        # for i in range(span[0],span[1]):
        #     if(scores[i] == 1):
        #         break
        #     if(scores[i] < 1 and getattr(seq[i], "exact_match", None)):
        #         span = (i, span[1])
        #     else:
        #         break
        # print("NEW SPAN", span)
        # for i in range(span[1]-1, span[0]-1, -1):
        #     print(i)
        #     if(scores[i] == 1):
        #         break
        #     if(scores[i] < 1 and getattr(seq[i], "exact_match", None)):
        #         print(scores[i], getattr(seq[i], "exact_match", None))
        #         span = (span[0], i+1)
        #     else:
        #         break
        # print("NEW SPAN", span, method_match_tree.skill)


        # if(np.sum(scores) == 0):
        #     continue
        # if(span == 0)
        
        # print("->")
        # print(scores)

    # print("$$$$$$$$$$$$$$$")

    
    # for method_match_tree in method_match_trees:

        # method_app, item_apps = app_pair
        # L = len(item_apps)
        # method = method_app.skill

        # print("TTT", [type(x) for x in item_apps])
        # print(item_apps)
        # print("-", [(x.skill, id(x), [y.id for y in x.match]) for x in seq])
        # scores, alignment, span, item_m_skill_apps = best_alignment(method_match_tree, seq)
        # overlap = np.sum(scores)


        # print("&&", method_match_tree.skill, alignment, subseq)
        # method_app.child_apps = subseq

        subseq = seq[span[0]:span[1]]
        # print("SUBSEQ", subseq)
        # print()
        # print("+", [(x.skill, id(x),  [y.id for y in x.match]) for x in subseq])
        
        # print()
        # print("::", span, subseq)
        alignment[alignment>=0] -= span[0]
        b_alignment[b_alignment>=0] -= b_span[0]

        method_app = copy(method_match_tree.skill_apps[0])
        method_app.state = subseq[0].state
        method_app.child_apps = item_m_skill_apps

        rc = find_rhs_change(method_app, subseq, alignment,
         span=span, seq=seq, scores=scores, seq_trees=seq_trees,
         b_span=b_span, b_alignment=b_alignment) 
        rc.align_scores = scores
        best_changes.append(rc)
        # print("@",overlap, alignment)
        # print(rc.alignment, seq)
        # print("<<", rc.cost, rc)
        

    return best_changes


def fill_span_holes(changes, seq):
    head, end = 0, len(seq)
    rcs = sorted(changes, key=lambda x:x.span[0])

    # print("COV", upstream_covered)

    # print(">>", [(rc.span, rc.edits) for rc in rcs], len(seq))
    # print("??", [x.span for x in changes], end)
    new_rcs = []
    for i, rc in enumerate(rcs):
        if(rc.nomatch):
            new_rcs.append(rc)
            continue

        s0, e0 = rc.span
        # print(head, rc.rhs.symbol, rc.cost, rc.span, rc.edits)

        # If hole between 0 and first span
        if(head < s0):
            # if(not np.all(upstream_covered[head:s0])):
            new_edits = []
            new_edits.append(('insert', head, head, seq[head:s0]))
            # for j in range(head,s0):
                # if(j < len(seq_trees)):
                #     print("SQT:", seq_trees[j])
                # if(not upstream_covered[j]):
                
            new_edits += rc.edits
            # ins_cost = cost_of_edits(rc.rhs, new_edits, e0-s0)
            # Prepend any uncovered symbols to beginning
            rc = RHSChange(rc.method, span=(head,e0), edits=new_edits)
            new_rcs.append(rc)

        elif(e0 <= head):
            # If candidate changes overlap use the one that costs less 
            if(rc.cost < new_rcs[-1].cost):
                new_rcs[-1] = rc    
        else:
            new_rcs.append(rc)
        
        if(i+1 < len(rcs)):
            nxt_rc = rcs[i+1]
            s1 = nxt_rc.span[0]
        else:
            # print("END", e0, end)
            s1 = end

        # If hole between spans
        if(e0 < s1):
            # Append "inserts" to end
            # if(not np.all(upstream_covered[s0:s1])):
            new_edits = [*rc.edits]
            new_edits.append(('insert', e0, e0, seq[e0:s1]))
            # for j in range(e0,s1):
                # if(j < len(seq_trees)):
                #     print("SQT:", seq_trees[j])
                # if(not upstream_covered[j]):
                
                
            # print(new_edits)
            # ins_cost = cost_of_edits(rc.rhs, new_edits, s1-head)
            new_rcs[-1] = RHSChange(rc.method_app, span=(head,s1), edits=new_edits)
            
        head = max(s1,e0)
    # print(">>>", [(rc.span, rc.edits) for rc in new_rcs], len(seq))
    return new_rcs

class ParseTree:
    def __init__(self, rhs_change, child_trees=[]):
        # print(rhs_change.rhs, rhs_change.span)
        # print("CHILD TREES", child_trees)
        # print("PT MACRO", rhs_change.method, rhs_change.method.macro)
        self.method = rhs_change.method

        # rhs_change, child_trees = self._mergeInsDel(rhs_change, child_trees)
        self.change = rhs_change
        # self.macro = self.method.macro
        self.child_trees = child_trees
        self.cost = 0.0
        self.recalc_cost()        

    @property
    def span(self):
        return self.change.span

    def recalc_cost(self):
        # self.cost = self.change.cost + sum([rc.cost for rc in self.child_trees if rc])
        self.cost = sum([rc.cost for rc in self.changes if rc])

    def __str__(self):
        # if(self.macro is None):
        #     raise ValueError("NO MACRO", self.macro)
        # s = f"{self.cost:.2f} {self.macro}->{self.method} {self.change.edits}"
        s = f"{self.cost:.2f} {self.change}"
        for ct in self.child_trees:
            if(isinstance(ct, ParseHole)):
                continue
            if(ct):
                s += "\n" + str(ct)
        return s

    def __repr__(self):
        return f"PT[{self.change.macro_str}]"


    def _copy_child_trees(self, replacements=[]):
        child_copies = []
        for ct in self.child_trees:
            if(ct):
                replaced = False
                for old, new in replacements:
                    if(ct is old):
                        if(new is None):
                            continue
                        ct = new
                        replaced = True
                if(not replaced):
                    ct = ct._deep_copy(replacements)

            child_copies.append(ct)
        return child_copies

    def _deep_copy(self, replacements=[]):
        child_copies = self._copy_child_trees(replacements)
        return ParseTree(self.change, child_copies)

    @property
    def trees_by_depth(self):
        if(not hasattr(self, '_trees_by_depth')):
            trees_by_depth = [[self]]
            pts = [self]
            # depth = 0
            while(len(pts) > 0):
                new_pts = []
                for i, pt in enumerate(pts):
                    # print("PT", pt.method)
                    if(not pt or isinstance(pt, ParseHole)):
                        continue

                    child_trees = []
                    for cpt in pt.child_trees:
                        if(cpt and not isinstance(cpt, ParseHole)):
                            new_pts.append(cpt)
                # depth += 1
                trees_by_depth.append(new_pts)
                pts = new_pts
            # print(trees_by_depth)
            self._trees_by_depth = trees_by_depth
        return self._trees_by_depth

    def iter_trees(self):
        for depth, trees in enumerate(self.trees_by_depth):
            for tree in trees:
                yield tree

    @property
    def changes(self):
        if(not hasattr(self, '_changes')):
            self._changes = []
            trees = reversed(list(self.iter_trees()))
            # print(trees)
            for pt in trees:
                # print(">>", pt.change.method_app)
                self._changes.append(pt.change)
        return self._changes

    # @property
    # def joined_changes(self):
    #     if(not hasattr(self, '_joined_changes')):
    #         joined_changes
    #         rhs_to_pts = {}

    #         for pt in self.iter_trees():
    #             if(len(pt.change.edits) == 0 or 
    #                id(pt) in covered):
    #                 continue
    #             _, lst = joined_changes.get(pt.change.method, [])
    #             lst.append(pt)
    #             joined_changes[pt.change.method] = (None, lst)
    #             covered.add(id(pt))

    #         for rhs, (_,pts) in [*joined_changes.items()]:

    #             if(len(pts) > 1):
    #                 _edits = []
    #                 for pt in pts:
    #                     for edit in pt.change.edits:
    #                         if(edit not in _edits):
    #                             _edits.append(edit)
    #                 # Combine
    #                 edits = sorted(list(_edits),key=lambda x: x[1])
    #                 # RHSChange()
    #             else:
    #                 rc = pts[0].rc


    #             for pt in pts:
    #                 pt.change.edits = edits
    #                 orig_cost = pt.change.cost
    #                 pt.change.recalc_cost()
    #                 pt.cost += pt.change.cost-orig_cost
    #         self._joined_changes = 'TOTODOO'


class ParseHole(ParseTree):
    def __init__(self, skill_app, child_tree, prev_pt=None, next_pt=None, span=None):
        self.skill_app = skill_app
        if(isinstance(child_tree, ParseHole)):
            self.child_hole = child_tree
        self.child_trees = [child_tree]
        self.prev_pt = prev_pt
        self.next_pt = next_pt
        self._span = span if(span) else child_tree.span
        self.cost = child_tree.cost if child_tree else 0.0

    @property
    def span(self):
        return self._span

    def __repr__(self):
        return f"H[{self.skill_app}]"

    __str__ = __repr__

    def _get_next_symbols(self):
        '''TODO: Too complicated for now'''
        # print(self.symbol,"->", self.change.rhs, self.change.edits)

        if(len(self.child_trees) > 0):
            ns = []
            for ct in self.child_trees:
                ns += ct.get_next_symbols()
            return ns
        else:
            return []

    def _deep_copy(self, replacements=[]):
        child_copy = self._copy_child_trees(replacements)[0]
        return ParseHole(self.skill_app, child_copy,
             prev_pt=self.prev_pt, next_pt=self.next_pt)





    # def __str__(self):

        





# Adopted from https://stackoverflow.com/questions/54476451/how-to-get-all-maximal-non-overlapping-sets-of-spans-from-a-list-of-spans
def get_spanning_trees(parse_trees):
    if(len(parse_trees) == 0):
        return [parse_trees]

    parse_trees = sorted(parse_trees, key=lambda x: x.change.span[0])
    parse_trees = [pt for pt in parse_trees if not pt.change.nomatch]

    paths = [[ parse_trees.pop(0) ]]
    for pt in parse_trees:
        rc = pt.change
        l,r = rc.span

        to_extend = []
        longest = 0
        for path in paths:
            # If change overlaps with last span in path
            #  then try to add to its prefix
            lp, rp = path[-1].change.span
            if lp <= l < rp:
                prefix = path[:-1]
                if len(prefix) >= longest:
                    to_extend.append(prefix + [pt])
                    longest = len(prefix)
            # Otherwise, it's after so append it.
            else:
                path.append(pt)
                longest = len(path)
        for path in to_extend:
            if len(path) == longest + 1:
                paths.append(path)

      # print "\nResult: %s" % paths
    # print(">>", [(f"{sum([pt.cost for pt in path]):.2f}", [(pt.macro, *pt.change.span) for pt in path]) for path in paths])
    return paths

def filter_spanning_trees(paths):
    min_paths = []
    min_cost = float('inf')
    for path in paths:
        cost = sum([pt.cost for pt in path])
        if(cost == min_cost):
           min_paths.append(path) 
        elif(cost < min_cost):
            min_paths = [path]
            min_cost = cost
    return min_paths[0:1]

        # print(":", rc.rhs.symbol, rc.span)

def _sequence_upstreams(grammar, rhss):
    if(len(rhss) == 1):
        return [rhss]

    def n_upstreams(rhs):
        uptrms = grammar.get_upstream_RHSs(rhs)
        inrs = sum([x in uptrms for x in rhss])
        return inrs

    tups = [(n_upstreams(rhs), rhs) for rhs in rhss]
    tups = sorted(tups, key=lambda x: x[0], reverse=True)

    curr_ups, frst_rhs = tups.pop(0)
    out = [[frst_rhs]]
    for n_ups, rhs in tups:
        if(n_ups == curr_ups):
            out[-1] = [rhs]
        else:
            out.append([rhs])
            curr_ups = n_ups  
    # print("++", out)
    return out          



def get_upstream_RHSs(grammar, symbols):
    # print([type(x) for x in symbols])
    how_parts = set([x.skill.how_part for x in symbols if hasattr(x, 'skill')])
    # print("HP", how_parts)
    upstream_methods = {}        
    for _, macro in grammar.macro_skills.items():
        for method in macro.methods:
            for item in method.items:
                # print(item, symbols, getattr(item,'how_part',None))
                if(item in symbols or getattr(item,'how_part',None) in how_parts):
                    lst = upstream_methods.get(method,[])
                    lst.append(item)
                    upstream_methods[method] = lst
    
    return upstream_methods

def copy_macro_app_w_state(rc, match_tree=None):
    # Make a copy of the RHSChange's macro_app with
    #  state lifted from the covered subseqence.
    macro_app = None
    if(isinstance(rc, MergeChange)):
        macro_app = copy(rc.macro_app)
    else:
        # There are case
        if(rc.method_app is not None):
            # print(rc.method_app.parent)
            macro_app = copy(rc.method_app.parent)
        else:
            raise ValueError("Shouldn't be a terminal skill")
    # print("**", rc, rc.subseq)
    macro_app.state = rc.subseq[0].state
    macro_app.prob_uid = rc.subseq[0].prob_uid
    if(match_tree):
        annotate_exact_match(match_tree, macro_app)
    # macro_app.child_apps = [rc.method_app]
    return macro_app



def _make_seq_and_holes(parse_tree_seq, seq,
         seq_trees=None, match_tree=None, make_new_seq=True):
    # print("START MAKE HOLES")
    # If make_new_seq=False then only make new tree seq
    #  without bothering with making new_seq. 
    new_seq = []
    new_trees = []
    head = 0
    prev_pt = None
    for i, pt in enumerate(parse_tree_seq):
        print(pt)
        rc = pt.change
        s0,e0 = rc.span
        # print(rc, s0, e0)
        if(e0-s0 == 0):
            new_trees.append(pt)
            continue
        # print(pt)
        
        # print("@@", rc.span)

        # Fill in any symbols preceeding the previous non-terminal
        if(head < s0):
            # print("B: NEVER HAPPEN W FILLED SPAN")
            for j in range(head, s0):
                if(make_new_seq):
                    new_seq.append(seq[j])
                # if(seq_trees and isinstance(seq_trees[j], ParseHole)):
                #     ph = seq_trees[j]
                # else:
                t = seq_trees[j] if seq_trees else None
                print("  $$", seq[j],"prev", prev_pt, "ext", pt)
                ph = ParseHole(seq[j], t, prev_pt=prev_pt, next_pt=pt, span=(j,j+1))
                new_trees.append(ph)
                print(ph)

        # # Fill in the next non-terminal
        # new_seq.append(rc.method_app.parent)
        if(make_new_seq):
            new_seq.append(copy_macro_app_w_state(rc, match_tree))
        new_trees.append(pt)
        head = rc.span[1]
        prev_pt = pt

    # Fill in any trailing symbols
    if(head < len(seq)):
        # print("E: NEVER HAPPEN W FILLED SPAN", head, len(seq))
        for j in range(head, len(seq)):
            if(make_new_seq):
                new_seq.append(seq[j])
            if(seq_trees and isinstance(seq_trees[j], ParseHole)):
                ph = seq_trees[j]
            else:
                t = seq_trees[j] if seq_trees else None
                ph = ParseHole(seq[j], t, prev_pt=prev_pt, span=(j, j+1))
            new_trees.append(ph)
    # new_seq = [rc.rhs.symbol for rc in change_seq if not rc.nomatch]
    # print("NEW SEQ", new_seq)
    # assert new_seq != seq, f"Recursion made no parsing progress: {seq}, then {new_seq}."
    # print("!!NS", new_seq)
    if(make_new_seq):
        return new_trees, new_seq 
    else:
        return new_trees

def _hole_best_insert_cands(parse_hole):
    # Go through the successive child parse holes
    #  and find the best pairs of left and right candidate 
    #  trees to merge this hole into.
    best_l_aff = 0.0
    best_l_cand = None
    best_r_aff = 0.0
    best_r_cand = None
    depth = 0
    hole = parse_hole
    while(hole):
        if(hasattr(hole, 'exact_match')):
            continue
        prev = hole.prev_pt
        if(prev):
            aff = hole.skill_app.match_overlap(prev.change.method_app)
            if(aff >= best_l_aff):
                best_l_aff, best_l_cand = aff, prev
        nxt = hole.next_pt
        if(nxt):
            aff = hole.skill_app.match_overlap(nxt.change.method_app)
            if(aff >= best_r_aff):
                best_r_aff, best_r_cand = aff, nxt
        depth +=1
        # print(":::", best_l_cand, best_r_cand)
        # print("HOLE", hole, prev, nxt)
        hole = getattr(hole,'child_hole', None)
        # print("CHILD HOLE", hole)
    # print("BEST", "L:",best_l_aff, best_l_cand, "R:", best_r_aff, best_r_cand)
    return (best_l_aff, best_l_cand), (best_r_aff, best_r_cand)
    # # print("Hole:", parse_hole)
    # out = []
    # for depth, cand in cands:
    #     other_sa = cand.change.method_app
    #     self_sa = parse_hole.skill_app
    #     affinity = self_sa.match_overlap(other_sa)
    #     out.append((affinity, depth, parse_hole, cand))
    #     print(" ", depth, affinity, cand)
    # return out
    # print()

def _make_hole_groups(child_trees, seq):
    # Group together contiguous parse holes
    # print("INP", child_trees)
    groups = []
    grp_trees = []
    subseq = []
    prev_typ = None
    for i in range(0,len(child_trees)+1):
        # print(i, group)
        if(i == len(child_trees) or child_trees[i] is None):
            typ = None
        elif(isinstance(child_trees[i], ParseHole)):
            typ = "holes"
        else:    
            typ = "trees"

        if(i == len(child_trees) or typ != prev_typ):
            if(len(grp_trees) > 0):
                groups.append((prev_typ, grp_trees, subseq))
            grp_trees = []
            subseq = []
        # else:
            # print("PH", child_trees[i], child_trees[i].prev_pt, child_trees[i].next_pt)
        if(i != len(child_trees)):
            grp_trees.append(child_trees[i])
            subseq.append(seq[i])
            prev_typ = typ
    # print(groups)
    for group in groups:
        print("!!", group) 
    return groups

def _get_cum_grp_aff(grp_aff):
    N = len(grp_aff)
    cum_grp_aff = np.zeros(N, dtype=np.float64)
    for k in range(N):
        l_aff = np.sum(grp_aff[0:k, 0:k])
        inner_aff = np.sum(grp_aff[k:N, k:N])
        cum_grp_aff[k] = l_aff + inner_aff
    return cum_grp_aff

def _make_group_aff_matrix(group_sas, min_aff=.2):
    N = len(group_sas)
    grp_aff = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        for j in range(i+1, N):
            a, b = group_sas[i], group_sas[j]
            aff = a.match_overlap(b)
            if(aff > min_aff):
                grp_aff[i,j] = aff
            # grp_aff[j,i] = aff
    return grp_aff

def _best_group_parition(group, force_cands=True):
    N = len(group)
    grp_aff = _make_group_aff_matrix([x.skill_app for x in group])
    l_cands, r_cands = [None]*(N+1), [None]*(N+1)
    l_affs = np.zeros(N, dtype=np.float64)
    r_affs = np.zeros(N, dtype=np.float64)

    # print("group", group)
    cands = []
    for i, hole in enumerate(group):
        (l_affs[i],l_cands[1+i]),(r_affs[i],r_cands[i]) = \
            _hole_best_insert_cands(hole)
    # print("l_cands", l_cands)
    l_affs[l_affs < .2] = 0.0
    r_affs[r_affs < .2] = 0.0
    score, (pl, pr) = _best_parition_inds(l_affs, r_affs, grp_aff)

    if(force_cands):
        return (pl,pr), l_cands[-1], r_cands[0]
    elif(score == 0.0):
        return (pl,pr), None, None
    else:
        return (pl,pr), l_cands[pl], r_cands[pr]

    # print("CANDS", score, l_cands, r_cands)
    

@njit(cache=True)
def _best_parition_inds(l_affs, r_affs, grp_aff):
    N = len(l_affs)
    # affs = np.zeros((N+1,N+1), dtype=np.float64)
    best_inds = (0,N)
    best_score = 0.0
    for i in range(N+1):
        la = np.sum(grp_aff[0:i, 0:i]) + np.sum(l_affs[0:i])
        for j in range(i,N+1):
            inga = np.sum(grp_aff[i:j, i:j])
            ra = np.sum(grp_aff[j:N, j:N])+ np.sum(r_affs[j:N])
            score = la + inga + ra - 1e-6*i - 1e-6*(N-j)
            if(score > best_score):# and score > min_aff):
                best_score = score
                best_inds = (i,j)
    # print("best_score", best_score)
    return best_score, best_inds


# def _remove_reassigned_holes(pt, ind, pl, pr):
#     # print("@@", ind, pl, pr)    
#     for ei, edit in enumerate(pt.change.edits):
#         if(edit[0] == 'insert' and edit[1]==ind):
#             # print(pt.change.edits[ei])
#             pt.change.edits[ei] = ("insert", ind, ind, edit[3][pl:pr])
#             pt.change.recalc_cost()
#             # print(pt.change.edits[ei])
#     # print("@@")

def _insert_reassigned_hole(pt, part, kind="append"):
    skill_apps = [pt.skill_app for pt in part]

    if(kind == "prepend"):
        ind = 0
    elif(kind == "append"):
        ind = len(pt.change.method)

    # Delete any old 'insert' edits if necessary
    for ei, edit in enumerate(pt.change.edits):
        if(edit[0] == 'insert' and edit[1] == ind):
            del pt.change.edits[ei]
            if(kind == "prepend"):
                skill_apps = skill_apps + edit[3]
            elif(kind == "append"):
                skill_apps = edit[3] + skill_apps 

    # Insert the 'insert' edit
    s, e = pt.change.span
    if(kind == "prepend"):        
        pt.change.edits = [('insert', ind, ind, skill_apps)] + pt.change.edits
        s = s-len(skill_apps) if(ind == s) else s
        pt.change.span = (s, e)
    elif(kind == "append"):
        pt.change.edits = pt.change.edits + [('insert', ind, ind, skill_apps)]
        e = e+len(skill_apps) if(ind+1 == e) else e
        pt.change.span = (s, e)

    pt.change.edits = _merge_contiguous(pt.change.edits)
    pt.change.recalc_cost()
    # print("MERGE!", pt.change.edits)

def _fill_parse_holes(child_trees, seq, no_keep=False):
    # Group contiguous sequences of parse holes between 
    #  valid parsed sections and determine how the groups 
    #  will be assigned to adjacent parse trees
    groups = _make_hole_groups(child_trees, seq)
    # print("GROUPS", groups)
    new_trees = []
    new_seq = []
    for ind, (typ, grp_trees, subseq) in enumerate(groups):
        # print("##", typ, group)
        if(typ == "holes"):
            # print("group", group)
            (pl, pr), l_pt, r_pt = _best_group_parition(grp_trees, no_keep)
            left_part = grp_trees[:pl]
            keep_part = grp_trees[pl:pr]
            right_part = grp_trees[pr:]
            # print(l_pt, r_pt)
            # print(left_part, keep_part, right_part)

            # TODO: I don't remember what this does but it is
            #  probably necessary (error: pt undefined.. what goes there?)
            # if(len(left_part)+len(right_part) > 0):
            #     _remove_reassigned_holes(pt, ind, pl, pr)
            if(len(left_part) > 0):
                _insert_reassigned_hole(l_pt, left_part, "append")
            if(len(right_part) > 0):
                _insert_reassigned_hole(r_pt, right_part, "prepend")
            if(no_keep and len(keep_part)):
                if(r_pt):
                    _insert_reassigned_hole(r_pt, keep_part, "prepend")
                elif(l_pt):
                    _insert_reassigned_hole(l_pt, left_part, "append")
            else:
                new_trees += keep_part
                new_seq += subseq[pl:pr]
        else:
            new_trees += grp_trees
            new_seq += subseq
    # print(child_trees, [x for x in child_trees if not isinstance(x, ParseHole)])
    # print(">>",child_trees, new_trees)
    return new_trees, new_seq
    # return [x for x in child_trees if not isinstance(x, ParseHole)]
        # print(l_pt, "<", left_part, keep_part, right_part, ">", r_pt)

# def _fill_external_parse_holes(child_trees):

#     child_trees = pt.child_trees
#     span = pt.change.span
#     left_holes = [ParseHole(seq[i] for i in range(0,span[0]))]
#     right_holes = [ParseHole(seq[i] for i in range(span[1],len(seq)))]

#     _insert_reassigned_hole(l_pt, left_part, "append")
#     _insert_reassigned_hole(r_pt, right_part, "prepend")


# def _fill_parse_holes_legacy(pt):
#     rc = pt.change
#     child_trees = pt.child_trees
#     holes_spans = {}
#     # modified_edits = [*rc.edits]
#     # print(":>", rc)

#     if(not rc.edits or len(rc.edits) == 0):
#         return pt
#     # Check the highest edits in the parse tree and see if they
#     #  overlap with an unparsed a ParseHole symbol. 
#     for edit_ind, edit in reversed([*enumerate(rc.edits)]):
#         # NOTE: It is not clear if needs to also happen on replaces
#         if(edit[0] in ('insert')):#, 'replace')):
#             s = edit[1]+rc.span[0]
#             # print("S", s, child_trees[s] if child_trees and s < len(child_trees) else None)
#             if(child_trees and s < len(child_trees) and isinstance(child_trees[s], ParseHole)):
#                 # If a symbol precedes the hole use that
#                 #  otherwise insert the holes into the next symbol
#                 hp = child_trees[s]
#                 adj, is_prev = (hp.prev_pt, True) if hp.prev_pt else (hp.next_pt, False)
#                 _rc = adj.change

                

#                 # Skip inserting into unordered RHSs since
#                 if(_rc.method.unordered):
#                     continue

#                 # print("  ###:", rc, f"({rc.seq[rc.span[0]:rc.span[1]]})")
#                 # print(" :-", hp, "|", hp.prev_pt, "|", hp.next_pt)
                

#                 _, p_lst, n_lst = holes_spans.get(adj.change.method, (None, [],[]))
#                 lst = n_lst if is_prev else p_lst
#                 lst.append(hp)
#                 holes_spans[adj.change.method] = (adj, p_lst, n_lst)
#                 del rc.edits[edit_ind]
    
#     if(len(holes_spans) == 0):
#         return pt

#     # from copy import deepcopy
#     # pt = deepcopy(pt)
    
#     # print("# HP:", len(holes_spans))
#     replacements = []
#     for rhs, (adj, p_lst, n_lst) in holes_spans.items():
#         _rc = adj.change

#         # print(_rc.span)
#         prev, nxt = [], []
#         if(len(p_lst) > 0):
#             prev = [('insert', 0, 0, [hp.symbol for hp in p_lst])]
#         if(len(n_lst) > 0):
#             loc = _rc.span[1]-_rc.span[0]
#             nxt = [('insert', loc, loc, [hp.symbol for hp in n_lst])]
#         edits = [*prev, *_rc.edits, *nxt]
        
#         s,e = _rc.span[0]-len(p_lst), _rc.span[1]+len(n_lst)
#         # cost = cost_of_edits(rhs, edits, e-s)
#         rc = RHSChange(rhs, span=(s,e), edits=edits)
#         _pt = ParseTree(rc, child_trees=adj.child_trees)
#         # print(" ->", rc.cost, rc)
#         # print(_pt)
#         # print()
#         replacements.append((adj, _pt))

#     pt = pt._deep_copy(replacements)
#     pt.change.recalc_cost()
#     pt.recalc_cost()
#     return pt


def join_edits(edits0, edits1):
    # TODO: Make fancier when more test examples

    return list(set(edits0).intersection(set(edits1)))

def _parse_recursions(trees, seq):
    rec_rhs = None
    start = 0
    spans = []
    for i, (tree, item) in enumerate(zip(trees, seq)):
        # print(tree.rhs.is_recursive)
        if(type(tree) is ParseTree and tree.method.is_recursive):
            rhs = tree.method
            # print(i, rhs)
            if(not rec_rhs):
                rec_rhs = rhs
                start = i
            elif(rhs is not rec_rhs):
                rec_rhs = None
                spans.append((rec_rhs, start, i))
        elif(rec_rhs):
            rec_rhs = None
            spans.append((rec_rhs, start, i))

    for rhs, s, e in reversed(spans):
        if(e-s >= 2):
            prev_pt = trees[e-1]
            for i in range(e-2, s-1, -1):
                pt = trees[i]
                pt.change.span = (pt.change.span[0], pt.change.span[1]+1)
                pt.child_trees.append(prev_pt)

                # pt.join_changes(prev_pt)
                # pt.change.edits = join_edits(pt.change.edits, prev_pt.change.edits)
                # print(pt.change.edits)
                # prev_pt.change.cost = 0
                # prev_pt.change.edits = []
                # prev_pt.cost = 0.0
                # prev_pt = pt
                # pt.recalc_cost()

                
                # pt = ParseTree( pt.parse_trees[])
                # print("!>>", trees[i], trees[i].change.span)

            trees = trees[:s]+[pt]+trees[e:]
            seq = seq[:s]+[pt.symbol]+seq[e:]

    # print("::", spans)
    return trees, seq


def _only_least_cost(changes):
    min_cost = float('inf')
    min_rc = None
    for rc in changes:
        if(rc.cost < min_cost):
            min_cost = rc.cost
            min_rc = rc
    # print("MIN RC", min_rc.cost, min_rc)
    return [rc for rc in changes if rc.method.macro is min_rc.method.macro or rc.cost == min_cost]


def inds_to_unorder_spans(ind_set):
    spans = []
    m = min(ind_set)
    arr = np.arange(len(ind_set))-(np.array(ind_set)-m)
    cc = np.cumsum(arr)
    prev_start = -1
    for i, v in enumerate(cc):
        if(v == 0):
            if(prev_start != -1):
                spans.append((prev_start+m, i+1+m))
                prev_start = -1
        elif(prev_start == -1):
            prev_start = i

    return spans
# inds_to_unorder_spans([1,0,3,2])
# inds_to_unorder_spans([3,0,2,1])
# inds_to_unorder_spans([0,1,2,3])
# inds_to_unorder_spans([2,4,3,5])

# raise ValueError()


def _make_merge(macro_app, ptrees, seq, seq_trees):
    # Sort parse trees by span
    ptrees = sorted(ptrees, key=lambda x : (x.change.b_span[0]))

    # Find the total span of the disjoint methods span
    pt0 = ptrees[0]
    e0 = max([x.change.b_span[1] for x in ptrees])
    s0 = pt0.change.b_span[0]
    meth0 = pt0.method
    print("\t", macro_app, f"s0={s0},e0={e0}")
    print("\tmeth0=", meth0)
    print("SUBSEQ", [str(x) for x in seq[s0:e0]])
    edits = []
    
    meth_map = np.full((e0-s0, len(ptrees)),-2, dtype=np.int64)
    aligned_map = np.full((e0-s0, len(ptrees)),0, dtype=np.uint8)
    score_map = np.full((e0-s0, len(ptrees)),0, dtype=np.float64)
    methods = []
    method_apps = []
    for k, pt in enumerate(ptrees):
        pt = ptrees[k]
        rc = pt.change

        # Offset Alignment w.r.t shared seq[s0:e0] instead of local subseq
        of_alignment = rc.b_alignment.copy()
        offset = rc.b_span[0]-s0
        of_alignment[rc.b_alignment>=0] += offset
        methods.append(rc.method)
        method_apps.append(rc.method_app)

        print(rc.method, f"al={rc.b_alignment}", f"oal={of_alignment}")

        # Fill meth_map and aligned_map with shapes
        #  (Len Subseq, N Methods). For method k, 
        #  item at index i  
        #   1) meth_map has the index of the item 
        #       in method k overlaps with item i.
        #       And -2 Otherwise.
        #   2) aligned_map i,k is true if method k
        #   aligns with the sub sequence at i.
        # frst = offset-np.argmax(of_alignment >= 0)
        # print("frst", frst, offset)
        for i, ind in enumerate(of_alignment):
            if(ind >= 0):
                # print("MM", ind, k, ":", i)
                meth_map[ind][k] = i
                aligned_map[ind][k] = 1
                score_map[ind][k] = rc.align_scores[i]
            # else:
            #     pass
                # print("NO", i+frst, k, ":", i)
                # meth_map[i+frst][k] = i     

    # Fill in the unaligned but overlapping items to meth_map
    main_meth_inds = np.zeros(len(aligned_map),dtype=np.int64)
    for i, (mp, ap, sp) in enumerate(zip(meth_map, aligned_map, score_map)):
        # print(mp != -1)
        k = np.argmax(sp)
        if(mp[k] >= 0):
            main_meth_inds[i] = k
        else:
            main_meth_inds[i] = -1

        sp_str = np.array2string(sp, precision=2, floatmode='fixed')
        mp_str = np.array2string(mp, sign=" ")
        print(f'{i} {seq[s0+i]}: '+
            ' '.join([(f"{methods[k].items[j]}" if j >= 0 else "-").ljust(2) for k,j in enumerate(mp)])+
            f' | a={ap} m={mp_str}, s={sp_str} k={main_meth_inds[i]}')

    # Partition by grouping contiguous align_map patterns.
    partitions = []
    p0 = 0
    prev_ap = aligned_map[0]
    prev_k = main_meth_inds[0]
    for i in range(1,len(aligned_map)):
        ap = aligned_map[i]
        k = main_meth_inds[i]
        if( #np.sum(prev_ap) > 1 and
            (not np.array_equal(prev_ap, ap) or k != prev_k)):
            partitions.append((p0, i, prev_k))
            prev_ap = ap
            prev_k = k
            p0 = i

    partitions.append( (p0,len(aligned_map), main_meth_inds[-1]) )

    print("partitions", partitions)
    part_edits = [[] for _ in range(len(partitions))]

    # Make spans of the aligned indicies for each method
    #  within each partition. Some spans will be missing.
    span_sets = []
    for i,(p0, p1, main_k) in enumerate(partitions):
        inds = [None]*len(methods)
        spans = [None]*len(methods)

        # for k in range(len(methods)):
        for k in range(len(methods)):
            ind_set = meth_map[p0:p1,k]
            ind_set = ind_set[ind_set>=0]
            if(len(ind_set) > 0):
                spans[k] = (min(ind_set), max(ind_set)+1)
                # print((p0,p1,k), meth_map[p0:p1,k], '++', ind_set, spans[k], (min(ind_set), max(ind_set)+1))
                inds[k] = ind_set

        span_sets.append(spans)

        span = spans[main_k]
        ind_set = inds[main_k]

        if(ind_set is None):
            continue
        
        # Ensure any unordering added
        if(methods[k].unordered):
            part_edits[i].append(('unorder', span[0], span[1]))

        elif(not np.array_equal(ind_set, np.arange(span[0],span[1]))):
            for us, ue in inds_to_unorder_spans(ind_set):
                part_edits[i].append(('unorder', us, ue))
        # else:
        #     spans[k] = None
        

            


        

    change_parts = []
    # for i, (part,spans) in enumerate(zip(partitions, span_sets)):
    #     meth_edit = ()
    #     change_parts.append((part, [meth_edit]))
        
    m_offs = np.zeros(len(methods), dtype=np.int64)
    edits = []
    child_trees = [None]*len(meth_map)
    for i, (part,spans) in enumerate(zip(partitions, span_sets)):
        (ps, pe, main_k) = part        
        # All None indicates partions with subsequence
        #   that no methods aligned with. So
        #   insert that subsequence. 
        if(all(x is None for x in spans)):
            edit = ('insert', i, i, seq[s0+part[0]:s0+part[1]])
            change_parts.append((part, [(-1, None, None, [edit])]))
        #     change_parts.append((part, []))
        #     edits = find_edits([], seq[s0+part[0]:s0+part[1]])
            # print(">", part,":", (-1, None, None, [edit]))
            # print("ALL", part)
            # edit = ('insert', m_offs[0], m_offs[0], seq[s0+part[0]:s0+part[1]])
            # edit = ('insert', i, i, seq[s0+part[0]:s0+part[1]])
            # edits.append()
            # change_parts.append((part, [(-1, None, None, [edit])]))
        # Otherwise 
        else:
            # print(i, "before", spans)
            # Impute the missing overlapping
            #  span for each unaligned method.
            for k, span in enumerate(spans):
                if(span is None):
                    upper = None
                    if(i+1 < len(span_sets)):
                        u_span = span_sets[i+1][k]
                        # print("u_span1", u_span)
                        if(u_span is not None):
                            upper = u_span[0]
                    if(upper is None and i+2 < len(span_sets)):
                        u_span = span_sets[i+2][k]
                        # print("u_span2", u_span)
                        if(u_span is not None):
                            upper = u_span[0]
                    if(upper is None):
                        upper = len(methods[k])

                    spans[k] = (m_offs[k], upper)
                else:
                    m_offs[k] = span[1]

            meth_edits = []
            # print(i, "after", spans)
            for k, span in enumerate(spans):
                if(span[1]-span[0] > 0):
                    meth_edits.append((
                        k, # Method index 
                        span, # Span within method
                        np.arange(ps, pe, dtype=np.int64),#meth_map[part[0]:part[1], k], # Alignment
                        part_edits[i] if k == main_k else [] # Edits
                    ))
                    
            change_parts.append((part, meth_edits))

    # Fill gaps in method spans
    gap_pairs = []
    # m_ends = np.zeros(len(methods),dtype=np.int64)
    prevs = [None]*len(methods)
    for i, ((sp, ep, main_k), meth_mods) in enumerate(change_parts):
        if(main_k == -1):
            continue
        for j, (k, meth_span, alignment, meth_edits) in enumerate(meth_mods):

            ms, me = meth_span
            # print("^^", k, prevs[k][2][1] if prevs[k] else None, ms)
            gl = prevs[k][2][1] if prevs[k] else 0
            if(gl < ms):
                gap_r = (i,j, meth_span)
                gap_pairs.append((k, (gl, ms), prevs[k], gap_r))
            prevs[k] = (i,j, meth_span)

    for k in range(len(methods)):
        gap_l = prevs[k] 
        if(gap_l[2][1] < len(methods[k])):
            gap_pairs.append((k, (gap_l[2][1], len(methods[k])), gap_l, None))
            # print("KK", k, meth_span)
            # m_ends[k] = me
    # print("GAP PAIRS", len(gap_pairs))
    del_sections = []
    for k, gap_span, gap_l, gap_r in gap_pairs:
        # print("GS", gap_span, gap_l, gap_r)
        N = len(methods[k])
        method_app = ptrees[k].change.method_app
        # print("child_trees", method_apps[k].child_apps)
        l_sa, r_sa = None, None
        if(gap_span[0]-1 >= 0):
            
            l_sa = method_app.child_apps[gap_span[0]-1]
        if(gap_span[1] < N):
            # print(method_app, method_app.child_apps)
            r_sa = method_app.child_apps[gap_span[1]]
        inner_sas = method_app.child_apps[gap_span[0]:gap_span[1]]
        # print(repr(l_sa), inner_sas, repr(r_sa))

        # Find the best way attribute the items in the gap
        #  to the left or right.
        grp_aff = _make_group_aff_matrix(inner_sas)
        l_affs = np.zeros(len(inner_sas), dtype=np.float64)
        r_affs = np.zeros(len(inner_sas), dtype=np.float64)
        for i, sa in enumerate(inner_sas):
            if(l_sa):
                l_affs[i] = sa.overlap(l_sa)
            if(r_sa):
                r_affs[i] = sa.overlap(r_sa)

        # print("AFFS", l_affs, r_affs)
        l_affs[l_affs < .2] = 0.0
        r_affs[r_affs < .2] = 0.0
        # print("AFFS", l_affs, r_affs)

        _, (pl, pr) = _best_parition_inds(l_affs, r_affs, grp_aff)
        pl += gap_span[0]
        pr += gap_span[0]
        print("PLUR", gap_span[0], pl, pr, gap_span[1])

        # Case: _best_parition_inds has assigned some subsequence
        #  of items in a gap to a method to the left.
        if(pl-gap_span[0] > 0):
            
            (i,j, _) = gap_l
            (_k, meth_span, alignment, meth_edits) = change_parts[i][1][j]
            (ms, me) = meth_span
            meth_span = (ms,pl)
            meth_edits.append(('delete', me-ms, pl-ms))
            # alignment = np.concatenate((alignment, np.arange(me,pl)))
            alignment = np.concatenate((alignment, -np.ones(pl-me, dtype=np.int64)))
            change_parts[i][1][j] = (_k, meth_span, alignment, meth_edits)

            # ((sp, ep, main_k), meth_mods) = change_parts[i]            
            # change_parts[i] = ((sp, ep+(pl-me), main_k), meth_mods)
            # print("JOIN >!!", i,j, (ms, me),"->",meth_span, meth_edits[-1])

        # Case: _best_parition_inds has assigned some subsequence
        #  of items in a gap to a method to the right.
        if(gap_span[1]-pr > 0):
            (i,j, (ms, me)) = gap_r
            (_k, meth_span, alignment, meth_edits) = change_parts[i][1][j]
            (ms, me) = meth_span
            meth_span = (pr,me)
            # print("meth_span", meth_span)
            meth_edits.append(('delete',0, ms-pr))
            # alignment = np.concatenate((alignment, np.arange(pr,ms)))
            alignment = np.concatenate((-np.ones(ms-pr,dtype=np.int64), alignment))
            change_parts[i][1][j] = (_k, meth_span, alignment, meth_edits)

            # ((sp, ep, main_k), meth_mods) = change_parts[i]            
            # change_parts[i] = ((sp-(ms-pr), ep, main_k), meth_mods)
            # print("JOIN <!!", i,j, (ms, me),"->", meth_span, meth_edits[-1])

        # print(pl, pr)
        # Case: _best_parition_inds has not assigned some subsequence
        #  of items in a gap to a method (right/left), so we need to 
        #  insert them into the parent method. 
        if(pr-pl > 0):
            if(gap_l):
                (i,_, _) = gap_l
                (_, ins_ind, _), _ = change_parts[i]

            else:
                (i,_, _) = gap_r
                (ins_ind, _, _), _ = change_parts[i]
            
            print("GS INT", i, ind, ind)
            edits.append(('insert', ins_ind, ins_ind, methods[k].items[pl:pr]))

        # print(k, gap_l, gr)
        # print(repr(l_sa), inner_sas, repr(r_sa))

    # for k, (i,j, (ms, me)) in enumerate(prevs):
    #     if(me < len(methods[k])):
    #         print(k, (i,j, (ms, me)))



            # m_ends[me] = 
            # # TODO: Make work for 3+ methods
            # k0s, k0e = spans[0]
            # k1s, k1e = spans[1]
            # if(k0e-k0s == 0):
            #     edits.append(('insert', k0s, k0e, methods[1].items[k1s:k1e]))
            # elif(k1e-k1s == 0):
            #     edits.append(('delete', k0s, k0e))
            # elif(methods[1].unordered):
            #     edits.append(('unorder', k0s, k0e, methods[1].items[k1s:k1e]))
            # else:
            #     edits.append(('replace', k0s, k0e, methods[1].items[k1s:k1e]))

        
        # for k in range(len(methods)):
        #     meth_edits.append((
        #         k, # Method index 
        #         spans[k], # Span within method
        #         meth_map[part[0]:part[1], k], # Alignment
        #         []#edits if k == main_k else [] # Edits
        #     ))
        #     print(part,":", meth_edits[-1])
        
        # change_parts.append((part, meth_edits))

            # print(i, "imputed", spans)


    # print("Edits", edits)
    # print("Post Edits", part_edits)
    for part, meth_edits in change_parts:
        for me in meth_edits:
            print(">", part,":", me)
    # pts = [] # ??
    # erasures = []
    # for k in range(1,len(method_apps)):
    #     sk, ek = spans[k]
    #     # print("spans", k , spans[k])
    #     rc = RHSChange(method_apps[k], span=(sk,ek), subseq=seq[sk:ek], edits=[('erase',0,0)])
    #     erasures.append(ParseTree(rc)) # ??

    mc =  MergeChange(macro_app, methods, change_parts, span=(s0,e0), subseq=seq[s0:e0], edits=edits)

    # rc = RHSChange(method_apps[0], span=(s0,e0), edits=edits, subseq=seq[s0:e0])
    child_trees = child_trees = [seq_trees[j] if seq_trees else None for j in range(s0,e0)]
    # return [ParseTree(rc, child_trees+erasures)]
    return ParseTree(mc, child_trees)
        # for k, span in enumerate(spans): 
        #     if(span is None):


    # print(partitions)


    # Use meth_map to derive edits on method
    #   k=0 from by merging with other methods.
    # edits = []
    # child_trees = [None]*len(meth_map)
    # for i, (mp,ap) in enumerate(zip(meth_map, aligned_map)):
    #     # If any overlap apply replace edits
    #     if(np.sum(mp >= 0) > 0):
    #         ks = np.nonzero((mp >= -1))[0]
    #         repl0 = 0 in ks
    #         for k in ks:
    #             if(k == 0):
    #                 continue
    #             item = methods[k].items[mp[k]]
    #             edits.append(('replace', i, i+repl0, [item]))

    #     # Insert or delete unaligned sections
    #     if(np.sum(ap) == 0):
    #         if(mp[0] >= 0):
    #             edits.append(('delete', i, i+1))
    #         else:
    #             ks = np.nonzero((mp >= -1))[0]
    #             to_ins = []
    #             for k in ks:
    #                 if(k == 0):
    #                     continue
    #                 item = methods[k].items[mp[k]]
    #                 to_ins.append(item)
    #             edits.append(('insert', i, i, to_ins))



        # # Target method k=0 aligns
        # if(mp[0] >= 0):
        #     # Insert if other not overlap or align
        #     if(np.any(mp[1:])==-2):
        #         pass
        #     # Disjoin if other not overlap or align
        #     elif(np.any(mp[1:])==-1):
        #         pass
        #     # Otherwise they both align
        #     else:
        #         pass

        # # Target method k=0 overlap but not align
        # elif(mp[0] == -1):
        #     # Insert if others not overlap or align
        #     if(np.any(mp[1:])>=0):
        #         pass
        #     # Disjoin if others not overlap or align
        #     elif(np.any(mp[1:])==-1):
        #         pass
        #     # Otherwise they both align
        #     else:
        #         pass
        # # Target method k=0 not overlap or align
        # elif(mp[0] == -2):
        #     # Delete 



        
        # if(k == 0):
        #     of_alignment0 = of_alignment

        #     for i, ind in enumerate(of_alignment):
        #         # Case 1: Item i of the target method aligned
        #         #    with the app at 'ind'
        #         if(ind >= 0):
        #             print(len(pt.child_trees),ind)
        #             child_trees[ind] = pt.child_trees[ind]
        #         # Case 2: Item i of the target method did not 
        #         #  align with the app at 'ind' so delete the item.
        #         else:
        #             edits.append(('delete', i, i+1))
        # else:
        #     # align0 = np.full(len(meth0), -1, dtype=np.int64)
        #     for i, ind in enumerate(of_alignment):
        #         items = [rc.method.items[i]]
        #         # Case 3: An item i in another method parsed 
        #         #   the app at'ind'
        #         if(ind >= 0):

        #             # indices = np.nonzero(ind==of_alignment0)[0]
        #             eq_i = of_alignment0[i]
        #             print("eq_i", eq_i)
        #             # if(len(indices) > 0):
        #             if(eq_i < 0):
        #                 # Case 3.1: An item in the target also aligned
        #                 #  with app at 'ind' so disjoin that item
        #                 #  with with item i.
        #                 # eq_i = indices[0]
        #                 items = [rc.method.items[i]]
        #                 edits.append(('replace', i, i+1, items))
        #                 # child_trees[ind] = pt.child_trees[i]
        #             else:
        #                 # Case 3.2: ??
        #                 pass
        #         else:
        #             # Case 4: ??
        #             # Insert items that this method didn't cover in 
        #             #   the seq into the joint method
        #             edits.append(('insert', i, i+1, items))
        # print("align0", align0)
        # print(find_edits(meth0.items, rc.method.items, align0, return_align=True))
    # edits = sorted(edits, key=lambda x: x[1])
    
    
    # rc = RHSChange()
    # print(edits) 

def _detect_disjoint_parses(parse_trees, seq, seq_trees):
    # Group parse trees by their parent macros
    groups = {}
    for pt in parse_trees:
        rc = pt.change
        parent_app = rc.method_app.parent
        # print("PARENT APP", parent_app, type(parent_app))

        # Only bother disjoining skill apps that align at all
        #  and have at least one full match
        if(np.sum(rc.align_scores) > 0.0 and 1.0 in rc.align_scores):
            arr = groups.get(parent_app, [])
            arr.append(pt)
            groups[parent_app] = arr

    # print(groups)
    # If they don't share a macro then do nothing 
    if(all(len(g) <= 1 for g in groups.values())):
        return parse_trees

    print()
    print("&&&&&&&&&&&&&&&&&&&&")
    # Otherwise multiple methods with a valid parse,
    #  are disjoined under the same macro. So merge them.
    new_parse_trees = []
    for macro_app, ptrees in groups.items():
        if(len(ptrees) == 1):
            new_parse_trees.append(ptrees[0])
            continue

        # TODO: If the align_scores of the trees
        #  all go favor one method then just use that one
        # score_sets = np.zeros((len(ptrees), len(seq)))
        max_align_scores = np.zeros(len(seq),dtype=np.float64)
        max_align_k = -np.ones(len(seq),dtype=np.int64)
        for k, pt in enumerate(ptrees):
            (s0,_) = pt.change.b_span
            scores, align = pt.change.align_scores, pt.change.b_alignment
            for i, (s, ind) in enumerate(zip(scores, align)):
                print(k, ind+s0, s)
                if(s > 0.0 and ind >= 0 and 
                    s > max_align_scores[ind+s0]):
                    max_align_scores[ind+s0] = s
                    max_align_k[ind+s0] = k

        print("MAX", max_align_scores, max_align_k)

        if(len(np.unique(max_align_k[max_align_k != -1])) == 1):
            print("SKIP")
            for k, pt in enumerate(ptrees):
                new_parse_trees.append(pt)
            continue
        
        merge_pt = _make_merge(macro_app, ptrees, seq, seq_trees)
        new_parse_trees.append(merge_pt)
        # print(merge_pt.b_span)
    print("&&&&&&&&&&&&&&&&&&&&")
    print()
    return new_parse_trees
           
                #     if(ind >= 0):
                #         pt0_alignment[ind] = i
                
            # seq_align = np.full(e0-s0, -2, dtype=np.int64)
            # print(rc.span[0]-s0,rc.span[1]-s0, rc.alignment)
            # seq_inds = np.nonzero(rc.alignment >= 0)[0]
            # print(seq_inds)
            # seq_align[seq_inds-(rc.span[0]-s0)] = rc.alignment[seq_inds]
            # print(rc.alignment, seq_align)
            # print([str(x) for x in seq[rc.span[0]:rc.span[1]]])
        # print()




    
    # print()
def print_seq_trees(seq_trees):
    if(seq_trees is None):
        print("SEQTREES:", None)
        return
    print("SEQTREES:")

    s = ""
    for pt in seq_trees:
        if(isinstance(pt, ParseHole)):
            s += f"{pt.skill_app}, {pt}, {pt.span}\n"
        else:
            subseq = pt.change.subseq
            s += F"{subseq[0]}, {pt}, {pt.span}, {getattr(pt.change,'scores', None)}\n"
            for i in range(1,len(subseq)):
                item_app = subseq[i]
                s += f"{item_app}, {pt.span}\n"
    print(s)
    return

# def _realign_parses()

def _bottom_up_recurse(grammar, seq, match_tree, depth, seq_trees=None):
    print("\nSEQ", f'[{" ".join([str(x) for x in seq])}]', len(seq), len(seq_trees) if seq_trees else None)
    # print("\nSEQTREES", f'[{" ".join([str(x) for x in seq_trees])}]'  if seq_trees else None)
    print_seq_trees(seq_trees)

    # print("app_pair_groups", app_pair_groups)

    # Should never enter terminal case at statrt
    if(depth not in match_tree.trees_by_depth):
        raise ValueError(f"No match trees for depth {depth}")
        # return seq_trees

    # Try to parse using each rhs
    # match_tree.trees_by_depth[depth]
    depth_match_trees = match_tree.trees_by_depth[depth]
    print("-DEEP", depth, depth_match_trees)

    best_changes = get_best_changes(grammar, seq, seq_trees, depth_match_trees)
    # print("best_changes", [(rc, f"{rc.cost:.2f}", rc.span) for rc  in best_changes])
    # best_changes = _only_least_cost(best_changes)

    # Turn each partial parse into a parse tree 
    parse_trees = []
    for rc in best_changes:
        print(":", rc, rc.span, rc.cost, rc.gen_penl, seq[rc.span[0]:rc.span[1]])
        child_trees = [seq_trees[j] if seq_trees else None for j in range(*rc.span)]
        pt = ParseTree(rc, child_trees)
        parse_trees.append(pt)

    
    parse_trees = _detect_disjoint_parses(parse_trees, seq, seq_trees)
    print("P TEE", parse_trees)

    #  If an insertion is required use ParseHoles to move it
    #   to the lowest possible section of the parse. 
    # print("B parse_trees", parse_trees)
    # for pt in parse_trees:
    #     pt.child_trees = _fill_parse_holes(pt.child_trees)
    # parse_trees = [_fill_parse_holes(pt) for pt in parse_trees]
    # print("A parse_trees", parse_trees)
    # for k, tree_seq in enumerate(get_spanning_trees(parse_trees, seq)):
    #     for i, pt in enumerate(tree_seq):
    #         rc = pt.change
    #         print((k,i), ":::", r.rhs.symbol, rc.span, f"{rc.cost:0.2f}", rc.edits)
    
    sp_trees = get_spanning_trees(parse_trees)
    # print("sp_trees", sp_trees)
    # sp_trees = filter_spanning_trees(sp_trees)

    out = []
    for k, tree_seq in enumerate(sp_trees):
        # print("TS", [(str(t),t.span) for t in tree_seq])
        
        # print("----<", seq)
        # change_seq = fill_span_holes(change_seq, seq)
        # for i, pt in enumerate(tree_seq):
        #     rc = pt.change
        #     print( (k, i), ":", rc.rhs.symbol, rc.span, f"{rc.cost:0.2f}", rc.edits)

        # Make the new sequence by replacing each method_app
        #  with its upstream macro_app. Keep track of any ParseHoles. 
        # print("0:", tree_seq)

        new_trees, new_seq = _make_seq_and_holes(tree_seq, seq, seq_trees, match_tree)
        # print("1:", new_trees, new_seq)
        # new_trees, new_seq = _parse_recursions(new_trees, new_seq)
        # print("2:", new_trees, new_seq)
        # new_app_pair_groups = app_pair_groups[1:]

        # 
        if(depth <= 1):
            print("--BEFORE--", new_trees)
            new_trees, new_seq = _fill_parse_holes(new_trees, new_seq, True)
            out += new_trees
            continue
        else:
            new_trees, new_seq = _fill_parse_holes(new_trees, new_seq)

        # if(len(new_app_pair_groups) == 0):
        #     _fill_parse_holes()            

        out += _bottom_up_recurse(grammar, new_seq, match_tree, depth=depth-2, seq_trees=new_trees)

    # print("OUT", out)
    return out

# def group_method_apps(proposed_apps):
#     # First split into terminals and non-terminals
#     terminals = []
#     nonterminals = []
#     for i, skill_app in enumerate(proposed_apps):
#         print(f"depth={skill_app.depth}", skill_app)
#         skill = skill_app.skill
#         if(isinstance(skill, MethodSkill)):
#             if(all([isinstance(x, PrimSkill) for x in skill.items])):
#                 terminals.append((skill_app, skill.items))
#             else:
#                 nonterminals.append((skill_app, skill.items))

#     print("terminals", terminals)
#     print("nonterminals", nonterminals)
#     # TODO: Should actually require that all item Macros
#     #  are covered.  
#     # Then group the apps by inspecting their parents
#     lowest = terminals
#     rest = nonterminals
#     covered_macro_ids = set()
#     groups = []
#     while(len(lowest) > 0):
#         parents = []
#         for j, (method_app, _) in enumerate(lowest):
#             covered_macro_ids.add(id(method_app.parent))

#         for i in range(len(rest)-1, -1, -1):
#             app_pair = rest[i]
#             _, items = app_pair
#             if(all([id(x) in covered_macro_ids for x in items if isinstance(x, MacroSkill)])):
#                 parents.append(app_pair)
#                 del rest[i]

#         groups.append(lowest)
#         lowest = parents

#     # print("$$", groups)
#     for group in groups:
#         print("###", [str(x[0]) for x in group])

#     return groups

# def _assign_parse_holes(pt):

def group_app_pairs(proposed_apps):
    app_pair_groups = []
    curr_depth = np.inf
    curr_grp = None
    for skill_app in proposed_apps:
        # print(f"D={skill_app.depth}", skill_app)
        skill = skill_app.skill
        if(isinstance(skill, MethodSkill)):
            pair = (skill_app, skill_app.child_apps)
            if(skill_app.depth < curr_depth):
                curr_grp = [pair]
                app_pair_groups.append(curr_grp)
                curr_depth = skill_app.depth
            else:
                curr_grp.append(pair)
    return app_pair_groups

def annotate_exact_match(match_tree, skill_app):
    skill_app.exact_match = match_tree.find_exact_match(skill_app)


def parse_w_changes(grammar, seq, state=None):
    # print("SEQ", seq)
    # prods = grammar.prods
    # print(prods)

    # top_down_align(grammar, seq)
    if(state is None):
        state = seq[0].state

    root = grammar.root_symbols[0]

    from proclrn.base import build_macro_match_tree

    match_tree = build_macro_match_tree(root, state)
    # build
    # proposed_apps = []
    # root_apps = root.get_apps(state)
    # for root_app in root_apps:
    #     proposed_apps += root_app.rollout_child_apps()

    # for app in proposed_apps:
    #     print(f"{' '*app.depth}d={app.depth}", repr(app), f"LC={len(app.child_apps) if app.child_apps else 0}")


    # # print("**", [x[0].depth for x in prop_app_pairs])
    # proposed_apps = sorted(proposed_apps, key=lambda x:x.depth, reverse=True)
    # item_apps = [x[1] for x in prop_app_pairs]

    # print("************")
    # print(proposed_apps)
    # Group skill apps by depth
    # apps_by_depth = {}
    # curr_grp = [proposed_apps[0]]
    # app_pair_groups = [curr_grp]
    

    
    # app_pair_groups = group_app_pairs(proposed_apps)
    # print("app_pair_groups", app_pair_groups)
        # arr = apps_by_depth.get(skill_app.depth)
        # arr.append(skill_app)
        # apps_by_depth[skill_app.depth] = arr
    # app_pair_groups = [app_pair_groups[key] for key in sorted(apps_by_depth)]


    # app_pair_groups = group_method_apps(proposed_apps)
    # for group in app_pair_groups:
    #     for method, sa_items in group:
    #         # print(">", sa_items, ":", [seq[ind] for ind in alignment])
    #         print(method.depth, ">", method, sa_items)
    print(match_tree)
    print("MAX METH DEPTH", match_tree.max_depth-1)

    for sa in seq:
        annotate_exact_match(match_tree, sa)

    parse_trees = _bottom_up_recurse(grammar, seq, match_tree, match_tree.max_depth-1)
    # parse_trees = [_assign_parse_holes(pt) for pt in parse_trees]
    print("\n  parse_trees:")
    for pt in parse_trees:
        print(f": {pt.cost:.2f}", )
        print(pt)

    parse_tree = sorted(parse_trees, key=lambda x:x.cost)[0]
    parse_tree.grammar = grammar
    # raise ValueError()

    # print("PT", parse_tree.rhs, parse_tree.cost, len(parse_tree.child_trees))
    # print(all_changes
    return parse_tree


def parse_only_deletes(pt):
    # TODO need to account for spanning deletes here
    rc = pt.change
    edits = rc.edits
    for edit in edits:
        kind = edit[0]
        if(kind != "delete"):
            return False

    # Check that the deletes are contiguous
    #  and terminate at end of rhs.
    if(len(edits) > 0 and not rc.method.unordered):
        d0 = edits[0][1] #First Delete
        k = 1
        # print("d0", d0)
        for i in range(d0+1, len(rc.method.items)):
            # print(":", k, len(edits), i, len(rc.rhs.items))
            if(k >= len(edits)):
                return False
            if(rc.method.is_optional(i)):
                continue
            # print(edits[k][1], i)
            if(edits[k][1] != i):
                return False
            k += 1

    for ct in pt.child_trees:
        if(ct.change.nomatch):
            continue
        if(not parse_only_deletes(ct)):
            return False
    return True



def _new_symbol_name(g, s_ind):
    while(True):
        _ord = 65+s_ind
        if(_ord <= 90):
            name = chr(_ord)
        else:
            name = "NT" + str(_ord-90)
        s_ind += 1
        if(name not in g.symbols):
            break
    return name

def apply_grammar_changes(g, parse_tree):
    if(hasattr(parse_tree, 'grammar')):
        if(g is not parse_tree.grammar):
            raise ValueError("Parse not associated with this grammar.")

    s_ind = 0
    symbol_factory = SymbolFactory(grammar=g)
    new_g = g.__copy__()
    # changes = parse_tree.changes
    print("------------")
    print(parse_tree)
    for rc in parse_tree.changes:
        new_g, _, new_rhs, _ = rc.apply(new_g, symbol_factory)

    new_g._simplify()
    return new_g

    # for rc in changes:
    #     print("_+", rc, rc.method_app) 
    # changes = sorted(changes, 
    #     key=lambda rc:rc.method_app.depth, reverse=True)
    # rhs_to_rcs = {}
    # for rc in changes:
    #     lst = rhs_to_rcs.get(rc.method,[])
    #     lst.append(rc)
    #     rhs_to_rcs[rc.method] = lst

    # new_prods = {}
    # for symb, rhss in g.prods:
    #     for rhs in rhss:
    #         rc = changes[rhs]

    
    # for rc in changes:
    #     rc.apply(new_g, symbol_factory)

    # for rhs, rcs in rhs_to_rcs.items():
    #     for rc in rcs:
    #         new_g, _, new_rhs, _ = rc.apply(new_g, symbol_factory)
            # print('>>', rc.edits, new_rhs)
            # print(new_g)
        # for i in range(1, len(rcs)):
            # print('>>', rc.edits, rcs[i].edits)
            # rc.edits = join_edits(rc.edits, rcs[i].edits)
        


    # print(new_g)
    

    
def generalize_from_seq(g, seq):
    print("----", seq, "-----")
    pt = parse_w_changes(g, seq)
    g = apply_grammar_changes(g, pt)
    for rhss in g.prods.values():
        for rhs in rhss:
            print(">", [(x.how_part, x.where_matches, x.skill_apps) for x in rhs.items if isinstance(x,SkillBase)])
    return g

'''
 Thoughts 11/11/23
 1.1 Without aligning to the input sequence we can
      only call get_matches() on the root symbol.
      Because we cannot get valid states for the 
      lower symbols without first aligning to the 
      sequence of PrimSkill Applications.   
 1.2 get_matches() can also easily fail if it is
      too specific, might need to:
      1.2.1 use a reduced set of conditions in this
        case (at least not .value)
      1.2.2 Or perhaps just use a MostSpecific
        implementation. Although, MostSpecific won't 
        generalize well with recursion.  
 1.3 Seems advantageous to implement some kind of
      partial matching system to circumvent this
      issue.

 2.1 To make use of lower symbols it is necessary to 
       either implement:
    2.1.1: get_matches() so that a subset of the matches
        can be set by the caller, and filled in
    2.1.2: or alternatively have some kind of mapping of 
       the inputs to Vars with dereference chains which 
       can be resolved into the missing arguments.
 
 Implementing 1.3 and 2.1.1 would take some time so it is 
    probably best to do 1.2.1 and 2.1.2. This opens an issue
    of the chains being unresolvable. In this case it is 
    probably fine for these missing pieces to be skipped over
    for the sake of alignment but filled in on parsing.
 [] 
'''

# def propose_child_apps(macro, state):        
#     apps = []
#     whr_lrn = macro.where_lrn_mech
#     # with PrintElapse("get_matches"):
#         # itr = whr_lrn.get_partial_matches(state)
#     itr = whr_lrn.get_matches(state)
#     # itr = list(itr)
#     # print("EN", len(itr))
#     # raise ValueError()
#     for match in itr:
#         print("MATCH", [m.id for m in match])

#         macro_app = SkillApp(macro, match, state=state, depth=0)
#         apps += rollout_child_apps(macro_app, depth=1)        
#     return apps

# def get_skill_app_matches(macro):
#     skill_apps = []
#     for i, method in enumerate(macro.methods):
#         skill_apps = method.get_apps()
#         method.child_arg_mapping

# def rollout_child_apps(macro_app, depth=0):
#     app_pairs = []
#     for method_app in macro_app.get_child_apps():
#         item_apps = method_app.get_child_apps()
#         app_pairs.append((method_app, item_apps))
#         method_app.depth = depth
#         for item_app in item_apps:
#             item_app.depth = depth
#             if(isinstance(item_app.skill, MacroSkill)):
#                 app_pairs += rollout_child_apps(item_app, depth+1)
#     return app_pairs



# def _propose_method_apps(g, method_app, state):
#     # print("===", method_app.skill, ":", seq)
#     method = method_app.skill
#     apps = []
#     # parse_trees = []
#     # s, e = 0, len(seq)
    
#     # If all symbols are terminal just run get_best_changes().
#     #  This could produce a full match or an edit.
#     for i, item in enumerate(method.items):
#         # print("ITEM", i, item, isinstance(item, MacroSkill))
#         if(isinstance(item, MacroSkill)):
#             # print(len(item.methods))
#             for _meth in item.methods:
#                 # print("M", _meth)
#                 for match in _meth.where_lrn_mech.get_matches(state):
#                     sa = SkillApp(_meth, match, state=state)
#                     child_apps = sa.get_child_apps()
#                     # child_apps = _propose_method_apps(g, sa, state)
#                     # print("child", child_apps)
#                     # apps.append(sa)
#                     # apps += [sa, *child_apps]
#                     # print(apps)
#     return apps

# def _is_term(sym):
#     return isinstance(sym, PrimSkill)
# ---------------------------------------
# : Top-down

def calc_align_score_matrix(item_match_trees, sequence):
    l0, l1 = len(item_match_trees), len(sequence)
    scores = np.zeros((l0, l1), dtype=np.float64)
    best_item_skill_apps = [None]*len(item_match_trees)

    for i, item_m_tree in enumerate(item_match_trees):
        max_score = 0.0
        best_item_skill_apps[i] = item_m_tree.skill_apps[0]
        for j, s1 in enumerate(sequence):
            
            score, m_skill_app = item_m_tree.overlap(s1)
            if(score < 1 and getattr(s1,"exact_match", None) is not None):
                # print(s1, item_m_tree.skill, (s1.exact_match.parent is item_m_tree.parent) and s1.exact_match.parent.skill)
                if(s1.exact_match.parent.parent is item_m_tree.parent.parent and
                    s1.exact_match.parent is not item_m_tree.parent):
                    pass
                else:
                    # print("MEEE HAPPENED", repr(s1), s1.exact_match)
                    scores[i][j] = 0.0
                    continue
            else:
                pass
                # print("NO EXACT MATCH", repr(s1))
                

            scores[i][j] = score
            if(score > max_score):
                best_item_skill_apps[i] = m_skill_app

    return scores, best_item_skill_apps


# import numba
# np_sparse_score_type = np.dtype([
#     ('score', np.float64),
#     ('i', np.uint32),
#     ('j', np.uint32),
# ])

# sparse_score_type = numba.from_dtype(np_sparse_score_type)

from scipy import sparse

@njit(cache=True)
def align_greedy(score_matrix, score_thresh):
    N, M = score_matrix.shape
    alignment = -np.ones(N, dtype=np.int64)
    covered = -np.ones(M, dtype=np.int64)
    scores = np.zeros(N, dtype=np.float64)

    c = 0
    min_ind, max_ind = 99999, -1
    for flat_ind in np.argsort(score_matrix.flatten())[::-1]:
        i, j = flat_ind // M, flat_ind % M
        score = score_matrix[i,j]

        # Early stop
        if(score < score_thresh or c == N):
            break
                    
        if(alignment[i] == -1 and j not in covered):
            # print("ij", i, j)
            alignment[i] = j
            scores[i] = score
            covered[c] = j
            min_ind = min(j, min_ind)
            max_ind = max(j, max_ind)
            c += 1
    min_ind = min(min_ind, max_ind+1)
    # print(alignment)
    return scores, alignment, (min_ind, max_ind+1)

def best_alignment(method_match_tree, sequence, score_thresh=0.001):

    # TODO: Maybe should be a loop
    item_trees = method_match_tree.child_trees[0]
    # for i, item_trees in :

    score_matrix, item_m_skill_apps = calc_align_score_matrix(item_trees, sequence)
    # print(item_m_skill_apps)
    # print(score_matrix)
    scores, alignment, span = align_greedy(score_matrix, score_thresh)
    print("<<",method_match_tree.parent.skill, "->", method_match_tree.skill, scores, alignment, sequence[span[0]:span[1]])
    
    return scores, alignment, span, item_m_skill_apps#/(l0)#+abs(l0-l1))

# #NOTE: Think I abandoned this
# def top_down_align(app_pairs, seq):
#     print("TDA")
#     if(isinstance(app_pairs, SkillGrammar)):
#         g = app_pairs
#         state = seq[0].state
#         app_pairs = propose_child_apps(g.root_symbols[0], state)
#     # print("-------------")
#     for method, items in app_pairs:
#         # print(type(items[0].skill), type(seq[0].skill))
#         overlap, alignment = best_alignment(items, seq)
#         print(">", items, ":", [seq[ind] for ind in alignment])
#         # print(seq[0].match)
#         # print(overlap, alignment)



# NOTE: 11/11/23
# Abandoning top down parsing for just generating
#  raw skill applications (the difference being)
#  we don't need to align the traversal with the 
#  input sequence.
# def top_down_parse(g, seq):
#     root = g.root_symbols[0]
#     state = seq[0].state

#     for method in g.methods:
#         for match in method.where_lrn_mech.get_matches(state):
#             method_app = SkillApp(method, match, state=state, depth=1)
#             # print([m.id for m in match])
#         # print(rhs)
#             return _top_down_parse(g, method_app, seq)

# import itertools        
# def _top_down_parse(g, method_app, seq):
#     print("===", method_app.skill, ":", seq)
#     method = method_app.skill
#     # print(g)
#     parse_trees = []
#     # for rhs in g.prods[sym]:
#     s, e = 0, len(seq)
#     # print("SEQ", seq)
#     # print(rhs.symbol, "->", rhs)
    
#     # If all symbols are terminal just run get_best_changes().
#     #  This could produce a full match or an edit.
#     are_term = [isinstance(x, PrimSkill) for x in method.items]
#     # if(all(are_term)):
#     #     best_changes = get_best_changes(g, seq, [method])
#     #     print(">>", seq, [rhs], best_changes)
#     #     if(len(best_changes) > 0):
#     #         rc = best_changes[0]
#     #         print(rc)
#     #         print("::", rc.rhs.symbol, rc.span, rc.cost)
#     #         return rc.span, [ParseTree(rc)]
    
#     if(False in are_term):
#         first_nt = are_term.index(False)   
#         last_nt = len(are_term) - 1 - are_term[::-1].index(False)

#         for i in range(0, first_nt):            
#             ov = method.items[i].overlap(seq[i])
#             print("1**", seq[i], method.items[i], ov)
#             if(ov != 1.0):
#                 break
#             s = i+1

#         for i in range(0, last_nt):
#             j = -i-1
#             ov = method.items[j].overlap(seq[j])
#             print("2**", seq[j], method.items[j], ov)
#             if(ov != 1.0):
#                 break
#             e = len(seq)+j

#         # print("NT", s, e, first_nt, last_nt)
    
#         # If not all terminal see how far can get left and right
#         if(not method.unordered):
#             # Otherwise return
#             for i in range(first_nt, last_nt+1):
#                 # print(rhs.items, i)
#                 item = method.items[i]
#                 # print(rhs.symbol, ">>", item, s)
#                 print("RECURSE", item, seq[s:e])
#                 child_trees = []
#                 if(isinstance(item, MacroSkill)):
#                     for _meth in item.methods:
#                         # print((s,e), seq)
#                         state = seq[s].state
#                         for match in _meth.where_lrn_mech.get_matches(state):
#                             sa = SkillApp(_meth, match, state)
#                             (s0, e0), pts = _top_down_parse(g, sa, seq[s:e])
#                             # print("s0", "e0", s0, e0, (s,e))
#                         # if(s0 == 0):
#                             s += e0
#                             child_trees.append(pts)
#                         # else:
#                         #     raise ValueError("Subproblem")
#                 # print(child_trees)
#                 for cts in itertools.product(*child_trees):
#                     # print(cts)
#                     # method, span, subseq=[], edits=[]
#                     pt = ParseTree(RHSChange(method, span=(0, len(seq)), seq=seq), child_trees=cts)
#                     parse_trees.append(pt)


#             # If terminal increment s
#             # if(_is_term(item, g)):
#             #     print(item, s, seq)
#             #     print(item, seq[s], seq[s] == item)
#             #     if(seq[s] == item):
#             #         s += 1
#             #     else:
#             #         print("BEF RET")
#             #         return (s, e), []
#             # # If non-terminal recurse
#             # else:
#     else:
#         L = min(len(method.items),len(seq))
#         for i in range(0, L):
#             ov = method.items[i].overlap(seq[i])
#             # print(">>", seq[i], method.items[i], ov)
#             if(ov != 1.0):
#                 break
#             s = i+1


#     return (s, e), parse_trees


def find_recursion(g):
    # print("START FIND REC")

    g = copy(g)
    sf = SymbolFactory(grammar=g)
    for sym, rhss in [*g.prods.items()]:
        for k, rhs in enumerate(rhss):
            part_slices = best_partitions(rhs)

            if(len(part_slices) < 2):
                continue

            # seq = rhs.items

            rec_sym = sf()
            act_sym = sf()    
            g.add_production(rec_sym, RHS([act_sym, rec_sym], optionals=[1]))

            for i in range(0, len(part_slices)):
                s, e = part_slices[i]
                if(i == 0):
                    recRHS = rhs[s:e]
                    recRHS.symbol = act_sym
                    g.add_production(act_sym, recRHS)
                else:
                    rc = find_rhs_change(apps_pair, rhs[s:e])
                    g, recRHS = rc.apply(g, sf)
                    # print("--",recRHS.eq_toks)

            s,e = part_slices[0][0], part_slices[-1][1]
                    
            
            # un_prod = None
            # if(recRHS.unordered):
            #     un_sym = sf()
            #     un_prod = (un_sym, recRHS)
            #     recRHS = RHS([un_sym])
            # print("INSERT",recRHS, len(recRHS))

            # recRHS.insert(len(recRHS), new_sym, optional=True)
            # print(recRHS)
            rhss[k] = rhs[:s]+RHS([rec_sym])+rhs[e:]

            
            
            

            # if(un_prod):
            #     g.add_production(*un_prod)                

            # print(rhs)
            # print(" ; ".join([f"{str(rhs[s:e])}" for s,e in part_slices]))
            # print("RC", recRHS, recRHS.eq_toks)
            # print()

    # print("BEF")
    # print(g)
    return g



def best_partitions(rhs):

    # print("RHS", rhs)

    # Look for repeated uses of skills with same how_parts in rhs
    skill_locs = {}
    for i, skill in enumerate(rhs.items):
        how_part = getattr(skill,'how_part', skill)
        slocs = skill_locs.get(how_part,[])
        slocs.append(i)
        skill_locs[how_part] = slocs

    # Sort skills by frequency and first->last span
    freq_len = lambda x : (len(x[1]), max(x[1])-min(x[1]))
    most_freq_how = sorted(list(skill_locs.items()), key=freq_len)[-1][0]
    n_recurse = len(skill_locs[most_freq_how])

    if n_recurse == 1:
        return []
    # print(skill_locs)
    # print(most_freq_skill, skill_locs[most_freq_skill])

    # Find skills that are interleaved with the skill that 
    #  occurs the most and has the longest first->last span
    intrl_skills = set([most_freq_how])
    fs_locs = skill_locs[most_freq_how]
    for i in range(1, len(fs_locs)):
        plocs, loc = fs_locs[i-1], fs_locs[i]
        # print(plocs, loc)
        for j in range(plocs, loc):
            item_j = rhs.items[j]
            intrl_skills.add(getattr(item_j,'how_part', item_j))

    # print(intrl_skills, [skill_locs[x] for x in intrl_skills])

    # Get the span of the interleaved skills
    s = min([skill_locs[s][0] for s in intrl_skills])
    e = max([skill_locs[s][-1] for s in intrl_skills])+1

    # print(rhs.items[s:e], s,e)
    

    part_overlaps = np.zeros(len(rhs))
    for i in range(s+1,e):
        p_i = rhs.items[i-1]
        x_i = rhs.items[i]
        if(hasattr(p_i, 'arg_overlap')):
            part_overlaps[i] = p_i.arg_overlap(x_i)

    part_slices = []
    while(len(skill_locs) > 0):
        # Find the minimal end point of the partition
        min_end = e
        for locs in skill_locs.values():
            if(len(locs) >= 2):
                if(locs[1] < min_end):
                    min_end = locs[1]

        # If that end point is at the end of the span of 
        # interleaved skills then stop looping
        if(min_end == e):
            break

        # Calulate the weight o
        part_weight = np.zeros(len(rhs))
        for locs in skill_locs.values():
            for i in range(1, len(locs)):
                ls, le = locs[i-1], locs[i]
                for j in range(0, le-ls):
                    # part_weight[le-j] += 1.0 / ((j+1) * len(skill_locs))
                    part_weight[le-j] += 1.0 / (len(skill_locs))

        score = part_overlaps[s:min_end+1] - part_weight[s:min_end+1]

        # NOTE: Using all min_inds works well for current test cases
        #  but might be finnicky overall.
        min_inds = (score == np.min(score)).nonzero()[0]
        # print("min_inds", min_inds)
        for i in range(len(min_inds)):
            part_end = max(min_inds[i], 1) + s
            start = max(s,min_inds[i-1] if i > 0 else 0)
            part_slices.append((start, part_end))

            # print(part_overlaps[s:min_end+1], part_weight[s:min_end+1])
            # print(part_overlaps, part_weight)
            # print("score", score, min_end)

            for how_part, locs in [*skill_locs.items()]:
                if(locs[0] <= part_end):
                    locs.pop(0)
                if(len(locs) == 0):
                    del skill_locs[how_part]
        s = part_end
            # min_inds -= 
            

    part_slices.append((s,e))
    # print("<<", part_slices)
    return part_slices 


# Thoughts 1/11/2024
'''
The idea of merging methods when a parse crosses a disjunction
allows us to essentially loosin the commitments made
by a previous disjunciton. Disjunctions are the most specific
edit, but impose structural commitments on the grammar which
may need to be altered later. 

If items from two methods with the same parent macro have
skill applications in the action sequence, then we are likely
in a situation like the following:

(C.1)
0 : a b
1 : c d
produces:

S -> a b | c d

but then:

2: a d 

Which implies that perhaps the actual grammar is something like:

S -> A B
A -> a | c
B -> b | d

We could represent this as:

a b [replace(0,1,[c]), replace(1,2,[d])]
c d [erase()]


(C.2) In another case:

S -> u a b | c d v

2: u a d v

S -> A B
A -> u a | c
B -> b | d v

with edits:

u a b [replace(0,2,[c]), replace(2,3, [d, v])]
c d v [erase()]


(C.3) However, we might find ourselves in a situation like

S -> a b | c d v

2: u a q d v

In which case how we treat q is somewhat ambiguous. We could
1. Make q optional after a
2. make q optional before d

For something like:
a b [insert(0,0,[u]), 
    replace(0,1,[c]), insert(1,1,[q]),
    replace(1,2,[d, v])]
c d v [erase()]

S -> u* A B
A -> a q* | c
B -> b | d v

However, we shouldn't
3. disjoin q with b
4. disjoin q with c

For something like this:

S -> u* A B v*
A -> a C
B -> b | d
C -> q | c

Since this actually wouldn't parse [c d v]

(C.4) In another case we might have 

0 : a b
1 : c d

S -> a b | c d

2: a b c d

In which case we find that S has two method parsings
which are adjacent instead of disjoint. Nontheless
if both are produced by top-down generation then we should
proceed in the same way. We'll get:

a b [delete(0,2), insert(0,2, [c d])]
c d [erase()]

S -> A* B*
A -> a b
B -> c d

Implementation:
1) Find all disjoint methods (i.e. having a common parent)
    which parse some part of the input sequence, and 
    thus must be merged.
2) Find the subsequence spanning the common parsed pieces.
2) Select one method to be edited... by inserting 
   elements of the other methods (which will be deleted)
3) Find the alignment of each method with the spanning  
    subsequence to align each method with the method 
    selected for editing. 
4) Apply find_edits()
    


# Thoughts 1/16/2024

The rules are something like:

Try to map each method onto the seq.
    This will be roughly outline the new method
1. For pieces which do not align (-1) and overlap
    with this region disjoin with the parts they
    overlap with.
2. For pieces which do not overlap across methods
    delete or insert them
3. Insert seq segments which do not align with anything
4. Delete seq segments which don't align or overlap with anything 


--Aligns S 1 1 1 1 # 0 0 0 0 --
Aligns T 1 1 0 0 # 1 1 0 0
Aligns O 1 0 1 0 # 1 0 1 0  
         1 2 3 4 5 6 7 8

1. They all align so keep
2. Target matches seq, but not other -> insert other
3. Target doesn't match seq, but other does -> delete target
3. Target doesn't match seq, but other does -> delete target


# Thoughts 1/11/2024
Merge edits may be more complex than I initially thought,
and they may not be composable from delete, insert, replace.

The main issue is that when a subset of a method is interleaved
with another we can have one of the methods stick around
but the other needs to be (moved) erased and reinserted
this raises an issue w.r.t to how the skill apps of
each item in the removed method are merged. 

This may require a special merge edit to be created. The
overall parse tree for the merge has a particular span
and alignment but there are actually multiple new subsets
of items from different methods which need to be associated 
with particular elements of the parsed sequence. 

A normal RC is associated with one method. So perhaps
it makes sense for merge changes to be a seperate datatype
with their own seperate apply() method. 

It's not clear how this could work if the methods don't
  share the same Macro. For instance:

S-> A | B
A -> a C b 
C -> [d e]
B -> f g

0: a d e b
1: a e d b
2: f g

3: a f e b

which implies we need to merge [d e] and f g
but they have macros C and B, with parents A and S.

S-> A | B
A -> a* C b*
C -> [E F]
B -> [E F]
E -> d | f
F -> e | g

As above we could replace each of the methods everywhere
with the new E F. Perhaps it is fine if the merged method
occurs twice, or perhaps some simplification would be better.
If they share a macro, the simplification would just be to
not write ? -> [E F] | [E F] since it is redundant.

Another consideration is how we identify these situations. 
We need to know that the two methods have a different 
history of parents which diverge at some point.

In any case MergeChange needs to be defined:
It has:
 1) a set of methods or method_apps
 2) a set of partitions w/ individual spans, alignments,
     and method inds.
 3) a total span 
 4) a total alignment
 4) a total subseq
 5) a set of edits within each partition 
 6) a get_cost function 
 6) an apply function 

It seems like RHSChange and MergeChange share elements 
with a potential Parse() object including:
    -method/method_app
    -span
    -subseq
    -alignment
    -gen_penl





'''
