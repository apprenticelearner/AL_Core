import numpy as np
from itertools import permutations
from copy import copy

# -------------------------------------
# : Edit Sequence

# From: https://blog.paperspace.com/implementing-levenshtein-distance-word-autocomplete-autocorrect/
def levDistances(seq1, seq2, replace_cost=1, insert_cost=1, delete_cost=.5, swap_cost=.5):
    distances = np.zeros((len(seq1) + 1, len(seq2) + 1))

    for i in range(1,len(seq1) + 1):
        distances[i][0] = i*delete_cost

    for j in range(1,len(seq2) + 1):
        distances[0][j] = j*insert_cost
        
    for i in range(1, len(seq1) + 1):
        for j in range(1, len(seq2) + 1):
            # Match case
            if (seq1[i-1] == seq2[j-1]):
                distances[i][j] = distances[i - 1][j - 1]
            else:
                d_insert = distances[i][j - 1] + insert_cost # insert
                d_delete = distances[i - 1][j] + delete_cost  # delete
                d_replace = distances[i - 1][j - 1] + replace_cost # replace
                
                m = min(d_insert, d_delete, d_replace)

                for l in range(2, i):
                    # print(l)
                    # if
                    if(i >= l and j >= l and 
                       seq1[i-l] == seq2[j-l+1] and seq1[i-l+1] == seq2[j-l]):
                        d_swap = distances[i - l][j - l] + swap_cost # swap
                        m = min(m, d_swap)

                distances[i][j] = m

    return distances

def minEdits(seq1, seq2, lev_dists=None):
    if(lev_dists is None):
        distances = levDistances(seq1, seq2)
    else:
        distances = lev_dists
    # print(distances)
    edits = []
    i, j = len(seq1), len(seq2)
    while(i > 0 or j > 0):
        d_curr = distances[i][j]
        d_insert = distances[i][j - 1] if i >= 1 else float('inf')
        d_delete = distances[i - 1][j] if i >= 1 else float('inf')
        d_replace = distances[i - 1][j - 1] if i >= 1 and j >= 1 else float('inf')
        d_swap = distances[i - 2][j - 2] if i >= 2 and j >= 2 else float('inf')
        m = min(d_insert, d_delete, d_replace)
        # if(i >= 2 and j >= 2):
        #     print(d_swap, m, seq1[i-2], seq2[j-1], seq1[i-1], seq2[j-2])
        if(i >= 2 and j >= 2 and 
           seq1[i-2] == seq2[j-1] and seq1[i-1] == seq2[j-2] and 
           d_swap <= m):

            edits.append(('swap', i-2, i-1))
            i, j = i-2, j-2

            continue

        # Note: in normal edit seq the index would be 'j' for 'delete' and 'j-1'  
        #  for 'replace' and 'insert'. Consecutive ('delete', 0,),('delete', 0,)  
        #  allows for in-place string changes, but for our grammar 'delete' causes 
        #  symbols to be optional. Thus the absolute index 'i-1' is desirable.
        if(m == d_replace):
            if(m < d_curr):
                edits.append(('replace', i-1, seq1[i-1], seq2[j-1]))
            i, j = i-1, j-1
        elif(m == d_delete):
            edits.append(('delete', i-1, seq1[i-1]))
            i = i-1
        elif(m == d_insert):
            edits.append(('insert', i-1, seq2[j-1]))
            j = j-1
    edits = list(reversed(edits))
    edits = replaceExtendedSwaps(edits)
    return edits, distances



            




def replaceExtendedSwaps(edits):
    repl_set = {}
    repl_set = {}
    for i, edit in enumerate(edits):
        if(edit[0] == 'replace'):
            repl_set[edit[2]] = (edit[1], edit[3], i)
    covered = set()
    for symA, (frm_ind, symB, e_indA) in repl_set.items():
        tup = repl_set.get(symB, None)
        if(tup is None): 
            continue
        
        (to_ind, target_A, e_indB) = tup
        if(symA == target_A and frm_ind not in covered):
            covered.add(to_ind)
            covered.add(frm_ind)
            i, j = min(e_indA, e_indB), max(e_indA, e_indB)
            edits[i] = ('swap', frm_ind, to_ind)
            edits[j] = None

    edits = [x for x in edits if x is not None]
    return edits

    # print(list(reversed(edits)))

# -------------------------------------
# : Grammar

class RHS:
    def __new__(cls, items, symbol=None, unordered=None, optionals=None):
        
        if(isinstance(items, RHS)):
            self = items.__copy__()
            if(symbol is not None):
                self.symbol = symbol
            if(unordered is not None):
                self.unordered = unordered
            if(optionals is not None):
                self.optional_mask = np.zeros(len(items))
                self.optional_mask[optionals] = 1
        else:
            self = super().__new__(cls)
            self.items = [Sym(x) if isinstance(x, str) else x for x in items]
            self.unordered = False if not unordered else unordered
            self.symbol = symbol
            self.optional_mask = np.zeros(len(items))
            if(optionals is not None):
                self.optional_mask[optionals] = 1

        return self


    def is_optional(self, ind):
        return self.optional_mask[ind]

    def set_optional(self, ind, is_optional=1):
        self.optional_mask[ind] = is_optional

    def __copy__(self):
        new_rhs = RHS([*self.items], symbol=self.symbol, unordered=self.unordered)
        new_rhs.optional_mask = self.optional_mask.copy()
        return new_rhs

    def __str__(self):
        item_strs = []
        for i, sym in enumerate(self.items):
            if(isinstance(sym, Sym)):
                name = sym.name
            else:
                name = str(sym)


            if(self.optional_mask[i]):
                item_strs.append(f"{name}*")
            else:
                item_strs.append(str(name))

        rhs_str = " ".join(item_strs)

        # underline if unordered
        if(self.unordered):
            rhs_str = f"\033[4m{rhs_str}\033[0m"                         

        return rhs_str

    __repr__ = __str__

    # RHS instances are only equal to themselves
    def __eq__(self, other):
        return self is other

    def __hash__(self):
        return id(self)


    def __len__(self):
        return len(self.items)

    def __getitem__(self, key):
        # Implement RHS Slicing
        if(isinstance(key, slice)):
            subitems = self.items[key]
            subopt = self.optional_mask[key]
            new_rhs = RHS(subitems)
            new_rhs.items = subitems
            new_rhs.optional_mask = subopt
            new_rhs.unordered = self.unordered
            new_rhs.symbol = self.symbol
            return new_rhs
        raise NotImplemented()


class Sym:
    def __new__(cls, name, prods=[]):
        if(isinstance(name, Sym)):
            return name

        self = super().__new__(cls)
        global symbol_index
        self.name = name
        
        self.prods = prods
        return self

    def add_RHS(self, items, unordered=False):
        self.prods.append(RHS(items, symbol=self, unordered=unordered))

    def __eq__(self, other):
        if(isinstance(other, str)):
            return self.name == other
        else:
            return self.name == other.name

    def __hash__(self):
        return hash(self.name)

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

# -----------------------------------------------
# : Grammar Change

def merge_overlapping_spans(spans):
    if(len(spans) == 0): 
        return spans
    sorted_spans = sorted(spans)
    merged_spans = []
    start, end = sorted_spans[0]

    for span in sorted_spans[1:]:
        if span[0] <= end:
            end = max(end, span[1])
        else:
            merged_spans.append((start, end))
            start, end = span

    merged_spans.append((start, end))
    return merged_spans


class Grammar:
    def __init__(self):
        self.symbols = {}
        self.prods = {}

    def add_symbol(self, symbol):
        if(isinstance(symbol, str)):
            symbol = Sym(symbol)

        if(symbol not in self.symbols):
            self.symbols[symbol] = symbol
            self.prods[symbol] = []
        symbol = self.symbols[symbol]
        return symbol


    def add_production(self, symbol, rhs):
        # if(isinstance(rhs, list)):
        rhs = RHS(rhs, symbol=symbol)

        symbol = self.add_symbol(symbol)

        for i, item in enumerate(rhs.items):
            item = self.add_symbol(item)
            rhs.items[i] = item

        self.prods[symbol].append(rhs)

    def __str__(self):
        s = ""
        for symb, prods in self.prods.items():
            name = symb.name if isinstance(symb, Sym) else str(symb)
            if(len(prods) > 0):
                rhs_strs = []
                for rhs in prods:
                    # rhs_str = "".join([sym.name for sym in rhs.items])

                    # # underline if unordered
                    # if(rhs.unordered):
                    #     rhs_str = f"\033[4m{rhs_str}\033[0m"                         

                    rhs_strs.append(str(rhs))
                s += f'{name} -> {" | ".join(rhs_strs)}\n'
        return s

    def __copy__(self):
        new_g = Grammar()
        new_g.symbols = {**self.symbols}
        new_prods = {}
        for symb, prods in self.prods.items():
            new_prods[symb] = [copy(rhs) for rhs in prods]
        new_g.prods = new_prods
        return new_g

    def _ensure_root_symbols(self):
        if(getattr(self, '_root_symbols', None) is None):
            self._root_symbols = []
            for sym, rhss in self.prods.items():
                if(len(self.get_RHSs_with(sym)) == 0):
                    self._root_symbols.append(sym)

    @property
    def root_symbols(self):
        self._ensure_root_symbols()
        return self._root_symbols


    @property
    def preterminal_RHSs(self):
        if(getattr(self, '_preterminals', None) is None):
            self._preterminal_RHSs = {}
            for sym, rhss in self.prods.items():
                for rhs in rhss:
                    all_items_terminal = True
                    for item in rhs.items:
                        if(len(self.prods.get(item,[])) != 0):
                            all_items_terminal = False
                            break
                    if(all_items_terminal):
                        self._preterminal_RHSs[rhs] = []
        return self._preterminal_RHSs


    def _ensure_rhss_with(self):
        if(getattr(self,'_rhss_with_map', None) is None):
            self._rhss_with_map = {}        
            for _, rhss in self.prods.items():
                for rhs in rhss:
                    for item in rhs.items:
                        lst = self._rhss_with_map.get(item,[])
                        lst.append(item)
                        self._rhss_with_map[rhs] = lst
        
    def get_RHSs_with(self, symbol):
        ''' Get all RHSs that contain symbol '''
        self._ensure_rhss_with()    
        return self._rhss_with_map.get(symbol,[])

    def _ensure_upstream_rhss(self):
        if(getattr(self,'_upstream_rhs_map', None) is None):
            self._upstream_rhs_map = {}
            syms = self.root_symbols
            # Base case roots have no upstream
            for sym in self.root_symbols:
                self._upstream_rhs_map[sym] = set()
            covered_syms = set() # track just in case of recursion
            while(len(syms) > 0):
                new_syms = []
                for sym in syms:
                    if(sym in covered_syms):
                        continue
                    sym_upstrm = self._upstream_rhs_map[sym]
                    for rhs in self.prods.get(sym,[]):
                        for _sym in rhs.items:
                            _sym_upstrm = self._upstream_rhs_map.get(_sym, set())
                            _sym_upstrm.add(rhs)
                            for x in sym_upstrm:
                                _sym_upstrm.add(x)
                            self._upstream_rhs_map[_sym] = _sym_upstrm
                            new_syms.append(_sym)
                        self._upstream_rhs_map[rhs] = sym_upstrm
                    covered_syms.add(sym)
                syms = new_syms


    def get_upstream_RHSs(self, sym_or_rhs):
        self._ensure_upstream_rhss()
        return self._upstream_rhs_map[sym_or_rhs]
        # if(isinstance(sym_or_rhs, RHS)):
        #     rhss = [sym_or_rhs]
        # else:
        #     rhss = self.get_RHSs_with(rhs_or_symbol) 
        # return rhss


    def _clear_internals(self):
        self._roots = None
        self._preterminals = None
        self._rhss_with_map = None
        self._upstream_rhs_map = None
        



    # def annotate_preterminals(self):
    #     for sym, prods in self.prods.items():
    #         for rhs in prods:
    #             is_pre_terminal = True
    #             for item in rhs.items:
    #                 if(len(self.prods.get(item,[])) > 0):
    #                     is_pre_terminal = False
    #             rhs.is_pre_terminal = is_pre_terminal

    # def _best_pre_terminals(self, seq):

    #     candidate_changes = [] 
    #     for symbol, prods in self.prods.items():
    #         for rhs in prods:
    #             if(rhs.is_pre_terminal):
    #                 l = len(rhs.items)
    #                 items = [x.name for x in rhs.items]
    #                 item_set = set(items)

    #                 best_gc = None
    #                 best_gc_cost = float('inf')

    #                 # Convolve application of 
    #                 for i in range(len(seq)):
    #                     for j in range(i+l, len(seq)+1):
    #                         subseq = seq[i:len(seq)]
    #                         intr = set(subseq).intersection(item_set)

    #                         # Skip if there is no overlap  
    #                         if(len(intr) == 0):
    #                             continue

    #                         edits, distances = minEdits(rhs.items, subseq)
    #                         gc = RHSChange(rhs, (i,j), edits)
    #                         if(gc.cost < best_gc_cost):
    #                             print(rhs.items, subseq)
    #                             print(edits)
    #                             print(distances)
    #                             print(gc)
    #                             best_gc_cost = gc.cost
    #                             best_gc = gc

    #                 print(rhs.items, seq[best_gc.span[0]:best_gc.span[1]])
    #                 print(best_gc)
    #                 print("---")
    #                 candidate_changes.append(best_gc)

        # print([gc.span for gc in candidate_changes])


                        # print(distances)
                        


    def _recurse_top_down(self, symbol, seq):
        self.annotate_preterminals()
        self._best_pre_terminals(seq)

        for rhs in self.prods[symbol]:
            # Pre-terminal case
            if(rhs.is_pre_terminal):
                offset = 0
                # print("pre-term:", rhs.items)
                l = len(rhs.items)
                cov_seq = seq[:l]

                if(rhs.unordered):
                    intr = set(rhs.items).intersection(set(cov_seq))
                    # print("INTR:", intr)
                else:
                    pass
                
                offset += l

            # Non-terminal case
            else:
                if(rhs.unordered):
                    symb_seqs = permutations(rhs.items)
                else:
                    symb_seqs = [rhs.items]

                for sym_seq in symb_seqs:
                    offset = 0
                    # print(sym_seq)
                    for i, item in enumerate(sym_seq):
                        # print('\t', f"{i}:{item}")
                        # Non-terminal case
                        # if(item in self.prods):
                        span = self._recurse_top_down(item, seq[offset:])
                        offset += span
                        # Terminal case
                        # else:

                        #     offset += 1

        return 0

    def top_down_parse(self, seq, start_symbol='start'):
        start_symbol = self.symbols[start_symbol]

        self._recurse_top_down(start_symbol, seq)


# A = ['A', 'B' ,'C', 'D', 'E', 'F']
# C = ['A', 'Q' ,'C', 'Z', 'E', 'F']
# B = ['B', 'A' ,'D' ,'C', 'F', 'E']

# g = Grammar()
# g.add_production("S", [Sym("T"), Sym("Q"), Sym("R")])
# g.add_production("T", RHS(["A", "B"], unordered=True))
# g.add_production("Q", RHS(["C", "D"], unordered=True))
# g.add_production("R", RHS(["F", "E"], unordered=True))


# print(g)

# g.top_down_parse(A, start_symbol='S')
# print("")
# g.top_down_parse(B, start_symbol='S')

# print("")

# g = Grammar()
# g.add_production("S", A)
# g.top_down_parse(B, start_symbol='S')

# print()
# g.top_down_parse(['C', 'B' ,'A' ,'F', 'E', 'D'], start_symbol='S')


# print("-----------------------------------------")
# g = Grammar()
# g.add_production("S", A)
# g.top_down_parse(['A', 'B' ,'C', 'X', 'D', 'E', 'F', 'Y'],start_symbol='S')


# ------------------------------------------------
# : Find Edits btw. two sequences 

def find_edits(seq0, seq1):
    eq_matrix = np.zeros((len(seq0), len(seq1)))
    for i, c1 in enumerate(seq0):
        for j, c2 in enumerate(seq1):
            if(c1 == c2):
                eq_matrix[i][j] = 1
    # print(eq_matrix)
    edits = []
    c = 0
    s0 = s1 = 0
    e0 = e1 = 1
    spanning = False
    for _ in range(max(len(seq0),len(seq1))):
        # print(f's0={s0} e0={e0} s1={s1} e1={e1}, c={c}')
        # Note: printing a copy with +.1 added to the span of s0:e1,s1:e1
        #  is helpful for ensuring that each iterator is changing correctly
        # eq_copy = eq_matrix.copy()
        # eq_copy[s0:e0,s1:e1] += .1
        # print(eq_copy)

        is_del = e0 <= len(seq0) and np.sum(eq_matrix[e0-1,s1:]) == 0
        is_ins = e1 <= len(seq1) and np.sum(eq_matrix[s0:,e1-1]) == 0

        # print(is_del, is_ins)
        if(is_del or is_ins):
            if(is_del and is_ins):
                # print(('replace', e0-1))
                
                edits.append(('replace', e0-1, e0, [seq1[e1-1]]))
            elif(is_del):
                # print(('delete', s0))
                edits.append(('delete', e0-1))
            else:
                # print(('insert', s0))
                edits.append(('insert', e0-1, [seq1[e1-1]]))

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
            s0, s1 = e0, e1

            if(e0 >= len(seq0) or e1 >= len(seq1)):
                break

            e0, e1 = e0+1, e1+1
            continue

        # Otherwise check for out order span
        if(np.sum(eq_matrix[s0:e0, s1:e1]) == e0-s0-c):
            # print(('unorder', s0, e0))
            edits.append(('unorder', s0, e0))
            s0, s1 = e0, e1
            spanning = False
            c = 0
        else:
            spanning = True

        if(e0 >= len(seq0) or e1 >= len(seq1)):
            break

        e0, e1 = e0+1, e1+1

    if(e0 < len(seq0)):
        for i in range(e0,len(seq0)):
            edits.append(("delete", i))

    if(e1 < len(seq1)):
        # print("THIS", seq1, s0, seq1[e1:])
        edits.append(("insert", s0, list(seq1[s1:])))

    edits = sorted(edits, key=lambda x: x[1])
    
    # print(edits)
    # Merge contiguous replace and insert edits
    # TODO: 
    prev_repl = None
    filtered_edits = []
    for edit in edits:
        kind = edit[0]
        if(prev_repl is not None):
            if(kind in ['insert','replace'] and edit[1] == prev_repl[2]):
                # print(prev_repl, edit)
                prev_val = prev_repl[3]
                # if(not isinstance(prev_val, list)):
                #     prev_val = [prev_val]
                if kind == 'replace':
                    val = prev_val+edit[3]
                    e = edit[2]
                else:
                    val = prev_val+edit[2]
                    e = prev_repl[2]

                prev_repl = ('replace', prev_repl[1], e, val)
            else:
                filtered_edits.append(prev_repl)
                filtered_edits.append(edit)
                prev_repl = None
        elif(kind == 'replace'):
            prev_repl = edit
        else:
            filtered_edits.append(edit)

        # if(kind == 'replace'):
        #     if(prev_repl is not None and edit[1] == prev_repl[2]):
        #         prev_repl = ('replace', prev_repl[1], edit[2], prev_repl[3]+edit[3])
        #     else:
        #         prev_repl = edit
        # else:
        #     if(prev_repl is not None):
        #         filtered_edits.append(prev_repl)
        #         prev_repl = None
        #     filtered_edits.append(edit)
    if(prev_repl is not None):
        filtered_edits.append(prev_repl)

    # print(filtered_edits)
    return filtered_edits

# --------------------------------------------
# : Modify RHS from Edits

def apply_rhs_edits(rhs, edits, unorder=False):
    new_rhs = copy(rhs)

    delete_edits = [edit for edit in edits if edit[0]=='delete']
    ins_repl_edits = [edit for edit in edits if edit[0] in ['insert', 'replace']]

    # For each delete edit make the corresponding symbol optional.
    for edit in delete_edits:
        new_rhs.set_optional(edit[1])

    if(len(ins_repl_edits) > 0):
        items_opt = list(zip(new_rhs.items, new_rhs.optional_mask))

        # For each insert edit add an optional symbol. For each replace 
        #  make the appropriate replacement with non-optional symbols.
        #  Apply in reversed order (should already be ordered by index)
        #  to prevent index invalidation
        for edit in reversed(ins_repl_edits):
            kind = edit[0]
            if(kind == "insert"):
                items_opt = items_opt[:edit[1]] + [(e,1) for e in edit[2]] + items_opt[edit[1]:]
            elif(kind == "replace"):
                new_syms = [(symb, False) for symb in edit[3]]
                items_opt = items_opt[:edit[1]] + new_syms + items_opt[edit[2]:]


        items = [x[0] for x in items_opt]
        optional_mask = np.array([x[1] for x in items_opt])
        new_rhs.items = items
        new_rhs.optional_mask = optional_mask
        

    if(unorder):
        new_rhs.unordered = True

    return new_rhs

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
            (sym, rhss, (s,e)) = unorder_substs[i]
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
        (sym, rhss, (s,e)) = unorder_substs[split_ind]
        rhs = rhss[-1] # Last RHS is always the new one
        rhss[-1] = apply_rhs_edits(rhs, edits)

    # for (sym, rhss, (s,e)) in [*unorder_substs, *disjoin_substs]:
    #     print(sym, '->', " | ".join([str(rhs) for rhs in rhss]))
    return filtered
    # non_unorder_edits = filtered

class SymbolFactory():
    def __init__(self, sym_ord=65, grammar=None):
        self.sym_ord = sym_ord
        self.grammar = grammar

    def __call__(self):
        while(True):
            if(self.sym_ord <= 90):
                name = chr(self.sym_ord)
            else:
                name = "NT" + str(self.sym_ord-90)
            self.sym_ord += 1
            if(self.grammar is None or 
                name not in self.grammar.symbols):
                break
        return name



def edits_to_changes(rhs, edits, symbol_factory=None):
    ''' Preprocess a set of edits into a set of RHS changes which can
        consist of fully unordering the parent rhs, direct edits to 
        the 'parent' rhs and substitutions of it's parts with new productions.
    '''
    # print("EDITS", edits)
    # seq1 = rhs.items

    if(symbol_factory is None):
        symbol_factory = SymbolFactory(ord("T"))

    # Look for an unorder edit that spans the whole RHS
    full_unorder = False
    for i, edit in enumerate(edits):
        if(edit[0] == 'unorder' and edit[1] == 0 and edit[2] == len(rhs)):
            full_unorder = True
            edits.pop(i)
            break

    any_new_sym = any([edit[0] in ["unorder", "replace"] for edit in edits])
    prev_end = 0 if any_new_sym else None

    # Make new symbols for each 'unorder' and 'replace' edit
    unorder_substs = [] # [(sym, [rhs,...], span),...]
    disjoin_substs = [] # [(sym, [rhs,...], span),...]
    unorder_edits = []
    non_unorder_edits = []
    for edit in edits:
        kind = edit[0]
        if(kind in ["unorder", "replace"]):
            s,e = edit[1], edit[2]

            # Fill any holes preceeding this edit
            # if(prev_end is not None and prev_end < s):
            #     new_sym = Sym(symbol_factory())
            #     rhs0 = RHS(seq1[prev_end:s], new_sym)
            #     disjoin_substs.append((new_sym, [rhs0], (prev_end,s)))
            #     non_unorder_edits.append(('replace', prev_end, s, [new_sym]))

            prev_end = e

            new_sym = Sym(symbol_factory())

            # UnorderSubst Case 
            if(kind == "unorder"):
                rhs0 = rhs[s:e] #RHS(seq1[s:e], new_sym, unordered=True)
                rhs0.unordered = True
                unorder_substs.append((new_sym, [rhs0], (s,e)))
                unorder_edits.append(('replace', s, e, [new_sym]))


            # DisjoinSubst Case 
            elif(kind == "replace"):
                new_seq = edit[3] if isinstance(edit[3], list) else [edit[3]]
                rhs0 = rhs[s:e]
                rhs0.symbol = new_sym
                # rhs0 = RHS(seq1[s:e], new_sym, unordered=rhs.unordered)
                rhs1 = RHS(new_seq, new_sym)
                # print("0:", rhs0)
                # print("1:", rhs1)

                if(e-s == len(rhs)):
                    # print("--ADD--", rhs.symbol, [rhs0, rhs1])
                    disjoin_substs.append((rhs.symbol, [rhs1], (s,e))) 
                else:
                    # print("--SUBST--", rhs.symbol, [rhs0, rhs1])
                    disjoin_substs.append((new_sym, [rhs0, rhs1], (s,e)))
                    # Swap the new symbol in for the RHS of the replace edit 
                    edit = ('replace', s, e, [new_sym])
                    non_unorder_edits.append(('replace', s, e, [new_sym]))
        else:
            # Insert and Delete Edits
            non_unorder_edits.append(edit)

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
    # for (sym, rhss, (s,e)) in unorder_substs:
    #     parent_edits.append(('replace', s, e, [sym]))
    # for (sym, rhss, (s,e)) in disjoin_substs:
    #     parent_edits.append(('replace', s, e, [sym]))
    parent_edits = sorted(parent_edits, key=lambda x:x[1])
    return (full_unorder, unorder_substs, disjoin_substs, parent_edits)


class RHSChange:
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
    def __init__(self, rhs, span, edits=[], nomatch=False, seq=None):
        self.rhs = rhs
        self.span = span
        self.edits = edits
        # self.cost = cost
        self.nomatch = nomatch
        self.seq = seq
        self.recalc_cost()

    def recalc_cost(self):

        if(self.nomatch):
            cost = self.span[1]-self.span[0]
        else:
            cost = 0.0
            for edit in self.edits:
                kind = edit[0]
                if(kind == "insert"):
                    cost += 1.0 * len(edit[2])
                elif(kind == "delete"):
                    # High enough that insertions between spans are better than deleting
                    cost += .85 
                elif(kind == "replace"):
                    cost += 1.0 * (edit[2]-edit[1]) + .25*len(edit[3])
                elif(kind == "unorder"):
                    # print(edit)
                    cost += .2*(edit[2]-edit[1])
            seq_len = self.span[1]-self.span[0]
            cost /= min(len(self.rhs.items), seq_len)
        self.cost = cost


    def ensure_changes_computed(self, symbol_factory=None):
        if(not hasattr(self, 'parent_edits')):
            full_unorder, unorder_substs, disjoin_substs, parent_edits = \
                edits_to_changes(self.rhs, self.edits, symbol_factory)
            self.full_unorder = full_unorder
            self.unorder_substs = unorder_substs
            self.disjoin_substs = disjoin_substs
            self.parent_edits = parent_edits

    def apply(self, grammar, symbol_factory=None):
        self.ensure_changes_computed(symbol_factory)

        print("P", self.rhs, self.parent_edits)
        old_rhs = self.rhs

        new_rhs = apply_rhs_edits(self.rhs, self.parent_edits, self.full_unorder)

        rhss = grammar.prods[old_rhs.symbol]
        rhs_ind = -1
        for i, _rhs in enumerate(rhss):
            # print(i, "EQ", str(_rhs),"==", str(old_rhs), str(_rhs) == str(old_rhs))
            if(str(_rhs) == str(old_rhs)):
                rhs_ind = i
                break
        # print(rhs_ind)
        if(rhs_ind == -1):
            raise KeyError(f"Grammar does not contain RHS {old_rhs}.")
        # rhs_ind = prod.index(old_rhs)
        rhss[rhs_ind] = new_rhs
        
        for (sym, rhss, (s,e)) in [*self.unorder_substs, *self.disjoin_substs]:
            for rhs in rhss:
                grammar.add_production(sym, rhs)
        return grammar, new_rhs


    def __str__(self):
        return f"{self.rhs.symbol}->{self.rhs} {self.edits}"

    def __copy__(self):
        return RHSChange(self.rhs, self.span, self.edits, self.nomatch, self.seq)



    #     rhs = self.rhs
    #     return f"changes({rhs.symbol}->{''.join([str(x) for x in rhs.items])}, {seq2})")



def find_rhs_change(rhs, subseq, span=None, seq=None):
    # print(f"changes({rhs.symbol}->{''.join([str(x) for x in rhs.items])}, {seq2})")

    seq1 = rhs.items
    edits = find_edits(seq1, subseq)
    # print(edits)

    # Filter out any edits that are redundant with 
    #  the exiting unorder / optionals of the RHS
    filtered_edits = []
    for edit in edits:
        kind = edit[0]
        if(rhs.unordered and kind == "unorder"):
            continue
        if(kind == "delete" and rhs.is_optional(edit[1])):
            continue
        filtered_edits.append(edit)

    
    # cost = cost_of_edits(rhs, filtered_edits, len(subseq))

    # print("****:", rhs, subseq, cost, edits)

    # print("COST", cost)
    if(span is None):
        span = (0, len(seq1))
    change = RHSChange(rhs, span, filtered_edits, seq=seq)
    
    return change
    # g = Grammar()
    # g.add_production("S", rhs)
    # new_g, new_rhs = change.apply(g)
    # print(g)

    # return new_rhs

def test_find_edits():
    # 0. Aligned
    print("0.")
    seq1 = 'ABCDEF'
    find_edits(seq1, seq1)
    print()

    # 1. Unorder
    print("1.")
    seq2 = 'CBAFED'
    assert find_edits(seq1, seq2) == \
        [('unorder', 0, 3), ('unorder', 3, 6)]


    # 2. Unorder + Aligned
    print("2.")
    seq2 = 'ACBDFE'
    assert find_edits(seq1, seq2) == \
        [('unorder', 1, 3), ('unorder', 4, 6)]


    # # 3. Delete
    print("3.")
    seq2 = 'ACDF'
    assert find_edits(seq1, seq2) == \
        [('delete', 1), ('delete', 4)]

    # # 4. Unorder + Delete
    print("4.")
    seq2 = 'CBED'
    assert find_edits(seq1, seq2) == \
        [('delete', 0), ('unorder', 1, 3), ('unorder', 3, 5), ('delete', 5)]


    # # 5. Unorder Subsume Delete
    print("5.")
    seq2 = 'CAFD'
    assert find_edits(seq1, seq2) == \
        [('unorder', 0, 3), ('delete', 1), ('unorder', 3, 6), ('delete', 4)]

    # # 6. Insert
    print("6.")
    seq2 = 'XABCDEFY'
    assert find_edits(seq1, seq2) == \
        [('insert', 0, ['X']),('insert', 6, ['Y'])]

    # # 7. Unorder + Insert
    print("7.")
    seq2 = 'XCBAFEDY'
    assert find_edits(seq1, seq2) == \
        [('insert', 0, ['X']), ('unorder', 0, 3), ('unorder', 3, 6), ('insert', 6, ['Y'])]

    # # 8. Unorder Subsume Insert
    print("8.")
    seq2 = 'CXBAFEYD'
    assert find_edits(seq1, seq2) == \
        [('unorder', 0, 3), ('insert', 1, ['X']), ('unorder', 3, 6), ('insert', 5, ['Y'])]

    # # 9. Replace
    print("9.1")
    seq2 = 'XYZDEF'
    assert find_edits(seq1, seq2) == \
        [('replace', 0, 3, ['X','Y','Z'])]

    print("9.2")
    seq2 = 'ABXYEF'
    assert find_edits(seq1, seq2) == \
        [('replace', 2, 4, ['X','Y'])]

    print("9.3")
    seq2 = 'ABCXYZ'
    assert find_edits(seq1, seq2) == \
        [('replace', 3, 6, ['X','Y','Z'])]

    # # 10. Unorder + Replace
    print("10.")
    seq2 = 'BAXYFE'
    assert find_edits(seq1, seq2) == \
        [('unorder', 0, 2), ('replace', 2, 4, ['X','Y']), ('unorder', 4, 6)]

    # # 11. Unorder + Replace
    print("11.1")
    seq2 = 'CXAFYD'
    assert find_edits(seq1, seq2) == \
        [('unorder', 0, 3), ('replace', 1, 2, ['X']), ('unorder', 3, 6), ('replace', 4, 5, ['Y'])]

    # 12. Edge Cases
    print("12.1")
    seq2 = 'CXZAFYQD'
    assert find_edits(seq1, seq2) == \
        [('unorder', 0, 3), ('replace', 1, 2, ['X','Z']), ('unorder', 3, 6), ('replace', 4, 5, ['Y','Q'])]

    print("12.2")
    seq2 = 'CXYZQRV'
    assert find_edits(seq1, seq2) == \
        [('delete', 0), ('delete', 1), ('replace', 3, 6, ['X','Y','Z','Q','R','V'])]
    

    print("12.3")
    seq2 = 'XYZ'
    assert find_edits(seq1, seq2) == \
        [('replace', 0, 3, ['X','Y','Z']), ('delete', 3), ('delete', 4), ('delete', 5)]
    

    # 13. Inner Delete + Trailing Insert
    print("13.")
    assert find_edits("EQF", "EF") == \
        [('delete', 1)]
    assert find_edits("EQF", "EFH") == \
        [('delete', 1), ('insert', 3, ['H'])]
    assert find_edits("EQF", "EFHZ") == \
        [('delete', 1), ('insert', 3, ['H','Z'])]


def _seq_to_initial_grammar(seq):
    g = Grammar()
    g.add_production("S", list(seq))
    rhs = list(g.prods.values())[0][0]
    return g, rhs

#print\(\[.*\], '\\n'\)

def test_find_rhs_change():
    g, rhs = _seq_to_initial_grammar('abcdef')

    # 0. Aligned
    print("0.")
    seq2 = 'abcdef'
    find_rhs_change(rhs, seq2)
    print()

    # 1. Unorder
    print("1.1")
    seq2 = 'cbafed'
    find_rhs_change(rhs, seq2)
    print("1.2")
    seq2 = 'fedcba'
    find_rhs_change(rhs, seq2)

    # 2. Unorder + Aligned
    print("2.")
    seq2 = 'acbdfe'
    find_rhs_change(rhs, seq2)

    # # 3. Delete
    print("3.")
    seq2 = 'acdf'
    find_rhs_change(rhs, seq2)

    # # 4. Unorder + Delete
    print("4.")
    seq2 = 'cbed'
    find_rhs_change(rhs, seq2)


    # # 5. Unorder Subsume Delete
    print("5.")
    seq2 = 'cafd'
    find_rhs_change(rhs, seq2)

    # # 6. Insert
    print("6.")
    seq2 = 'xabcdefy'
    find_rhs_change(rhs, seq2)

    # # 7. Unorder + Insert
    print("7.")
    seq2 = 'xcbafedy'
    find_rhs_change(rhs, seq2)

    # # 8. Unorder Subsume Insert
    print("8.")
    seq2 = 'cxbafeyd'
    find_rhs_change(rhs, seq2)

    # # 9. Replace
    print("9.1")
    seq2 = 'xyzdef'
    find_rhs_change(rhs, seq2)

    print("9.2")
    seq2 = 'abxyef'
    find_rhs_change(rhs, seq2)

    print("9.3")
    seq2 = 'abcxyz'
    find_rhs_change(rhs, seq2)

    # # 10. Unorder + Replace
    print("10.")
    seq2 = 'baxyfe'
    find_rhs_change(rhs, seq2)

    # # 11. Unorder + Replace
    print("11.1")
    seq2 = 'cxafyd'
    find_rhs_change(rhs, seq2)


    print("12.1")
    seq2 = 'cxzafyqd'
    find_rhs_change(rhs, seq2)

    print("12.2")
    seq2 = 'cxyzqrv'
    find_rhs_change(rhs, seq2)

    print("12.3")
    seq2 = 'xyz'
    find_rhs_change(rhs, seq2)





# def best_alignment(rhss, seq):
#     for rhs in rhss:
#         l = len(rhs.items)
#         items = [x.name for x in rhs.items]
#         item_set = set(items)

#         best_gc = None
#         best_gc_cost = float('inf')

#         # Convolve application of 
#         for i in range(len(seq)):
#             for j in range(i+l, len(seq)+1):
#                 subseq = seq[i:len(seq)]
#                 intr = set(subseq).intersection(item_set)

#                 # Skip if there is no overlap  
#                 if(len(intr) == 0):
#                     continue

#                 # edits, distances = find_edits(rhs, subseq)
#                 # gc = RHSChange(rhs, (i,j), edits)
#                 change = find_rhs_change(rhs, subseq)
#                 if(gc.cost < best_gc_cost):
#                     print(rhs.items, subseq)
#                     print(edits)
#                     print(distances)
#                     print(gc)
#                     best_gc_cost = gc.cost
#                     best_gc = gc
#         candidate_changes.append(best_gc)




# def get_preterminal_RHSs(grammar):
#     # Find the pre-terminal RHSs 
    


# class ModParseTree:
#     def __init__(self, rhs_change, span, children=[]):
#         self.rhs_change = rhs_change
#         self.symbol = rhs_change
#         self.span = span
#         self.children = children

def calc_overlap(pattern, sequence):
    l0, l1 = len(pattern), len(sequence)
    scores = np.empty((l0, l1), dtype=np.float64)
    for i, s0 in enumerate(pattern):
        for j, s1 in enumerate(sequence):
            if(hasattr(s0,'overlap')):
                scores[i][j] = s0.overlap(s1)
            else:
                scores[i][j] = s0 == s1
    return np.sum(np.max(scores, axis=0))/(l0)#+abs(l0-l1))


def get_best_changes(grammar, seq, rhss):
    best_changes = []
    for rhs in rhss:
        l = len(rhs.items)
        items = [x for x in rhs.items]
        # item_set = set(items)

        best_rc = None
        best_rc_cost_tup = (float('inf'), 0)

        # print(rhs.symbol,"->", rhs)

        best_mp_tree = None

        # Convolved RHS application 
        for i in range(len(seq)):
            for j in range(i+1, min(len(seq), i+l+1)+1):
            # for j in range(i+1, min(len(seq), i+l)+1):
            # for j in range(i+1, len(seq)+1):
                subseq = seq[i:j]
                overlap = calc_overlap(items, subseq)
                # intr = set(subseq).intersection(item_set)

                
                
                # Skip if there is no overlap  
                if(overlap == 0):
                    continue

                rc = find_rhs_change(rhs, subseq, span=(i,j), seq=seq)
                # print(i,j, subseq)
                print(f"{rhs.symbol}->{rhs} : c:{rc.cost:.2f} o:{overlap:.2f} r:{rc.cost/overlap:.2f}", subseq, rc.edits)
                cost_tup = (rc.cost/overlap, -overlap)


                # print(f"{rc.cost/overlap:.2f}", subseq, rc.edits)
                if(cost_tup < best_rc_cost_tup):
                    best_rc_cost_tup = cost_tup
                    best_rc = rc

                # print(rc.cost)
        
        if(best_rc is None):
            # raise ValueError("NO MATCH OCCURED")
            cost = len(rhs.items)
            best_rc = RHSChange(rhs, (0, len(rhs.items)), None, nomatch=True, seq=seq)
        best_changes.append(best_rc)
        s0, s1 = best_rc.span
        # print(rhs.symbol,"->", rhs, ":", seq[s0:s1], f"{best_rc.cost:.2f}", best_rc.edits)


        # mp_trees.append(best_mp_tree)
        # print("---")
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
            new_edits.append(('insert', head, seq[head:s0]))
            # for j in range(head,s0):
                # if(j < len(seq_trees)):
                #     print("SQT:", seq_trees[j])
                # if(not upstream_covered[j]):
                
            new_edits += rc.edits
            # ins_cost = cost_of_edits(rc.rhs, new_edits, e0-s0)
            # Prepend any uncovered symbols to beginning
            rc = RHSChange(rc.rhs, (head,e0), new_edits, seq=seq)
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
            new_edits.append(('insert', e0, seq[e0:s1]))
            # for j in range(e0,s1):
                # if(j < len(seq_trees)):
                #     print("SQT:", seq_trees[j])
                # if(not upstream_covered[j]):
                
                
            # print(new_edits)
            # ins_cost = cost_of_edits(rc.rhs, new_edits, s1-head)
            new_rcs[-1] = RHSChange(rc.rhs, (head,s1), new_edits, seq=seq)
            
        head = max(s1,e0)
    # print(">>>", [(rc.span, rc.edits) for rc in new_rcs], len(seq))
    return new_rcs

class ParseTree:
    def __init__(self, rhs_change, child_trees=[]):
        # print(rhs_change.rhs, rhs_change.span)
        # print("CHILD TREES", child_trees)
        self.rhs = rhs_change.rhs

        # rhs_change, child_trees = self._mergeInsDel(rhs_change, child_trees)
        self.change = rhs_change
        self.symbol = self.rhs.symbol
        self.child_trees = child_trees
        self.recalc_cost()        


    def recalc_cost(self):
        self.cost = self.change.cost + sum([rc.cost for rc in self.child_trees if rc])

    def __str__(self):
        s = f"{self.cost:.2f} {self.symbol}->{self.rhs} {self.change.edits}"
        for ct in self.child_trees:
            if(isinstance(ct, ParseHole)):
                continue
            if(ct):
                s += "\n" + str(ct)
        return s

    def __repr__(self):
        return f"PT[{self.symbol}]"


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
    # def __copy__(self):


class ParseHole(ParseTree):
    def __init__(self, symb, child_tree, prev_pt=None, next_pt=None):
        self.symbol = symb
        self.child_trees = [child_tree]
        self.prev_pt = prev_pt
        self.next_pt = next_pt
        self.cost = child_tree.cost if child_tree else 0.0

    def __repr__(self):
        return f"H[{self.symbol}]"

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
        return ParseHole(self.symbol, child_copy,
             prev_pt=self.prev_pt, next_pt=self.next_pt)





    # def __str__(self):

        





# Adopted from https://stackoverflow.com/questions/54476451/how-to-get-all-maximal-non-overlapping-sets-of-spans-from-a-list-of-spans
def get_spanning_trees(parse_trees, seq):
    parse_trees = sorted(parse_trees,key=lambda x: x.change.span[0])
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
    print(">>", [(f"{sum([pt.cost for pt in path]):.2f}", [(pt.symbol, *pt.change.span) for pt in path]) for path in paths])
    return paths

def filter_spanning_trees(paths):
    min_path = None
    min_cost = float('inf')
    for path in paths:
        cost = sum([pt.cost for pt in path])
        if(cost < min_cost):
            min_path = path
            min_cost = cost
    return [min_path]

        # print(":", rc.rhs.symbol, rc.span)

def _sequence_upstreams(grammar, rhss):
    if(len(rhss) == 1):
        return [rhss]

    def n_upstreams(rhs):
        uptrms = grammar.get_upstream_RHSs(rhs)
        inrs = sum([x in uptrms for x in rhss])
        return inrs

    tups = [(n_upstreams(rhs), (rhs, cov)) for rhs,cov in rhss.items()]
    tups = sorted(tups, key=lambda x: x[0], reverse=True)

    curr_ups, (frst_rhs, frst_cov) = tups.pop(0)
    out = [{frst_rhs: frst_cov}]
    for n_ups, (rhs,cov) in tups:
        if(n_ups == curr_ups):
            out[-1][rhs] = cov
        else:
            out.append({rhs: cov})
            curr_ups = n_ups  
    # print("++", out)
    return out          



def get_upstream_RHSs(grammar, symbols):
    upstream_rhss = {}        
    for _, rhss in grammar.prods.items():
        for rhs in rhss:
            for item in rhs.items:
                # print(item, symbols)
                if(item in symbols):
                    lst = upstream_rhss.get(rhs,[])
                    lst.append(item)
                    upstream_rhss[rhs] = lst
    
    return upstream_rhss


class DeferredRHSChange(RHSChange):
    ''' '''
    def __init__(self, rhs, span):
        self.rhs = rhs
        self.span = span


def _make_seq_and_holes(tree_seq, seq, seq_trees):
    new_seq = []
    new_trees = []
    head = 0
    prev_pt = None
    for i, pt in enumerate(tree_seq):
        # print(pt)
        rc = pt.change
        # print("@@", rc.span)

        # Fill in any symbols preceeding the previous non-terminal
        if(head < rc.span[0]):
            # print("B: NEVER HAPPEN W FILLED SPAN")
            for j in range(head, rc.span[0]):
                new_seq.append(seq[j])
                if(seq_trees and isinstance(seq_trees[j], ParseHole)):
                    ph = seq_trees[j]
                else:
                    t = seq_trees[j] if seq_trees else None
                    ph = ParseHole(seq[j], t, prev_pt=prev_pt, next_pt=pt)
                new_trees.append(ph)

        # # Fill in the next non-terminal
        new_seq.append(pt.symbol)
        new_trees.append(pt)
        head = rc.span[1]
        prev_pt = pt

    # Fill in any trailing symbols
    if(head < len(seq)):
        # print("E: NEVER HAPPEN W FILLED SPAN", head, len(seq))
        for j in range(head, len(seq)):
            new_seq.append(seq[j])
            if(seq_trees and isinstance(seq_trees[j], ParseHole)):
                ph = seq_trees[j]
            else:
                t = seq_trees[j] if seq_trees else None
                ph = ParseHole(seq[j], t, prev_pt=prev_pt)
            new_trees.append(ph)
    # new_seq = [rc.rhs.symbol for rc in change_seq if not rc.nomatch]
    # print("NEW SEQ", new_seq)
    assert new_seq != seq, f"Recursion made no parsing progress: {seq}, then {new_seq}."
    # print("!!NS", new_seq)
    return new_trees, new_seq 

def _fill_holes_downstream(pt):
    rc = pt.change
    child_trees = pt.child_trees
    holes_spans = {}
    # modified_edits = [*rc.edits]

    # Check the highest edits in the parse tree and see if they
    #  overlap with an unparsed a ParseHole symbol. 
    for edit_ind, edit in reversed([*enumerate(rc.edits)]):
        if(edit[0] in ('insert', 'replace')):
            s = edit[1]+rc.span[0]
            # print("S", s, child_trees[s] if child_trees and s < len(child_trees) else None)
            if(child_trees and s < len(child_trees) and isinstance(child_trees[s], ParseHole)):

                # If a symbol precedes the hole use that
                #  otherwise insert the holes into the next symbol
                hp = child_trees[s]
                adj, is_prev = (hp.prev_pt, True) if hp.prev_pt else (hp.next_pt, False)
                _rc = adj.change

                # Skip inserting into unordered RHSs since
                if(_rc.rhs.unordered):
                    continue

                # print("  ###:", rc, f"({rc.seq[rc.span[0]:rc.span[1]]})")
                # print(" :-", hp, "|", hp.prev_pt, "|", hp.next_pt)

                _, p_lst, n_lst = holes_spans.get(adj.change.rhs, (None, [],[]))
                lst = n_lst if is_prev else p_lst
                lst.append(hp)
                holes_spans[adj.change.rhs] = (adj, p_lst, n_lst)
                del rc.edits[edit_ind]
    
    if(len(holes_spans) == 0):
        return pt

    # from copy import deepcopy
    # pt = deepcopy(pt)
    
    # print("# HP:", len(holes_spans))
    replacements = []
    for rhs, (adj, p_lst, n_lst) in holes_spans.items():
        _rc = adj.change

        # print(_rc.span)
        prev, nxt = [], []
        if(len(p_lst) > 0):
            prev = [('insert', 0, [hp.symbol for hp in p_lst])]
        if(len(n_lst) > 0):
            nxt = [('insert', _rc.span[1]-_rc.span[0], [hp.symbol for hp in n_lst])]
        edits = [*prev, *_rc.edits, *nxt]
        
        s,e = _rc.span[0]-len(p_lst), _rc.span[1]+len(n_lst)
        # cost = cost_of_edits(rhs, edits, e-s)
        rc = RHSChange(rhs, (s,e), edits, seq=_rc.seq)
        _pt = ParseTree(rc, child_trees=adj.child_trees)
        # print(" ->", rc.cost, rc)
        # print(_pt)
        # print()
        replacements.append((adj, _pt))

    pt = pt._deep_copy(replacements)
    pt.change.recalc_cost()
    pt.recalc_cost()
    return pt



def _bottom_up_recurse(grammar, seq, rhss=None, seq_trees=None):
    print("\nSEQ", seq, len(seq), len(seq_trees) if seq_trees else None)
    out = []

    # Find RHSs immediately upstream from the symbols in seq
    if(rhss is None):
        rhss = get_upstream_RHSs(grammar, set(seq))

    # Check terminating case when seq is the start symbol
    if(len(rhss) == 0):
        return seq_trees

    # Try to parse using each rhs
    best_changes = get_best_changes(grammar, seq, rhss)

    # Turn each partial parse into a parse tree 
    parse_trees = []
    for rc in best_changes:
        child_trees = [seq_trees[j] if seq_trees else None for j in range(*rc.span)]
        pt = ParseTree(rc, child_trees)
        #  If an insertion is required use ParseHoles to move it
        #   to the lowest possible section of the parse. 
        pt = _fill_holes_downstream(pt)
        parse_trees.append(pt)    

    # for k, tree_seq in enumerate(get_spanning_trees(parse_trees, seq)):
    #     for i, pt in enumerate(tree_seq):
    #         rc = pt.change
    #         print((k,i), ":::", rc.rhs.symbol, rc.span, f"{rc.cost:0.2f}", rc.edits)
    sp_trees = get_spanning_trees(parse_trees, seq)
    # sp_trees = filter_spanning_trees(sp_trees)
    for k, tree_seq in enumerate(sp_trees):
        # print("----<", seq)
        # change_seq = fill_span_holes(change_seq, seq)
        # for i, pt in enumerate(tree_seq):
        #     rc = pt.change
        #     print( (k, i), ":", rc.rhs.symbol, rc.span, f"{rc.cost:0.2f}", rc.edits)

        # Make the new sequence by replacing each RHS parse
        #  with its upstream symbol. Keep track of any ParseHoles. 
        new_trees, new_seq = _make_seq_and_holes(tree_seq, seq, seq_trees)
        out += _bottom_up_recurse(grammar, new_seq, seq_trees=new_trees)


    return out

def parse_w_changes(grammar, seq):
    # print(seq)
    prods = grammar.prods
    # print(prods)

    pre_terminals = grammar.preterminal_RHSs
    # print()
    # print("PRETERMINALS", [rhs.symbol for rhs in pre_terminals])
    # print()
    # all_changes = []
    parse_trees = _bottom_up_recurse(grammar, seq, pre_terminals)

    print("\n  parse_trees:")
    for pt in parse_trees:
        print(f": {pt.cost:.2f}", )
        print(pt)

    parse_tree = sorted(parse_trees, key=lambda x:x.cost)[0]

    # raise ValueError()

    # print("PT", parse_tree.rhs, parse_tree.cost, len(parse_tree.child_trees))
    # print(all_changes
    return parse_tree
    


def test_rhs_accept_cost():

    # -- Test unordered -- # 
    rhs = RHS(["a","b"], unordered=True)
    rc = find_rhs_change(rhs, "ba")
    print(rc.cost, rc.edits)
    assert rc.cost == 0

    rhs = RHS(["a","b","c"], unordered=True)
    rc = find_rhs_change(rhs, "bca")
    print(rc.cost, rc.edits)
    assert rc.cost == 0

    # -- Test Optional -- # 
    rhs = RHS(["a","b","c"],  optionals=[1])
    rc = find_rhs_change(rhs, "ac")
    print(rc.cost, rc.edits)
    assert rc.cost == 0

    # -- Test Mixed -- # 
    rhs = RHS(["a","b","c"], unordered=True, optionals=[1])
    rc = find_rhs_change(rhs, "ca")
    print(rc.cost, rc.edits)
    assert rc.cost == 0

    rhs = RHS(["a","b","c","e","f"], unordered=True, optionals=[1,3,4])
    rc = find_rhs_change(rhs, "ca")
    print(rc.cost, rc.edits)
    assert rc.cost == 0

    # -- Test Part Edit + Part Accept -- # 
    rhs = RHS(["a","b","c","e","f"], unordered=True, optionals=[1,3])
    rc = find_rhs_change(rhs, "eca")
    print(rc.cost, rc.edits)
    assert rc.cost > 0
    assert rc.edits[0][0] == "delete"

    rhs = RHS(["a","b","c","e","f"], optionals=[1,3])
    rc = find_rhs_change(rhs, "fca")
    print(rc.cost, rc.edits)
    assert rc.cost > 0
    assert rc.edits[0][0] == "unorder"


def parse_only_deletes(pt):
    rc = pt.change
    edits = rc.edits
    for edit in edits:
        kind = edit[0]
        if(kind != "delete"):
            return False

    # Check that the deletes are contiguous
    #  and terminate at end of rhs.
    if(len(edits) > 0 and not rc.rhs.unordered):
        d0 = edits[0][1] #First Delete
        k = 1
        # print("d0", d0)
        for i in range(d0+1, len(rc.rhs.items)):
            # print(":", k, len(edits), i, len(rc.rhs.items))
            if(k >= len(edits)):
                return False
            if(rc.rhs.is_optional(i)):
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

def check_accepts_subseq(g, seq, item):
    # pt = parse_w_changes(g, seq + item)
    # return is_subseq_parse(pt)
    s0, s1 = seq, seq+item
    pt0 = parse_w_changes(g, s0)
    pt1 = parse_w_changes(g, s1)
    only_del = parse_only_deletes(pt1)
    print(">>", s0, s1, pt0.cost, pt1.cost, pt1.cost <= pt0.cost, only_del)
    return only_del#pt1.cost <= pt0.cost and only_del


def _test_subseqs_acceptance(g, seq):
    for i in range(2, len(seq)):
        print()
        s0, item = seq[:i-1], seq[:i][-1]
        # pt0 = parse_w_changes(g, s0)
        # pt1 = parse_w_changes(g, s1)

        # check_accepts_subseq(g, s0, seq[i])
        # assert is_subseq_parse(pt1)
        # print(">>", s0, s1, pt0.cost, pt1.cost, pt1.cost <= pt0.cost)
        assert check_accepts_subseq(g, s0, item)

        # ptx = parse_w_changes(g, s0+"x")
        assert not check_accepts_subseq(g, s0, 'x')
        # print()
        # ptx = parse_w_changes(g, s0+"x")
        # print(">>", s0, s0+"x", pt0.cost, ptx.cost, ptx.cost > pt0.cost)
        # assert ptx.cost > pt0.cost


def test_accept_subseq():

    # -- Test Basic -- #
    g = Grammar()
    g.add_production("S", RHS(["A","B","C"]))
    g.add_production("A", RHS(["a","b"]))
    g.add_production("B", RHS(["d","c"], unordered=True))
    g.add_production("C", RHS(["e", "q", "f"], optionals=[1]))

    _test_subseqs_acceptance(g, "abcdef")
    _test_subseqs_acceptance(g, "abdceqf")

    # Check cannot skip
    assert not check_accepts_subseq(g, "ab", "e")
    assert not check_accepts_subseq(g, "ab", "q")
    assert not check_accepts_subseq(g, "ab", "f")
    
    g = Grammar()
    g.add_production("S", RHS(["A","B","C"], optionals=[1]))
    g.add_production("A", RHS(["a","b"]))
    g.add_production("B", RHS(["d","c"], unordered=True))
    g.add_production("C", RHS(["e", "q", "f"], optionals=[1]))

    _test_subseqs_acceptance(g, "abcdef")

    # Check can skip optional non-terminals
    assert check_accepts_subseq(g, "ab", "e")
    assert check_accepts_subseq(g, "abe", "q")
    assert check_accepts_subseq(g, "abe", "f")
    assert not check_accepts_subseq(g, "ab", "q")
    assert not check_accepts_subseq(g, "ab", "f")

    # -- Test Disjuncts -- #
    g = Grammar()
    g.add_production("S", RHS(["A","B","C"]))
    g.add_production("A", RHS(["a0","b0"]))
    g.add_production("A", RHS(["a1","b1"]))
    g.add_production("B", RHS(["d","c"], unordered=True))
    g.add_production("C", RHS(["e", "q", "f"], optionals=[1]))

    # Check not accept repeats on disjunction
    assert not check_accepts_subseq(g, ["a0", "b0"], ["a1"])
    assert not check_accepts_subseq(g, ["a0", "b0"], ["b1"])

# def apply(self, grammar):
#         self.ensure_changes_computed()

#         old_rhs = self.rhs
#         new_rhs = apply_rhs_edits(self.rhs, self.parent_edits)
        
#         prod = grammar.prods[old_rhs.symbol]
#         rhs_ind = prod.index(old_rhs)
#         prod[rhs_ind] = new_rhs
        
#         for (sym, rhss, (s,e)) in [*self.unorder_substs,*self.disjoin_substs]:
#             for rhs in rhss:
#                 grammar.add_production(sym, rhs)
#         return grammar, new_rhs


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

    s_ind = 0
    symbol_factory = SymbolFactory(grammar=g)#lambda : _new_symbol_name(g, s_ind)

    # Collect Changes
    changes = []
    pts = [parse_tree]
    depth = 0
    while(len(pts) > 0):
        new_pts = []
        for i, pt in enumerate(pts):
            if(not pt or isinstance(pt, ParseHole)):
                continue
            print("CHANGE", depth, i, pt.change.cost, pt.change.rhs.symbol, "->", pt.change.rhs, pt.change.edits, )
            # changes[pt.change.rhs] = pt.change
            changes.append(pt.change)
            new_pts += pt.child_trees
        depth += 1
        pts = new_pts


    # new_prods = {}
    # for symb, rhss in g.prods:
    #     for rhs in rhss:
    #         rc = changes[rhs]

    new_g = g.__copy__()
    for rc in changes:
        # print(rc.rhs.symbol,  rc.edits)
        rc.apply(new_g, symbol_factory)
    return new_g

    
def generalize_from_seq(g, seq):
    print("----", seq, "-----")
    pt = parse_w_changes(g, seq)
    g = apply_grammar_changes(g, pt)
    return g

def make_mc_skill_apps():
    '''c3 c2 c1
          t2 t1 t0
     +    b2 b1 b0
     -------------
       o3 o2 o1 o0
    '''
    a20 = SkillApp('a2', 'o0', ['t0','b0'], short_name='a20')
    c20 = SkillApp('c2', 'c1', ['t0','b0'], short_name='c20')
    a21 = SkillApp('a2', 'o1', ['t1','b1'], short_name='a21')
    c21 = SkillApp('c2', 'c2', ['t1','b1'], short_name='c21')
    a22 = SkillApp('c2', 'o2', ['t2','b2'], short_name='a22')
    c22 = SkillApp('c2', 'c3', ['t2','b2'], short_name='c22')
    cp3 = SkillApp('cp', 'o3', ['c3'], short_name='cp3')


    a31 = SkillApp('a3', 'o1', ['c1', 't1', 'b1'], short_name='a31')
    c31 = SkillApp('c3', 'c2', ['c1', 't1', 'b1'], short_name='c31')
    a32 = SkillApp('c3', 'o2', ['c2', 't2', 'b2'], short_name='a32')
    c32 = SkillApp('c3', 'c3', ['c2', 't2', 'b2'], short_name='c32')

    d = SkillApp('prs', 'd', [], short_name='d')

    return (a20, c20, a21, c21, a22, c22, cp3, a31, c31, a32, c32, d)

def _temp_test():
    (a20, c20, a21, c21, a22, c22, cp3, a31, c31, a32, c32, d) = make_mc_skill_apps()
    # S -> A B C
    # A -> a20 c20
    # B -> D E | c21 a21 c22 a22
    # C -> cp3 d
    # D -> c31 a31
    # E -> c32 a32

    g = Grammar()
    g.add_production("S", RHS(["A","B","C"]))
    g.add_production("A", RHS([a20,c20], unordered=True))
    g.add_production("B", RHS(["D", "E"]))
    g.add_production("B", RHS([c21, a21, c22, a22]))
    g.add_production("C", RHS([cp3, d]))
    g.add_production("D", RHS([c31,a31], unordered=True))
    g.add_production("E", RHS([c32,a32], unordered=True))
    print(g)    

    # print("vvvvvvvvvvvvvvvvv")
    # for sym, rhss in g.prods.items():
    #     print(sym, g.get_upstream_RHSs(sym))
    #     for rhs in rhss:
    #         print(" ", rhs, g.get_upstream_RHSs(rhs))

    # raise NotImplemented()
    g = generalize_from_seq(g, [a20, c20, a21, c21, a22, c22, cp3, d])    




def test_bottom_up_changes():
    (a20, c20, a21, c21, a22, c22, cp3, a31, c31, a32, c32, d) = make_mc_skill_apps()

    seqs = [
        [d, a20, c20, a21, c21, a22, c22, cp3, d],
        [d, c20, a20, c21, a21, c22, a22, cp3, d],
        [d, a20, c20, c31, a31, c32, a32, cp3, d],
        [d, a20, c20, c31, a31, c32, a32, d],
        [d, a20, c20, a31, c31, a32, c32, cp3, d],
        [d, a20, c20, a21, c21, a32, c32, d],
    ]
    from random import shuffle
    order = list(range(len(seqs)))
    # shuffle(order)
    order = [3, 2, 0, 1, 4, 5]
    # order =  [3, 1, 4, 0, 2]
    # order = [2, 1, 0, 4, 3, 5]
    # order = [3, 4, 0, 2, 1]
    # order = [2, 3, 1, 4, 0]
    # order = [2, 3, 1, 4, 0]
    # order = [3, 0, 2, 1, 4]
    # order = [3, 1, 0, 4, 2]
    # order = [2, 0, 4, 3, 1]
    # order = [3, 1, 0, 4]#, 2]
    print("ORDER", order)

    try: 
        g, rhs = _seq_to_initial_grammar(seqs[order[0]])
        print(f'\ngrammar:\n{g}')
        for i in range(1,len(order)):
            g = generalize_from_seq(g, seqs[order[i]])    
            print(f'\ngrammar:\n{g}')
    except Exception as e:
        print(g, i)
        print("ERROR ON ORDER", order)
        raise e

    # g, rhs = _seq_to_initial_grammar([a20, c20, a21, c21, a22, c22, cp3, d])
    # print(f'\ngrammar:\n{g}')

    # g = generalize_from_seq(g, [a20, c20, a21, c21, a22, c22, cp3, d])    
    # print(f'\ngrammar:\n{g}')

    # g = generalize_from_seq(g, [c20, a20, c21, a21, c22, a22, cp3, d])
    # print(f'\ngrammar:\n{g}')
    # g = generalize_from_seq(g, [c20, a20, c21, a21, c22, a22, cp3, d])
    # print(f'\ngrammar:\n{g}')

    # g = generalize_from_seq(g, [a20, c20, c31, a31, c32, a32, cp3, d])
    # print(f'\ngrammar:\n{g}')

    # g = generalize_from_seq(g, [a20, c20, c31, a31, c32, a32, d])
    # print(f'\ngrammar:\n{g}')

    # g = generalize_from_seq(g, [a20, c20, a31, c31, a32, c32, cp3, d])
    # print(f'\ngrammar:\n{g}')

    # g = generalize_from_seq(g, ['c0', 'a0','c1','a1', 'c2', 'a2', 'a3', 'd'])

    
    # g = Grammar()
    # g.add_production("S", RHS(["A","B","C"]))
    # g.add_production("A", RHS(["a","b"]))
    # g.add_production("B", RHS(["d","c"], unordered=True))
    # g.add_production("C", RHS(["e", "q", "f"], optionals=[1]))

    

        
        # print()

    # print("---------------------")


    # g.add_production("S", RHS(["A","B","C"]))
    # g.add_production("A", RHS(["a","b"]))
    # g.add_production("B", RHS(["d","c"], unordered=True))
    # g.add_production("C", RHS(["e", "q", "f"], optionals=[1]))
    # parse_w_changes(g, 'abcdef')

    # -1: Grammar Sequence

    # g, rhs = _seq_to_initial_grammar('abcdef')


    # 0. Aligned
    # print("0.")
    # seq2 = 'abcdef'
    # parse_w_changes(g, seq2)
    # print()

    # 1. Unorder
    # print("1.1")
    # seq2 = 'cbafed'
    # parse_w_changes(g, seq2)
    # print("1.2")
    # seq2 = 'fedcba'
    # parse_w_changes(g, seq2)

    # # 2. Unorder + Aligned
    # print("2.")
    # seq2 = 'acbdfe'
    # parse_w_changes(g, seq2)

    # # # 3. Delete
    # print("3.")
    # seq2 = 'acdf'
    # parse_w_changes(g, seq2)

    # # # 4. Unorder + Delete
    # print("4.")
    # seq2 = 'cbed'
    # parse_w_changes(g, seq2)


    # # # 5. Unorder Subsume Delete
    # print("5.")
    # seq2 = 'cafd'
    # parse_w_changes(g, seq2)

    # # # 6. Insert
    # print("6.")
    # seq2 = 'xabcdefy'
    # parse_w_changes(g, seq2)

    # # # 7. Unorder + Insert
    # print("7.")
    # seq2 = 'xcbafedy'
    # parse_w_changes(g, seq2)

    # # # 8. Unorder Subsume Insert
    # print("8.")
    # seq2 = 'cxbafeyd'
    # parse_w_changes(g, seq2)

    # # # 9. Replace
    # print("9.1")
    # seq2 = 'xyzdef'
    # parse_w_changes(g, seq2)

    # print("9.2")
    # seq2 = 'abxyef'
    # parse_w_changes(g, seq2)

    # print("9.3")
    # seq2 = 'abcxyz'
    # parse_w_changes(g, seq2)

    # # # 10. Unorder + Replace
    # print("10.")
    # seq2 = 'baxyfe'
    # parse_w_changes(g, seq2)

    # # # 11. Unorder + Replace
    # print("11.1")
    # seq2 = 'cxafyd'
    # parse_w_changes(g, seq2)


    # print("12.1")
    # seq2 = 'cxzafyqd'
    # parse_w_changes(g, seq2)

    # print("12.2")
    # seq2 = 'cxyzqrv'
    # parse_w_changes(g, seq2)

    # print("12.3")
    # seq2 = 'xyz'
    # parse_w_changes(g, seq2)


def test_target_domain_changes():

    ## Multi Column Addition
    # 3x3 : First: add, carry then: carry, add
    print("MC.1")
    g, rhs = _seq_to_initial_grammar(['a0','c0','a1', 'c1', 'a2', 'c2', 'a3', 'd'])
    find_rhs_change(rhs, ['c0', 'a0','c1','a1', 'c2', 'a2', 'a3', 'd'])

    print("MC.2")
    # 3x3 : First: add, w/o carry (i.e. like 333+333) then: carry, add
    g, rhs = _seq_to_initial_grammar(['a0','a1','a2', 'd'])
    new_rhs = find_rhs_change(rhs, ['c0', 'a0','c1','a1', 'c2', 'a2', 'a3', 'd'])
    find_rhs_change(new_rhs, ['a0','c0','a1', 'c1', 'a2', 'c2', 'a3', 'd'])
    

def test_blarg():
    # find_edits()

    print("0.")
    seq1 = 'ABCDEF'
    edits = find_edits("ABC", "AA")


# class SkillAppGen(object):
#     def __init__(self):
#         self.skill_apps = []

class SkillApp(object):
    def __init__(self, skill, sel, args, short_name=None, state=None):
        self.skill = skill
        self.sel = sel
        self.args = args
        self.state = state
        self.short_name = short_name

    def arg_overlap(self, other):
        s_a, s_b = set(self.args), set(other.args)
        denom = max(len(s_a),len(s_b))
        if(denom == 0):
            return 0.0
        return len(s_a.intersection(s_b)) / denom

    def sel_overlap(self, other):
        return float(self.sel == other.sel)

    def skill_overlap(self, other):
        return float(self.skill == other.skill)

    # In principle this could be replaced with
    #  structure mapping score of where-part
    def overlap(self, other):
        if(not isinstance(other, SkillApp)):
            return 0.0
        return (self.skill_overlap(other) +
                self.sel_overlap(other) +
                self.arg_overlap(other)) / 3

    def depends_on(self, other):
        return other.sel in self.args

    def __eq__(self, other):
        if(not isinstance(other, SkillApp)):
            return False
        return (self.skill == other.skill and 
                self.sel == other.sel and
                self.args == other.args)

    def __hash__(self):
        return hash((self.skill, self.sel, tuple(self.args)))

    def __str__(self):
        if(self.short_name):
            return self.short_name
        else:
            return f"{self.skill}({','.join(self.args)})->{self.sel}"

    __repr__ = __str__

    @property
    def name(self):
        return str(self)
    


def test_skill_app():
    print("Same", a20.overlap(a20))
    print("Same args", a20.overlap(c20))
    print("Different", a21.overlap(c20))
    print("Same match", a21.overlap(a31))

def _is_term(sym, g):
    return len(g.prods.get(sym,[])) == 0

def top_down_parse(g, sym, seq):
    parse_trees = []
    for rhs in g.prods[sym]:
        # print(rhs)
        return _top_down_parse(g, rhs, seq)

import itertools        
def _top_down_parse(g, rhs, seq):
    parse_trees = []
    # for rhs in g.prods[sym]:
    s, e = 0, len(seq)
    # print("SEQ", seq)
    # print(rhs.symbol, "->", rhs)
    
    # If all symbols are terminal just run get_best_changes().
    #  This could produce a full match or an edit.
    are_term = [_is_term(x,g) for x in rhs.items]
    if(all(are_term)):
        rc = get_best_changes(g, seq, [rhs])[0]
        print("::", rc.rhs.symbol, rc.span, rc.cost)
        return rc.span, [ParseTree(rc)]
    
    first_nt = are_term.index(False)   
    last_nt = len(are_term) - 1 - are_term[::-1].index(False)   


    for i in range(0, first_nt):
        if(seq[i] != rhs.items[i]):
            break
        s = i+1

    for i in range(0, last_nt):
        j = -i-1
        if(seq[j] != rhs.items[j]):
            break
        e = len(seq)+j


    print("NT", s, e, first_nt, last_nt)
    
    # If not all terminal see how far can get left and right
    if(not rhs.unordered):
        # Otherwise return
        for i in range(first_nt, last_nt+1):
            # print(rhs.items, i)
            item = rhs.items[i]
            # print(rhs.symbol, ">>", item, s)
            print("RECURSE", item, seq[s:e])
            child_trees = []
            for _rhs in g.prods[item]:
                (s0, e0), pts = _top_down_parse(g, _rhs, seq[s:e])
                print("s0", "e0", s0, e0, (s,e))
                # if(s0 == 0):
                s += e0
                child_trees.append(pts)
                # else:
                #     raise ValueError("Subproblem")
            # print(child_trees)
            for cts in itertools.product(*child_trees):
                # print(cts)
                pt = ParseTree(RHSChange(rhs, (0, len(seq))), child_trees=cts, seq=seq)
                parse_trees.append(pt)


            # If terminal increment s
            # if(_is_term(item, g)):
            #     print(item, s, seq)
            #     print(item, seq[s], seq[s] == item)
            #     if(seq[s] == item):
            #         s += 1
            #     else:
            #         print("BEF RET")
            #         return (s, e), []
            # # If non-terminal recurse
            # else:
            

    else:
        pass


    


    return (s, e), parse_trees

        # print(s)






def test_top_down_parse():
    (a20, c20, a21, c21, a22, c22, cp3, a31, c31, a32, c32, d) = make_mc_skill_apps()

    # grammar:
    # S -> d* A B d*
    # A -> d* a20 c20
    # B -> C D d* cp3* | c21 a21 c22 a22 cp3
    # C -> c31 a31
    # D -> c32 a32


    g = Grammar()
    g.add_production("S", [d, "A", "B", d])
    g.add_production("A", RHS([a20, c20]))
    g.add_production("B", RHS([c31, a31, c32, a32]))
    g.add_production("B", RHS([c21, a21, c22, a22, cp3]))

    # Case 1: Full Match
    # seq = [d, a20, c20, c31, a31, c32, a32, d]
    # (s,e), pts = top_down_parse(g,"S", seq)

    # print()
    # Case 2: Not match
    seq = [d, a20, c20, a31, c31, c32, a32, d]
    (s,e), pts = top_down_parse(g,"S", seq)

    # Case 3: Not match
    g = Grammar()
    g.add_production("S", [d, "A", "B", d])
    g.add_production("A", RHS([a20, c20]))
    g.add_production("B", RHS(["C", "D"]))
    g.add_production("B", RHS([c21, a21, c22, a22, cp3]))
    g.add_production("C", RHS([c31, a31]))
    g.add_production("D", RHS([c32, a32]))

    print("GRAMMAR")
    print(g)


    seq = [d, a20, c20, a31, c31, c32, a32, d]
    (s,e), pts = top_down_parse(g,"S", seq)

    print("SEQ", seq)
    print(">>", (s,e), seq[s:e])
    print("\n  parse_trees:")
    for pt in pts:
        print(":", pt.cost)
        print(pt)
    







if(__name__ == "__main__"):
    # test_skill_app()
    # test_blarg()
    # test_find_edits()
    # test_find_rhs_change()
    # test_rhs_accept_cost()
    # test_accept_subseq()
    test_bottom_up_changes()
    # test_top_down_parse()
    # test_target_domain_changes()

    # _temp_test()



# find_edits(, ['C', 'B' ,'A', 'X', 'F', 'E', 'D', 'Y'])
# find_edits(['A', 'B' ,'C', 'D', 'E', 'F'], ['B', 'A' ,'C', 'X', 'D', 'F', 'E', 'Y'])
# find_edits(['A', 'B' ,'C', 'D', 'E', 'F'], ['B', 'A' ,'C', 'C', 'D', 'F', 'E', 'Y'])
    # First find all unorder matches:
    #  all sub-sequences of seq1 which
    #  occur (perhaps) out-of-order in seq2 
    # for s1 in range(len(seq1)):
    #     for e1 in range(s1, len(seq1)):
    #         for s2 in range(len(seq1)):
    #             for e2 in range(s1, len(seq1)):



# g = Grammar()
# g.add_production("S", [Sym("T"), Sym("Q")])
# g.add_production("T", RHS(["A", "B", "C"], unordered=True))
# g.add_production("Q", RHS(["D", "E", "F"], unordered=True))
# print()
# g.top_down_parse(["C", "B", "A", "F", "E", "D"], start_symbol='S')

# minEdits(A, B)

# minEdits(A, C)
