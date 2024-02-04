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

# class Lit:
#     def __init__(self, )

class RHS:
    def __new__(cls, items, symbol=None, unordered=None, optionals=None):
        if(isinstance(items, RHS)):
            self = items.__copy__()
            if(symbol is not None):
                self.symbol = symbol

            self.unordered = False if not unordered else unordered
            if(optionals is not None):
                self.optional_mask = np.zeros(len(items), dtype=np.bool_)
                self.optional_mask[optionals] = 1
        else:
            # print([type(x) for x in items])
            self = super().__new__(cls)
            
            self.unordered = False if not unordered else unordered
            self.symbol = symbol
            self.optional_mask = np.zeros(len(items), dtype=np.bool_)
            if(optionals is not None):
                self.optional_mask[optionals] = 1

            # Make equality literals
            self.items = []#][Sym(x) if isinstance(x, str) else x for x in items]
            self.rel_lits = []#][Sym(x) if isinstance(x, str) else x for x in items]
            for i, item in enumerate(items):
                if(isinstance(item, (Sym, str))):
                    self.items.append(Sym(item))
                elif(isinstance(item, SkillApp)):
                    # toks = self.eq_toks[i] = []
                    for j in range(i, len(items)):
                        item_j = items[j]
                        if(not isinstance(item_j, SkillApp)):
                            continue
                        for k, arg_k in enumerate(item.args):
                            for u, arg_u in enumerate(item_j.args):
                                rel = arg_k.get_relation(arg_u)
                                # if(arg_k == arg_u and (i != j or u != k)):
                                if(rel is not None and (i != j or u != k)):
                                    # print(item, k, item_j, u, rel)
                                    self.rel_lits.append((rel, i, j, k, u))


                    self.items.append(item.skill)
                elif(isinstance(item, Skill)):
                    self.items.append(item)
            # print(self.items, self.rel_lits)
        return self


    def is_optional(self, ind):
        return self.optional_mask[ind]

    @property
    def is_recursive(self):
        return self.symbol in self.items

    def set_optional(self, ind, is_optional=1):
        self.optional_mask[ind] = is_optional

    def __copy__(self):
        new_rhs = RHS([*self.items], symbol=self.symbol, unordered=self.unordered)
        new_rhs.optional_mask = self.optional_mask.copy()
        new_rhs.rel_lits = copy(self.rel_lits)
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
            
            # print("SLICE", key.__dict__ )
            subitems = self.items[key]
            subopt = self.optional_mask[key]
            new_rhs = RHS(subitems)
            # new_rhs.items = subitems
            new_rhs.optional_mask = subopt
            new_rhs.unordered = self.unordered
            new_rhs.symbol = self.symbol

            s,e = key.start, key.stop
            s = 0 if s is None else s
            e = len(self.items)-1 if e is None else e
            new_rhs.rel_lits = [(r, o0-s,o1-s,a0,a1) for r, o0,o1,a0,a1 in self.rel_lits 
                                if (o0 >= s and o1 >= s and o0 < e and o1 < e)]
            # print(s, e,)# [(o0 >= s, o1 >= s, o0 < e, o1 < e) for o0,o1,a0,a1 in self.eq_toks ])
            # print("NEW")                                 
            # print(self, self.rel_lits)
            # print("SLICE", new_rhs.eq_toks)


            return new_rhs
        raise NotImplemented

    def insert(self, i, val, optional=False):
        self.items.insert(i, val)
        self.optional_mask = np.insert(self.optional_mask, i, bool(optional))
        self.rel_lits += [(r, o0+1 if o0 >= i else o0, o1+1 if o1 >= i else o1, a0, a1)
                         for r,o0,o1,a0,a1 in self.rel_lits]

    def __add__(self, other):
        s = len(self)

        new_rhs = copy(self)
        new_rhs.items += other.items

        new_rhs.optional_mask = np.zeros(len(new_rhs.items),dtype=np.bool_)
        optionals = self.optional_mask.nonzero()[0]
        optionals += s + other.optional_mask.nonzero()[0]
        new_rhs.optional_mask[optionals] = True

        new_rhs.rel_lits = self.rel_lits
        new_rhs.rel_lits += [(r, o0+s,o1+s,a0,a1) for r,o0,o1,a0,a1 in other.rel_lits]


        return new_rhs


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
        elif(hasattr(other, 'name')):
            return self.name == other.name
        else:
            return False

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
        # print(">>", rhs.symbol)

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
    def RHSs(self):
        rhss = []
        for sym, _rhss in self.prods.items():
            rhss += _rhss
        return rhss

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
                        lst.append(rhs)
                        self._rhss_with_map[item] = lst
        
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
        # print(":::", self._upstream_rhs_map)
        self._ensure_upstream_rhss()
        # print(":::", self._upstream_rhs_map)
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

    def _simplify(self):
        lone_symbs = []
        for sym, rhss in self.prods.items():
            for i in range(len(rhss)-1, -1, -1):
                rhs = rhss[i]
                # If a RHS consists of a single non-terminal symbol then
                #  substitute it for all of its downstream RHSs.
                lone_symb = rhs.items[0]
                if(len(rhs) == 1 and len(self.prods.get(lone_symb,[])) > 0):
                    # print(lone_symb, self.get_RHSs_with(lone_symb))
                    lone_symbs.append(lone_symb)
                    del rhss[i]
                    for l_rhss in self.prods[lone_symb]:
                        l_rhss.symbol = sym
                        rhss.append(l_rhss)

        self._clear_internals()

        # Remove any symbols that never appear in RHSss
        for sym in lone_symbs:
            if(len(self.get_RHSs_with(sym)) == 0):
                del self.symbols[sym]
                del self.prods[sym]

        



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
           and  (kind == "replace" or pe_kind == "replace" # Either is replace
            or kind == pe_kind)): # or are same kind

            # If delete and contiguous w/ prev (delete/replace)
            if(kind == "delete"):
                # NOTE: Should probably delete this exception
                # Skip merge of replace, delete
                # if(kind == "delete"):
                #     merged_edits.append(pe) 
                #     pe = e
                # else:
                    # Extend deletion region (MAIN PART)
                pe = (pe_kind, pe[1], e[2], *pe[3:])

            # If insert and contiguous w/ prev (insert/replace)
            elif(kind == "insert"):
                # Append 
                # if(pe_kind == "insert"):
                #     # pe = (pe_kind, pe[1], pe[2]+e[2])
                # else: # pe_kind == "replace"
                pe = (pe_kind, pe[1], pe[2], pe[3]+e[3])

            # If replace and contiguous w/ prev (insert/delete/replace)
            elif(kind == "replace"):                
                ins_vals = [] if pe_kind == "delete" else pe[-1]
                pe = ('replace', pe[1], e[2], ins_vals+e[3])
                

        else:
            merged_edits.append(pe) 
            pe = e
    merged_edits.append(pe) 

    return merged_edits


def find_edits(seq0, seq1, alignment=None, return_align=False):
    eq_matrix = np.zeros((len(seq0), len(seq1)))

    # print(seq0)
    # print(seq1)
    if(alignment is None):
        for i, c1 in enumerate(seq0):
            for j, c2 in enumerate(seq1):
                if((hasattr(c1, 'skill_eq') and c1.skill_eq(c2)) or
                    c1 == c2):
                    eq_matrix[i][j] = 1
    else:
        for i, j in enumerate(alignment):
            if(j == -1):
                continue
            c1, c2 = seq0[i], seq1[j]
            # print(c1,c2, c1.skill_eq(c2))
            if((hasattr(c1, 'skill_eq') and c1.skill_eq(c2)) or
                c1 == c2):
                eq_matrix[i][j] = 1

    # print(seq0, seq1)
    # print(eq_matrix)
    edits = []
    c = 0
    s0 = s1 = 0
    e0 = e1 = 1
    spanning = False
    alignment = -1*np.ones(len(seq0), dtype=np.int64)
    for _ in range(max(len(seq0),len(seq1))):
        # print(f's0={s0} e0={e0} s1={s1} e1={e1}, c={c}')
        # Note: printing a copy with +.1 added to the span of s0:e1,s1:e1
        #  is helpful for ensuring that each iterator is changing correctly
        # eq_copy = eq_matrix.copy()
        # eq_copy[s0:e0,s1:e1] += .1
        # print(eq_copy)
        # print(e0-1, s1, e0 <= len(seq0), eq_matrix[e0-1,s1:])
        # print(s0 ,e1-1, e1 <= len(seq1), eq_matrix[s0:,e1-1])
        # print(eq_copy[s0:e0,s1:e1])

        is_del = e0 <= len(seq0) and np.sum(eq_matrix[e0-1,s1:]) == 0
        is_ins = e1 <= len(seq1) and np.sum(eq_matrix[s0:,e1-1]) == 0

        # print(is_del, is_ins)
        if(is_del or is_ins):
            if(is_del and is_ins):
                # print(('replace', e0-1))
                
                edits.append(('replace', e0-1, e0, [seq1[e1-1]]))
                alignment[e0-1:e0] = e1-1
            elif(is_del):
                # print(('delete', s0))
                edits.append(('delete', e0-1, e0))
            else:
                # print(('insert', s0))
                edits.append(('insert', e0-1, e0-1, [seq1[e1-1]]))

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
            

            if(e0 >= len(seq0) or e1 >= len(seq1)):
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

        if(e0 >= len(seq0) or e1 >= len(seq1)):
            break

        e0, e1 = e0+1, e1+1

    if(e0 < len(seq0)):
        # for i in range(e0,len(seq0)):
        edits.append(("delete", e0, len(seq0)))

    if(e1 < len(seq1)):
        # print("THIS", seq1, s0, seq1[e1:])
        edits.append(("insert", s0, s0, list(seq1[s1:])))

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

def apply_rhs_edits(rhs, edits, unorder=False):
    # print("EDITS", edits)
    new_rhs = copy(rhs)

    delete_edits = [edit for edit in edits if edit[0]=='delete']
    ins_repl_edits = [edit for edit in edits if edit[0] in ['insert', 'replace']]

    # For each delete edit make the corresponding symbol optional.
    for edit in delete_edits:
        for i in range(edit[1],edit[2]):
            new_rhs.set_optional(i)

    if(len(ins_repl_edits) > 0):
        items_opt = list(zip(new_rhs.items, new_rhs.optional_mask))

        # For each insert edit add an optional symbol. For each replace 
        #  make the appropriate replacement with non-optional symbols.
        #  Apply in reversed order (should already be ordered by index)
        #  to prevent index invalidation
        for edit in reversed(ins_repl_edits):
            kind = edit[0]
            if(kind == "insert"):
                items_opt = items_opt[:edit[1]] + [(e,1) for e in edit[3]] + items_opt[edit[1]:]
            elif(kind == "replace"):
                new_syms = [(symb, 0) for symb in edit[3]]
                items_opt = items_opt[:edit[1]] + new_syms + items_opt[edit[2]:]


        items = [x[0] for x in items_opt]
        optional_mask = np.array([x[1] for x in items_opt])
        new_rhs.items = items
        new_rhs.optional_mask = optional_mask
        
    # print("UNORDER", unorder)
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
    print("--EDITS", edits)
    # seq1 = rhs.items

    if(symbol_factory is None):
        symbol_factory = SymbolFactory(ord("T"))

    

    # Look for an unorder edit that spans the whole RHS
    full_unorder = False
    has_ins = any([edit[0] in ('insert', 'replace') for edit in edits])
    if(not has_ins):
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

def generalize_rhs(rhs, subseq, edits, alignment, lit_match_ids=None):
    new_rhs = copy(rhs)

    
    # Alignment with -1 for bit that become part of disjoint RHSs
    merge_alignment = alignment.copy()
    # Alignment with -1 for bit that are replaced with non-terminals
    retain_alignment = alignment.copy()
    # The amount by which other conditions need be shifted to be retained
    shift = np.zeros(len(rhs), dtype=np.int64)
    for edit in reversed(edits):
        kind = edit[0]
        if(kind == "replace"):
            merge_alignment[edit[1]:edit[2]] = -1
            retain_alignment[edit[1]:edit[2]] = -1
            shift[edit[2]:] += (edit[2]-edit[1])-1
        elif(kind == "unorder"):
            shift[edit[2]:] += (edit[2]-edit[1])-1
            retain_alignment[edit[1]:edit[2]] = -1
        # elif(kind == "delete"):
        #     retain_alignment[edit[1]:edit[2]] = np.arange(edit[1], edit[2])

    new_rel_lits = []
    if(lit_match_ids is None):
        lit_match_ids, _ = _check_rel_lit_matches(rhs, subseq, alignment)

    # print(rhs.rel_lits)
    for m_id in lit_match_ids:
        r, o0, o1, a0, a1 = rhs.rel_lits[m_id]
        if(retain_alignment[o0] == -1 or retain_alignment[o1] == -1):
            continue

        _o0, _o1 = o0 - shift[o0], o1 - shift[o1]
        # print(o0, o1 ,"->", _o0, _o1)
        if(_o0 < 0 or  _o1 < 0):
            continue
        new_rel_lits.append((r, _o0, _o1, a0, a1))
    new_rhs.rel_lits = new_rel_lits
    # print(subseq)
    # if(isinstance(subseq, RHS)):
    #     new_rhs.rel_lits = list(set(rhs.rel_lits).intersection(subseq.rel_lits))
    # else:
    #     # Select subset of relative literals that match 
    #     rel_lits = []
    #     for rel_lit in rhs.rel_lits:
    #         if(rel_lit_match(rel_lit, subseq)):
    #             rel_lits.append(rel_lit)
    #     new_rhs.rel_lits = rel_lits

    # Apply where-part anti-unification 
    _subseq = subseq.items if isinstance(subseq, RHS) else subseq
    new_items = [*rhs.items]
    for i, (ind, old_item) in enumerate(zip(merge_alignment, rhs.items)):
        if(isinstance(old_item, Skill) and ind != -1):
            new_items[i] = old_item.merge(_subseq[ind])
    new_rhs.items = new_items                

    # new_rhs = RHS(new_items, symbol=self.symbol, unordered=self.unordered)
    # new_rhs.optional_mask = self.optional_mask.copy()
    return new_rhs


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
    def __init__(self, rhs, span, subseq=[], edits=[], alignment=None, lit_matches=None, gen_penl=0.0, nomatch=False):
        self.rhs = rhs
        self.span = span
        self.subseq = subseq
        self.edits = edits
        self.alignment = np.arange(len(subseq)) if alignment is None else alignment
        self.lit_matches = lit_matches
        self.gen_penl = gen_penl
        self.nomatch = nomatch
        self.recalc_cost()

    def __str__(self):
        return f"{self.rhs.symbol}->{self.rhs} {'nomatch' if self.nomatch else self.edits}"

    __repr__ = __str__

    def __copy__(self):
        return RHSChange(self.rhs, self.span, self.subseq, self.edits, self.alignment,
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

    def ensure_changes_computed(self, symbol_factory=None):
        if(not hasattr(self, 'parent_edits')):
            full_unorder, unorder_substs, disjoin_substs, parent_edits = \
                edits_to_changes(self.rhs, self.edits, symbol_factory)
            self.full_unorder = full_unorder
            self.unorder_substs = unorder_substs
            self.disjoin_substs = disjoin_substs
            self.parent_edits = parent_edits

    def apply(self, grammar=None, symbol_factory=None, old_rhs=None):
        # if(grammar and len(self.edits) == 0):
        #     print("COPY")
        #     return grammar, self.rhs 

        self.ensure_changes_computed(symbol_factory)

        # print("P", self.rhs, self.parent_edits)
        old_rhs = self.rhs if(old_rhs is None) else old_rhs
        new_rhs = generalize_rhs(old_rhs, self.subseq, self.edits, self.alignment, self.lit_matches)
        # print("NEW RHS", new_rhs, self.full_unorder)
        # print("P EDITS", self.parent_edits)
        new_rhs = apply_rhs_edits(new_rhs, self.parent_edits, self.full_unorder)
        # print("After Edits", new_rhs)        
        if(grammar is None):
            return None, new_rhs 

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


    



    #     rhs = self.rhs
    #     return f"changes({rhs.symbol}->{''.join([str(x) for x in rhs.items])}, {seq2})")

# def _count_item_rels(rhs, rel_lits):
#     counts = np.ones(len(rhs), dtype=np.int64)
#     for r, o0, o1, a0, a1 in rhs.rel_lits:
#         counts[o0] += 1
#         counts[o1] += 1
#     return counts

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
    if(isinstance(subseq, RHS)):
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


# def _check_rels(rhs, seq, rel_lits):
#     total = np.ones(len(seq), dtype=np.int64)
#     matches = np.ones(len(seq), dtype=np.int64)
#     for rel_lit in rel_lits:
#         r, o0, o1, a0, a1 = rel_lit
        
#         total[o0] += 1
#         total[o1] += 1

#         if(rel_lit_match(rel_lit, seq)):
#             matches[o0] += 1
#             matches[o1] += 1
#     # print(rel_lits)
#     # print("!!!", matches / total, len(matches) - np.sum(matches / total))
#     return len(matches) - np.sum(matches / total)

# def _calc_gen_penalty(rhs, subseq, edits, alignment):
#     # print(rhs, "...", subseq)
#     # if(len(rhs.rel_lits) > 0):
#     #     print()
#     rel_lits = _align_rels(rhs.rel_lits, alignment)

#     # print("RT:", rel_lits, alignment, edits)
#     if(isinstance(subseq, RHS)):
#         intr = set(rel_lits).intersection(subseq.rel_lits)
#         # print(intr)
#         total = _count_item_rels(rhs, rel_lits)
#         shared = _count_item_rels(rhs, intr)
#         return len(shared) - np.sum(shared / total)
#         # print("!!!", shared / total, len(shared) - np.sum(shared / total))
#     else:
#         return _check_rels(rhs, subseq, rel_lits)



    
            # print(item0, item1, r, rel)






def find_rhs_change(rhs, subseq, alignment=None, span=None, seq=None):
    # print(f"changes({rhs.symbol}->{''.join([str(x) for x in rhs.items])}, {seq2})")

    seq1 = rhs.items
    seq2 = subseq.items if isinstance(subseq, RHS) else subseq
    edits, alignment = find_edits(seq1, seq2, alignment, return_align=True)

    lit_matches, gen_penl = _check_rel_lit_matches(rhs, subseq, alignment)
    # print("align", seq1, seq2, edits, alignment)

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

    if(span is None):
        span = (0, len(seq1))
    change = RHSChange(rhs, span, subseq, 
                    filtered_edits, alignment=alignment,
                    lit_matches=lit_matches, gen_penl=gen_penl)
    
    return change
    # g = Grammar()
    # g.add_production("S", rhs)
    # new_g, new_rhs = change.apply(g)
    # print(g)

    # return new_rhs

def test_find_edits():
    # # 0. Aligned
    # print("0.")
    seq1 = 'ABCDEF'
    seq2 = 'XCBAFEDY'
    edits, align = find_edits(seq1, seq2, True)
    print(edits)
    print(seq2)
    print(align)

    edits, align = find_edits(seq1, seq1, True)
    print(align)

    # raise ValueError()

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
    print(find_edits(seq1, seq2))
    assert find_edits(seq1, seq2) == \
        [('delete', 1, 2), ('delete', 4, 5)]

    # # 4. Unorder + Delete
    print("4.")
    seq2 = 'CBED'
    assert find_edits(seq1, seq2) == \
        [('delete', 0, 1), ('unorder', 1, 3), ('unorder', 3, 5), ('delete', 5, 6)]


    # # 5. Unorder Subsume Delete
    print("5.")
    seq2 = 'CAFD'
    assert find_edits(seq1, seq2) == \
        [('unorder', 0, 3), ('delete', 1, 2), ('unorder', 3, 6), ('delete', 4, 5)]

    # # 6. Insert
    print("6.")
    seq2 = 'XABCDEFY'
    print(find_edits(seq1, seq2))
    assert find_edits(seq1, seq2) == \
        [('insert', 0, 0, ['X']),('insert', 6, 6, ['Y'])]

    # # 7. Unorder + Insert
    print("7.")
    seq2 = 'XCBAFEDY'
    assert find_edits(seq1, seq2) == \
        [('insert', 0, 0, ['X']), ('unorder', 0, 3), ('unorder', 3, 6), ('insert', 6, 6, ['Y'])]

    # # 8. Unorder Subsume Insert
    print("8.")
    seq2 = 'CXBAFEYD'
    assert find_edits(seq1, seq2) == \
        [('unorder', 0, 3), ('insert', 1, 1, ['X']), ('unorder', 3, 6), ('insert', 5, 5, ['Y'])]

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

    # # 12. Edge Cases
    print("12.1")
    seq2 = 'CXZAFYQD'
    assert find_edits(seq1, seq2) == \
        [('unorder', 0, 3), ('replace', 1, 2, ['X','Z']), ('unorder', 3, 6), ('replace', 4, 5, ['Y','Q'])]

    print("12.2")
    seq2 = 'CXYZQRV'
    assert find_edits(seq1, seq2) == \
        [('delete', 0, 2), ('replace', 3, 6, ['X','Y','Z','Q','R','V'])]
    

    print("12.3")
    seq2 = 'XYZ'
    assert find_edits(seq1, seq2) == \
        [('replace', 0, 6, ['X','Y','Z'])]
    

    # # 13. Inner Delete + Trailing Insert
    print("13.")
    assert find_edits("EQF", "EF") == \
        [('delete', 1, 2)]
    assert find_edits("EQF", "EFH") == \
        [('delete', 1, 2), ('insert', 3, 3, ['H'])]
    assert find_edits("EQF", "EFHZ") == \
        [('delete', 1, 2), ('insert', 3, 3, ['H','Z'])]

    print("14.")
    seq1 = ['d', 'a20', 'c20', 'c31', 'a31', 'c32', 'a32', 'cp3', 'd']
    seq2 = ['d', 'a20', 'a21', 'a22','d']
    print(find_edits(seq1, seq2))
    print(find_edits(seq2, seq1))


def _seq_to_initial_grammar(seq):
    g = Grammar()
    # seq = [x.skill for x in seq]    
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

def calc_overlap(pattern, sequence, skip_penalty=0.1):
    l0, l1 = len(pattern), len(sequence)
    scores = np.zeros((l0, l1), dtype=np.float64)    
    for i, s0 in enumerate(pattern):
        for j, s1 in enumerate(sequence):
            # print(":", s0, s1)
            if(hasattr(s0,'overlap')):
                scores[i][j] = s0.overlap(s1)
            else:
                scores[i][j] = s0 == s1
    # print(scores)
    alignment = np.argmax(scores, axis=1)
    assert(len(alignment) == len(pattern))

    mx = np.max(scores, axis=1)
    assert(len(mx) == len(pattern))

    alignment[mx==0.0] = -1
    
    mx = [-skip_penalty if x == 0 else x for x in mx]
    overlap = np.sum(mx)
    if(len(sequence) > len(pattern)):
        overlap -= skip_penalty*(len(sequence)-len(pattern))
    return max(overlap,0), alignment#/(l0)#+abs(l0-l1))


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
    


def get_best_changes(grammar, seq, rhss):
    best_changes = []
    for rhs in rhss:
        L = len(rhs)
        items = [x for x in rhs.items]
        # item_set = set(items)

        

        # print(rhs.symbol,"->", rhs)

        # best_mp_tree = None

        best_rcs = []

        # Convolved RHS application 
        for i in range(len(seq)):
            best_rc = None
            # best_rc_cost_tup = float('inf')#(float('inf'), 0)
            best_rc_cost_tup = (float('inf'), 0)

            # for j in range(i+1, min(len(seq), i+L+1)+1):
            for j in range(i+1, len(seq)+1):
                subseq = seq[i:j]

                

                overlap, alignment = calc_overlap(items, subseq)
                rc = find_rhs_change(rhs, subseq, alignment, span=(i,j), seq=seq)

                # print("OV", overlap, rhs, subseq)

                
                # print(i,j, subseq)
                # print(f" {rhs.symbol}->{rhs} : c:{rc.cost:.2f} g:{rc.gen_penl:.2f} o:{overlap:.2f} r:{rc.cost/overlap if overlap else float('inf'):.2f}", subseq, rc.edits)
                
                # Skip if there is no overlap  
                if(overlap == 0):
                    continue

                
                # print(f"   {rhs.symbol}->{rhs} : c:{rc.cost:.2f} g:{rc.gen_penl:.2f}", subseq, rc.edits)
                # cost_tup = rc.cost#(rc.cost/overlap, -overlap)
                cost_tup = (rc.cost/overlap, -overlap)


                # print(f"{rc.cost/overlap:.2f}", subseq, rc.edits)
                if(cost_tup < best_rc_cost_tup):
                    best_rc_cost_tup = cost_tup
                    best_rc = rc

                # print(rc.cost)
            if(best_rc):
                # print(f"{rhs.symbol}->{rhs} : {best_rc.span} c:{best_rc.cost:.2f} g:{rc.gen_penl:.2f} o:{-best_rc_cost_tup[1]:.2f} r:{best_rc.cost/overlap if overlap else float('inf'):.2f}", best_rc.subseq, best_rc.edits)
                # print(f"{rhs.symbol}->{rhs} : {best_rc.span} c:{best_rc.cost:.2f} g:{best_rc.gen_penl:.2f}", best_rc.subseq, best_rc.edits)
                best_rcs.append(best_rc)
        
        # print([(x.cost,x.span) for x in best_rcs])
        # print(best_rcs)
        # print("BEST RCS", len(best_rcs))
        best_rcs = greedy_span_rcs(best_rcs)
        # print("BEST RCS", len(best_rcs))
        best_changes += best_rcs
        # for rc in best_rcs:
        #     print(f"{rhs.symbol}->{rhs} : {rc.span} c:{rc.cost:.2f}", rc.seq[rc.span[0]:rc.span[1]], rc.edits)
        #     best_changes.append(rc)

    # for rc in best_changes:
    #     print(f"{rhs.symbol}->{rhs} : {rc.span} c:{rc.cost:.2f}", rc.seq[rc.span[0]:rc.span[1]], rc.edits)
    #     best_changes.append(rc)

        # if(best_rc is None):
        #     # raise ValueError("NO MATCH OCCURED")
        #     best_rc = RHSChange(rhs, (0, L), None, nomatch=True, seq=seq)
        # best_changes.append(best_rc)
        # s0, s1 = best_rc.span
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
            new_edits.append(('insert', head, head, seq[head:s0]))
            # for j in range(head,s0):
                # if(j < len(seq_trees)):
                #     print("SQT:", seq_trees[j])
                # if(not upstream_covered[j]):
                
            new_edits += rc.edits
            # ins_cost = cost_of_edits(rc.rhs, new_edits, e0-s0)
            # Prepend any uncovered symbols to beginning
            rc = RHSChange(rc.rhs, (head,e0), edits=new_edits)
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
            new_rcs[-1] = RHSChange(rc.rhs, (head,s1), edits=new_edits)
            
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
        # self.cost = self.change.cost + sum([rc.cost for rc in self.child_trees if rc])
        self.cost = sum([rc.cost for rc in self.changes if rc])

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

    def iter_trees(self):
        if(not hasattr(self, '_tree_iter')):
            # Collect Trees
            tree_iter = []
            pts = [self]
            depth = 0
            while(len(pts) > 0):
                new_pts = []
                for i, pt in enumerate(pts):
                    if(not pt or isinstance(pt, ParseHole)):
                        continue
                    # print("CHANGE", depth, i, pt.change.cost, pt.change.rhs.symbol, "->", pt.change.rhs, pt.change.edits, )
                    # tree_iter[pt.change.rhs] = pt.change
                    tree_iter.append(pt)
                    # yield pt
                    new_pts += pt.child_trees
                depth += 1
                pts = new_pts
            self._tree_iter = tree_iter
        return self._tree_iter
    @property
    def changes(self):
        if(not hasattr(self, '_changes')):
            self._changes = []
            for pt in self.iter_trees():
                self._changes.append(pt.change)
        return self._changes

    @property
    def joined_changes(self):
        if(not hasattr(self, '_joined_changes')):
            joined_changes
            rhs_to_pts = {}

            for pt in self.iter_trees():
                if(len(pt.change.edits) == 0 or 
                   id(pt) in covered):
                    continue
                _, lst = joined_changes.get(pt.change.rhs, [])
                lst.append(pt)
                joined_changes[pt.change.rhs] = (None, lst)
                covered.add(id(pt))

            for rhs, (_,pts) in [*joined_changes.items()]:

                if(len(pts) > 1):
                    _edits = []
                    for pt in pts:
                        for edit in pt.change.edits:
                            if(edit not in _edits):
                                _edits.append(edit)
                    # Combine
                    edits = sorted(list(_edits),key=lambda x: x[1])
                    RHSChange()
                else:
                    rc = pts[0].rc


                for pt in pts:
                    pt.change.edits = edits
                    orig_cost = pt.change.cost
                    pt.change.recalc_cost()
                    pt.cost += pt.change.cost-orig_cost
            self._joined_changes = 'TOTODOO'


    # def _join_changes(self, other=None):
        


        # Then update trees
                

                







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
def get_spanning_trees(parse_trees):
    if(len(parse_trees) == 0):
        return parse_trees

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
    print(">>", [(f"{sum([pt.cost for pt in path]):.2f}", [(pt.symbol, *pt.change.span) for pt in path]) for path in paths])
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
    print([type(x) for x in symbols])
    how_parts = set([x.skill.how_part for x in symbols if hasattr(x, 'skill')])
    print("HP", how_parts)
    upstream_rhss = {}        
    for _, rhss in grammar.prods.items():
        for rhs in rhss:
            for item in rhs.items:
                print(item, symbols, getattr(item,'how_part',None))
                if(item in symbols or getattr(item,'how_part',None) in how_parts):
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
    # print(":>", rc)

    if(not rc.edits or len(rc.edits) == 0):
        return pt
    # Check the highest edits in the parse tree and see if they
    #  overlap with an unparsed a ParseHole symbol. 
    for edit_ind, edit in reversed([*enumerate(rc.edits)]):
        # NOTE: It is not clear if needs to also happen on replaces
        if(edit[0] in ('insert')):#, 'replace')):
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
            prev = [('insert', 0, 0, [hp.symbol for hp in p_lst])]
        if(len(n_lst) > 0):
            loc = _rc.span[1]-_rc.span[0]
            nxt = [('insert', loc, loc, [hp.symbol for hp in n_lst])]
        edits = [*prev, *_rc.edits, *nxt]
        
        s,e = _rc.span[0]-len(p_lst), _rc.span[1]+len(n_lst)
        # cost = cost_of_edits(rhs, edits, e-s)
        rc = RHSChange(rhs, (s,e), edits=edits)
        _pt = ParseTree(rc, child_trees=adj.child_trees)
        # print(" ->", rc.cost, rc)
        # print(_pt)
        # print()
        replacements.append((adj, _pt))

    pt = pt._deep_copy(replacements)
    pt.change.recalc_cost()
    pt.recalc_cost()
    return pt


def join_edits(edits0, edits1):
    # TODO: Make fancier when more test examples

    return list(set(edits0).intersection(set(edits1)))

def _parse_recursions(trees, seq):
    rec_rhs = None
    start = 0
    spans = []
    for i, (tree, item) in enumerate(zip(trees, seq)):
        # print(tree.rhs.is_recursive)
        if(type(tree) is ParseTree and tree.rhs.is_recursive):
            rhs = tree.rhs
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
    print("MIN RC", min_rc.cost, min_rc)
    return [rc for rc in changes if rc.rhs.symbol == min_rc.rhs.symbol or rc.cost == min_cost]




def _bottom_up_recurse(grammar, seq, rhss=None, seq_trees=None):
    print("\nSEQ", seq, len(seq), len(seq_trees) if seq_trees else None)
    out = []

    # print("rhss", rhss)

    # Find RHSs immediately upstream from the symbols in seq
    # if(rhss is None or len(rhss) == 0):
    #     print("THIS")
    #     rhss = get_upstream_RHSs(grammar, set(seq))
    import itertools 
    rhss = list(itertools.chain(*grammar.prods.values()))

    # print("SEQ",_sequence_upstreams(grammar, rhss))
    # print(_sequence_upstreams(grammar, rhss)[0])
    # rhss = _sequence_upstreams(grammar, rhss)[0]

    # print("rhss", rhss)

    # Check terminating case when seq is the start symbol
    if(len(rhss) == 0):
        return seq_trees

    # Try to parse using each rhs
    best_changes = get_best_changes(grammar, seq, rhss)
    print("best_changes", [(rc, f"{rc.cost:.2f}", rc.span) for rc  in best_changes])
    best_changes = _only_least_cost(best_changes)

    

    # Turn each partial parse into a parse tree 
    parse_trees = []
    for rc in best_changes:
        print(":", rc, rc.span, rc.cost, rc.gen_penl, seq[rc.span[0]:rc.span[1]])
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
    # print("parse_trees", parse_trees)
    sp_trees = get_spanning_trees(parse_trees)
    # print("sp_trees", sp_trees)
    # sp_trees = filter_spanning_trees(sp_trees)
    for k, tree_seq in enumerate(sp_trees):
        if(len(tree_seq) == 1):
            out += tree_seq
            continue
        # print("----<", seq)
        # change_seq = fill_span_holes(change_seq, seq)
        # for i, pt in enumerate(tree_seq):
        #     rc = pt.change
        #     print( (k, i), ":", rc.rhs.symbol, rc.span, f"{rc.cost:0.2f}", rc.edits)

        # Make the new sequence by replacing each RHS parse
        #  with its upstream symbol. Keep track of any ParseHoles. 
        new_trees, new_seq = _make_seq_and_holes(tree_seq, seq, seq_trees)
        new_trees, new_seq = _parse_recursions(new_trees, new_seq)
        out += _bottom_up_recurse(grammar, new_seq, seq_trees=new_trees)

    # print("OUT", out)
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
    # TODO need to account for spanning deletes here
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

from proclrn.parse import parse_only_deletes

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

    changes = parse_tree.changes

    rhs_to_rcs = {}
    for rc in changes:
        lst = rhs_to_rcs.get(rc.rhs,[])
        lst.append(rc)
        rhs_to_rcs[rc.rhs] = lst

    # new_prods = {}
    # for symb, rhss in g.prods:
    #     for rhs in rhss:
    #         rc = changes[rhs]

    new_g = g.__copy__()
    # for rc in changes:
    #     rc.apply(new_g, symbol_factory)

    for rhs, rcs in rhs_to_rcs.items():
        # rc = rcs[0]
        for rc in rcs:
            _, rhs = rc.apply(new_g, symbol_factory, old_rhs=rhs)
        # for i in range(1, len(rcs)):
            # print('>>', rc.edits, rcs[i].edits)
            # rc.edits = join_edits(rc.edits, rcs[i].edits)
        


    print(new_g)
    new_g._simplify()
    return new_g

    
def generalize_from_seq(g, seq):
    print("----", seq, "-----")
    pt = parse_w_changes(g, seq)
    g = apply_grammar_changes(g, pt)
    for rhss in g.prods.values():
        for rhs in rhss:
            print(">", [(x.how_part, x.where_matches, x.skill_apps) for x in rhs.items if isinstance(x,Skill)])
    return g

class IE():
    def __init__(self, name):
        self.name = name
        self.relations = {}

    def add_relation(self, kind, other):
        self.relations[other] = kind
        self.relations[kind] = other

    def get_relation(self, other):
        if(self == other):
            return '='
        return self.relations.get(other,None)

    def __eq__(self, other):
        return self.name == getattr(other, 'name', other)

    def __hash__(self):
        return hash(self.name)

    def __str__(self):
        return self.name

    __repr__ = __str__ 


def newSA(hp, sel, args, short_name=None):
    if(short_name is None):
        short_name = f"{hp}({sel})"
    skill = PrimSkill(how_part=hp)
    return SkillApp(skill, sel, args, short_name=short_name)    

def make_mc_skill_apps():
    '''c3 c2 c1
          t2 t1 t0
     +    b2 b1 b0
     -------------
       o3 o2 o1 o0
    '''
    c3 = IE("c3")
    c2 = IE("c2")
    c1 = IE("c1")
    t2 = IE("t2")
    t1 = IE("t1")
    t0 = IE("t0")
    b2 = IE("b2")
    b1 = IE("b1")
    b0 = IE("b0")
    o3 = IE("o3")
    o2 = IE("o2")
    o1 = IE("o1")
    o0 = IE("o0")
    done = IE("done")

    c3.add_relation("r", c2)
    c2.add_relation("l", c3)

    c2.add_relation("r", c1)
    c1.add_relation("l", c2)

    t2.add_relation("r", t1)
    t1.add_relation("l", t2)

    t1.add_relation("r", t0)
    t0.add_relation("l", t1)

    b2.add_relation("r", b1)
    b1.add_relation("l", b2)

    b1.add_relation("r", b0)
    b0.add_relation("l", b1)

    o3.add_relation("r", o2)
    o2.add_relation("l", o3)

    o2.add_relation("r", o1)
    o1.add_relation("l", o2)

    o1.add_relation("r", o0)
    o0.add_relation("l", o1)


    # a2 = Skill(how_part='a2')
    # c2 = Skill(how_part='c2')
    # a3 = Skill(how_part='a3')
    # c3 = Skill(how_part='c3')
    # cp = Skill(how_part='cp')
    # prs = Skill(how_part='prs')

    a20 = newSA('a2', o0, [t0, b0], short_name='a20')
    c20 = newSA('c2', c1, [t0, b0], short_name='c20')
    a21 = newSA('a2', o1, [t1, b1], short_name='a21')
    c21 = newSA('c2', c2, [t1, b1], short_name='c21')
    a22 = newSA('a2', o2, [t2, b2], short_name='a22')
    c22 = newSA('c2', c3, [t2, b2], short_name='c22')
    cp3 = newSA('cp', o3, [c3], short_name='cp3')
    a31 = newSA('a3', o1, [c1, t1, b1], short_name='a31')
    c31 = newSA('c3', c2, [c1, t1, b1], short_name='c31')
    a32 = newSA('a3', o2, [c2, t2, b2], short_name='a32')
    c32 = newSA('c3', c3, [c2, t2, b2], short_name='c32')

    d = newSA('d', done, [], short_name='d')

    return (a20, c20, a21, c21, a22, c22, cp3, a31, c31, a32, c32, d)

def make_subtract_skill_apps():
    ''' a2 a1 a0
        b2 b1 b0
        u2 u1 u0
     -  l2 l1 l0
     -----------
        o2 o1 o0
    '''
    ad0 = newSA('ad', 'b0', ['u0'], short_name='ad0')
    ad1 = newSA('ad', 'b1', ['u1'], short_name='ad1')
    ad2 = newSA('ad', 'b2', ['u2'], short_name='ad2')

    dc0 = newSA('dc', 'b0', ['u0'], short_name='dc0')
    dc1 = newSA('dc', 'b1', ['u1'], short_name='dc1')
    dc2 = newSA('dc', 'b2', ['u2'], short_name='dc2')

    sb0 = newSA('sb', 'o0', ['u0', 'l0'], short_name='sb0')
    sb1 = newSA('sb', 'o1', ['u1', 'l1'], short_name='sb1')
    sb2 = newSA('sb', 'o2', ['u2', 'l2'], short_name='sb2')

    cp0 = newSA('cp', 'o0', ['u0'], short_name='cp0')
    cp1 = newSA('cp', 'o1', ['u1'], short_name='cp1')
    cp2 = newSA('cp', 'o2', ['u2'], short_name='cp2')

    d = newSA('d', 'd', [], short_name='d')

    return (ad0, ad1, ad2, dc0, dc1, dc2, sb0, sb1, sb2, cp0, cp1, cp2, d)


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
        [a20, c20, a21, c21, a22, c22, cp3, d],
        [c20, a20, c21, a21, c22, a22, cp3, d],
        [a20, c20, c31, a31, c32, a32, cp3, d],
        [a20, c20, c31, a31, c32, a32, d],
        [a20, c20, a31, c31, a32, c32, cp3, d],
        [a20, c20, a21, c21, a32, c32, d],
        [a20, c20, a21, c21, a32, d],
        # [a20, c20, a21, c21, a32, c32, d],
    ]
    from random import shuffle
    # seqs = [[a20, c20, a31, c31, a22, c22, cp3, d],
    #         [c20, a20, a21, c22, a22, cp3, d]]
    order = list(range(len(seqs)))#[:6]
    # shuffle(order)
    # order = [3, 0, 1, 2]#, 5]#, 4]
    # order = [5, 0, 2, 1, 4, 3]
    # order = [3, 0, 1, 5, 4, 2]
    # order = [3, 2, 0, 1, 4, 5]
    # order =  [3, 1, 4, 0, 2]
    # order = [2, 1, 0, 4, 3, 5]
    # order = [3, 4, 0, 2, 1]
    # order = [2, 3, 1, 4, 0]
    # order = [2, 3, 1, 4, 0]
    # order = [3, 0, 2, 1, 4]
    # order = [3, 1, 0, 4, 2]
    # order = [2, 0, 4, 3, 1]
    # order = [3, 1, 0, 4, 2]

    # Sequences which create bad recursion if parse filtering is turned on
    # order = [2, 0, 4, 3, 5, 1] 
    # order = [0, 3, 1, 5, 2, 4] 

    print("ORDER", order)

    try:
        print("INITIAL SEQ:", seqs[order[0]]) 
        g, rhs = _seq_to_initial_grammar(seqs[order[0]])
        # g = find_recursion(g)
        print(f'\ngrammar:\n{g}')
        for i in range(1,len(order)):
            seq = seqs[order[i]]
            g = generalize_from_seq(g, seq)    
            # g = find_recursion(g)
            pt = parse_w_changes(g, seq)
            # assert pt.cost == 0.0
            print(f'\ngrammar:\n{g}')
    except Exception as e:
        print(g, i)
        print("ERROR ON ORDER", order)
        raise e



def test_hole_changes():
    (a20, c20, a21, c21, a22, c22, cp3, a31, c31, a32, c32, d) = make_mc_skill_apps()

    seqs = [
        [a20, c20, c31, a31, c32, a32, cp3, d],
        [a20, a21, a22, d],
        # [d, c20, a20, c21, a21, c22, a22, cp3, d],
        
        # [d, a20, c20, c31, a31, c32, a32, d],
        # [d, a20, c20, a31, c31, a32, c32, cp3, d],
        # [d, a20, c20, a21, c21, a32, c32, d], 
    ]

    print(a20.overlap(c20))
    print(c20.overlap(c21))
    # print(a21.overlap(c21))
    # print(a21.overlap(c21))
    # print(a20.overlap(c20))
    print('------------"')
    from random import shuffle
    order = list(range(len(seqs)))
    # shuffle(order)
    # order = [5, 0, 2, 1, 4, 3]
    # order = [3, 0, 1, 5, 4, 2]
    # order = [3, 2, 0, 1, 4, 5]
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

class SkillBase(object):
    def __init__(self, _id):
        self._id = _id

    def __str__(self):
        return self._id

    __repr__ = __str__

    def merge(self, other):
        new_skill = copy(new_skill)
         
        if(isinstance(other, Skill)):
            skill_apps = other.skill_apps
            where_matches = other.where_matches
        else:
            skill_apps = [other]
            where_matches = [(other.sel, *other.args)]
        
        new_skill.skill_apps = self.skill_apps.union(set(skill_apps))
        new_skill.where_matches = self.where_matches + where_matches
        return new_skill

    def add_app(self, skill_app):
        self.skill_apps.add(skill_app)
        self.where_matches.append((skill_app.sel, *skill_app.args))

    def arg_overlap(self, other):
        skill_apps = getattr(other, 'skill_apps', [other])
        m = 0.0
        for sa in skill_apps:
            v = max([x.arg_overlap(sa) for x in self.skill_apps])
            if(v > m):
                m = v
        return m    


class PrimSkill(object):
    def __init__(self, how_part=None):
        self.how_part = how_part
        self.skill_apps = set()
        self.where_matches = []
        super(self).__init__(self, how_part)

    def skill_eq(self, other):
        if(not isinstance(other, PrimSkill)):
            return False

        h_b = other
        if(hasattr(other, 'how_part')):
            h_b = other.how_part
        elif(hasattr(other, 'skill')):
            h_b = other.skill.how_part
        return self.how_part == h_b

    def overlap(self, other):
        # if(isinstance(other, SkillApp)):
        skill_apps = getattr(other, 'skill_apps', [other])
        # print("->", self.skill_apps, skill_apps)
        m = 0.0
        for sa in skill_apps:
            # print("->", sa)
            v = max([x.overlap(sa) for x in self.skill_apps])
            if(v > m):
                m = v
        return m




    # def __copy__(self):
    #     return Skill(self.how_part)


class SkillApp(object):
    def __init__(self, skill, sel, args, short_name=None, state=None):
        self.skill = skill
        self.sel = sel
        self.args = args
        self.state = state
        self.short_name = short_name

        if(isinstance(skill, Skill)):
            skill.add_app(self)

    def arg_overlap(self, other):
        if(not hasattr(other, 'args')):
            return 0.0

        s_a, s_b = set(self.args), set(other.args)
        if(len(s_a) == 0 and  len(s_b) == 0):
            return 1.0

        denom = max(len(s_a),len(s_b))
        if(denom == 0):
            return 0.0
        return len(s_a.intersection(s_b)) / denom

    def sel_overlap(self, other):
        # print(self.sel, other.sel, self, other)
        return float(self.sel == other.sel)

    def how_overlap(self, other):
        if(isinstance(self.skill, PrimSkill)):
            return float(self.skill.how_part == other.skill.how_part)
        else:
            return float(self.skill == other.skill)

    # In principle this could be replaced with
    #  structure mapping score of where-part
    def overlap(self, other):
        if(not isinstance(other, SkillApp)):
            return 0.0
        # print("H", self.how_overlap(other), "S", self.sel_overlap(other), "A", self.arg_overlap(other))
        # print(self.sel_overlap(other), self.sel_overlap(other), self.arg_overlap(other))
        return (self.how_overlap(other) +
                self.sel_overlap(other) +
                self.arg_overlap(other)) / 3

    def depends_on(self, other):
        return other.sel in self.args

    def __eq__(self, other):
        if(not isinstance(other, SkillApp)):
            return False
        return (self.skill.how_part == other.skill.how_part and 
                self.sel == other.sel and
                self.args == other.args)

    def __hash__(self):
        return hash((self.skill, self.sel, *self.args))

    def skill_eq(self, other):
        # print("skill_eq", type(self), type(other))

        h_b = other
        if(hasattr(other, 'how_part')):
            h_b = other.how_part
        elif(hasattr(other, 'skill')):
            h_b = other.skill.how_part

        return self.skill.how_part == h_b

    def __hash__(self):
        return hash((self.skill.how_part, self.sel, tuple(self.args)))

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
    # print(g)
    parse_trees = []
    # for rhs in g.prods[sym]:
    s, e = 0, len(seq)
    # print("SEQ", seq)
    # print(rhs.symbol, "->", rhs)
    
    # If all symbols are terminal just run get_best_changes().
    #  This could produce a full match or an edit.
    are_term = [_is_term(x,g) for x in rhs.items]
    if(all(are_term)):
        best_changes = get_best_changes(g, seq, [rhs])
        print(">>", seq, [rhs], best_changes)
        if(len(best_changes) > 0):
            rc = best_changes[0]
            print(rc)
            print("::", rc.rhs.symbol, rc.span, rc.cost)
            return rc.span, [ParseTree(rc)]
    
    if(False in are_term):
        first_nt = are_term.index(False)   
        last_nt = len(are_term) - 1 - are_term[::-1].index(False)   

        for i in range(0, first_nt):
            print("**", seq[i], rhs.items[i])
            if(rhs.items[i].overlap(seq[i]) != 1.0):
                break
            s = i+1

        for i in range(0, last_nt):
            j = -i-1
            print("**", seq[j], rhs.items[j])
            if(rhs.items[j].overlap(seq[j]) != 1.0):
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
                    # rhs, span, subseq=[], edits=[]
                    pt = ParseTree(RHSChange(rhs, (0, len(seq)), seq), child_trees=cts)
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


def test_top_down_parse():
    print("START")
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
    

def test_get_best_changes():
    (a20, c20, a21, c21, a22, c22, cp3, a31, c31, a32, c32, d) = make_mc_skill_apps()#\
        # tuple([x.skill for x in make_mc_skill_apps()])


    # g = Grammar()
    # g.add_production("S", RHS([d, "A", "B", d], optionals=[]))
    # g.add_production("A", RHS([a20, c20, d], optionals=[2]))
    # g.add_production("B", RHS([a31, c31, c32, a32, cp3], optionals=[4]))
    
    # seq = [d, a20, c20, d, a31, c31, c32, a32, d]
    # rcs = get_best_changes(g, seq, g.preterminal_RHSs)
    # for rc in rcs:
    #     print(rc)

    g = Grammar()
    g.add_production("S", RHS(["A", d], optionals=[]))
    g.add_production("A", RHS([a20, c20, "A"], optionals=[2]))

    # seq = [d, a20, c20, a21, c21, c22, a22, d]
    # seq = [d, a20, c20, a31, c31, c32, a32, d]
    seq = [d, c20, a20, c21, a21, c22, a22, d]
    rcs = get_best_changes(g, seq, g.RHSs)
    # print(rcs)
    # for rc in rcs:
    #     print(rc)
    


def test_subtract():
    (ad0, ad1, ad2, dc0, dc1, dc2, sb0, sb1, sb2, cp0, cp1, cp2, d) = make_subtract_skill_apps()

    seqs = [
        [sb0, sb1, sb2, d],
        [sb0, sb1, cp2, d],
        [dc1, ad0, sb0, cp1, d],
        [dc1, ad0, sb0, sb1, d],
        [dc1, ad0, sb0, sb1, d],
        [dc1, ad0, sb0, dc2, ad1, sb1, d],
    ]

    order = list(range(len(seqs)))
    
    print("ORDER", order)

    try: 
        g, rhs = _seq_to_initial_grammar(seqs[order[0]])
        g = find_recursion(g)
        print(f'\ngrammar:\n{g}')
        for i in range(1,len(order)):
            seq = seqs[order[i]]
            g = generalize_from_seq(g, seq)    
            g = find_recursion(g)
            pt = parse_w_changes(g, seq)
            assert pt.cost == 0.0
            print(f'\ngrammar:\n{g}')
    except Exception as e:
        print(g, i)
        print("ERROR ON ORDER", order)
        raise e


np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})

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
                    rc = find_rhs_change(recRHS, rhs[s:e])
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

            print(part_overlaps[s:min_end+1], part_weight[s:min_end+1])
            # print(part_overlaps, part_weight)
            print("score", score, min_end)

            for how_part, locs in [*skill_locs.items()]:
                if(locs[0] <= part_end):
                    locs.pop(0)
                if(len(locs) == 0):
                    del skill_locs[how_part]
        s = part_end
            # min_inds -= 
            

    part_slices.append((s,e))
    print("<<", part_slices)
    return part_slices 
    

    
    # print()




        # break
            # OR Perhaps this is enough
        # part_weight[min(locs)+1:max(locs)+1] += 1.0 / len(skill_locs)
        # print(skill, part_weight)

    

    # score = 
    # np.min()




    # print(part_overlaps[1:])
    # print(part_weight[1:])
    # print(":", part_overlaps[1:] - part_weight[1:])
    # print()


    

    # for i in range(s,e):
    #     rhs.items[]

    # partition_mask = np.ones(e-s, dtype=np.bool_)
    # for skill, locs in partition_mask:
    #     prev = s
    #     for loc in locs:
    #         partition_mask

    

    




    # Get skill that shows up most
    # sorted([(len(locs), skill) for skill, locs in skill_locs.items()])

    





    


def test_recursion():
    (a20, c20, a21, c21, a22, c22, cp3, a31, c31, a32, c32, d) = make_mc_skill_apps()

    cp2 = SkillApp('cp', 'o2', ['c2'], short_name='cp2')

    def make_g(seq):
        print(seq)
        g, _ = _seq_to_initial_grammar(seq)
        # g = Grammar()
        # g.add_production("S", seq)
        return g
    
    seq = [c31, a31, c32, a32, c31, a31, d]
    print(find_recursion(make_g(seq)))

    seq = [c31, a31, a32, c31, a31, d]
    print(find_recursion(make_g(seq)))

    seq = [c31, a31, a32, a31, c31, d]
    print(find_recursion(make_g(seq)))

    seq = [d, c31, a31, a32, a31, d]
    print(find_recursion(make_g(seq)))

    seq = [c31, a31, a32, c32, a31, c31, d]
    print(find_recursion(make_g(seq)))

    seq = [c31, a31, a22, c22, a31, c31, d]
    print(find_recursion(make_g(seq)))

    seq = [c31, a31, a22, a31, c31, d]
    print(find_recursion(make_g(seq)))
 

    (ad0, ad1, ad2, dc0, dc1, dc2, sb0, sb1, sb2, cp0, cp1, cp2, d) = make_subtract_skill_apps()
    seq = [dc1, ad0, sb0, dc2, ad1, sb1, d]
    print(find_recursion(make_g(seq)))

    seq = [dc1, ad0, sb0, cp0, dc2, ad1, sb1, d]
    print(find_recursion(make_g(seq)))

    
def test_merge_rhs():
    (a20, c20, a21, c21, a22, c22, cp3, a31, c31, a32, c32, d) = make_mc_skill_apps()

    r2_0 = RHS([a20, c20])
    print(r2_0.rel_lits)
    print([x.skill_apps for x in r2_0.items])

    edits, align = find_edits(r2_0.items, [a21, c21], True)

    print("ALIGN", align)
    r2_01 = generalize_rhs(r2_0, [a21, c21], align)
    print(r2_01.rel_lits)
    print([x.skill_apps for x in r2_01.items])

def test_g_cost():
    (a20, c20, a21, c21, a22, c22, cp3, a31, c31, a32, c32, d) = make_mc_skill_apps()

    r2_0 = RHS([a20, c20])
    print(_check_rel_lit_matches(r2_0, [a21], np.array([0,-1])))

    # 
    print(_check_rel_lit_matches(r2_0, [a21, c21], np.array([0,-1])))

def test_rel_toks_on_gen():
    (a20, c20, a21, c21, a22, c22, cp3, a31, c31, a32, c32, d) = make_mc_skill_apps()

    g, rhs = _seq_to_initial_grammar([c20, a20, c21, a21, c22, a22, cp3, d])
    sequence = subseq = [a20, c20, c31, a31, a22, cp3, d]
    print("****", a21.skill_eq(a31), a31.skill_eq(a21))
    print("****", c22.skill_eq(a22), a22.skill_eq(c22))
    # g, rhs = _seq_to_initial_grammar([c21, a21, c22, a22, d])
    # sequence = subseq = [c31, a31, a22, cp3, d]

    overlap, alignment = calc_overlap(rhs.items, subseq)
    print(":", alignment)
    edits, alignment = find_edits(rhs.items, subseq, alignment, True)
    print(edits)
    # raise ValueError()
    # edits = [("unorder", 0, 2), ("replace", 2, 4, [a31, c31]), ("delete", 5, 6), ("insert", 6, 7, ["cp3"])]
    # alignment = np.array([1,2, 3,4, -1, ])

    lit_matches, gen_penl = _check_rel_lit_matches(rhs, subseq, alignment)
    rc = RHSChange(rhs, (0, len(subseq)), subseq, 
                    edits, alignment=alignment,
                    lit_matches=lit_matches, gen_penl=gen_penl)

    g, rhs = rc.apply(g)
    print(g)

    Srhs = g.prods['S'][0]
    c2a2_0 = g.prods['T'][0]
    c2a2_1 = g.prods['U'][0]
    c3a3 = g.prods['U'][1]

    print(Srhs, Srhs.rel_lits)
    print(c2a2_0, c2a2_0.rel_lits)
    print(c2a2_1, c2a2_1.rel_lits)
    print(c3a3, c3a3.rel_lits)
    


def make_frac_skill_apps():
    '''
    no0  opo  no1
    -          -
    do0       do1
    x "..."
    nc0  opc nc1   na
    -         -  = -
    dc0      dc1   da
    d
    '''
    no0 = IE("no0")
    do0 = IE("do0")
    no1 = IE("no1")
    do1 = IE("do1")
    opo = IE("opo")
    x = IE("x")
    nc0 = IE("nc0")
    dc0 = IE("dc0")
    nc1 = IE("nc1")
    dc1 = IE("dc1")
    opc = IE("opc")
    na = IE("na")
    da = IE("da")
    done = IE("done")

    do0.add_relation("a", no0)
    no0.add_relation("b", do0)
    do1.add_relation("a", no1)
    no1.add_relation("b", do1)

    no0.add_relation("r", opo)
    opo.add_relation("l", no0)
    opo.add_relation("r", no1)
    no1.add_relation("l", opo)

    x.add_relation("a", do0)
    do0.add_relation("b", x)

    nc0.add_relation("a", x)
    x.add_relation("b", nc0)

    dc0.add_relation("a", nc0)
    nc0.add_relation("b", dc0)
    dc1.add_relation("a", nc1)
    nc1.add_relation("b", dc1)

    nc0.add_relation("r", opc)
    opc.add_relation("l", nc0)
    opc.add_relation("r", nc1)
    nc1.add_relation("l", opc)

    da.add_relation("a", na)
    na.add_relation("b", da)

    # Multiply
    mno = newSA('m', na, [no0, no1], short_name='mno')
    mdo = newSA('m', da, [do0, do1], short_name='mdo')

    # Add Same
    ano = newSA('a', na, [no0, no1], short_name='ano')
    cpo = newSA('cp', da, [do1], short_name='cpo')

    # Add Different
    mc0 = newSA('m', na, [no0, do1], short_name='mc0')
    mc1 = newSA('m', da, [no1, do0], short_name='mc1')
    mdc = newSA('m', dc0, [do0, do1], short_name='mdc')
    cpc = newSA('cp', dc1, [dc0], short_name='cpo')
    anc = newSA('a', na, [nc0, nc1], short_name='anc')

    d = newSA('d', done, [], short_name='d')

    return (mno, mdo, ano, cpo, mc0, mc1, mdc, cpc, anc, d)


def test_fractions():
    (mno, mdo, ano, cpo, mc0, mc1, mdc, cpc, anc, d) = make_frac_skill_apps()

    seqs = [
        [mno, mdo, d],
        [ano, cpo, d],
        [cpo, mc0, mc1, mdc, cpc, anc, d],
    ]

    order = list(range(len(seqs)))
    
    print("ORDER", order)

    try: 
        g, rhs = _seq_to_initial_grammar(seqs[order[0]])
        # g = find_recursion(g)
        print(f'\ngrammar:\n{g}')
        for i in range(1,len(order)):
            seq = seqs[order[i]]
            g = generalize_from_seq(g, seq)    
            # g = find_recursion(g)
            pt = parse_w_changes(g, seq)
            assert pt.cost == 0.0
            print(f'\ngrammar:\n{g}')
    except Exception as e:
        print(g, i)
        print("ERROR ON ORDER", order)
        raise e



if(__name__ == "__main__"):
    # test_skill_app()
    # test_blarg()
    # test_find_edits()
    # test_find_rhs_change()
    # test_rhs_accept_cost()
    # test_accept_subseq()
    # test_get_best_changes()
    test_bottom_up_changes()
    # test_hole_changes()
    # test_top_down_parse()
    # test_target_domain_changes()
    # test_subtract()
    # test_recursion()
    # test_merge_rhs()
    # test_g_cost()
    # test_rel_toks_on_gen()
    # test_fractions()

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


'''
TODO list:

1) Recurse -> modify grammer
2) Ensure recursive grammars can parse/modify
'''

'''
Thinking thoughts:

-Symbols need to be parametrized
-Need a sort of top-down parse hole structure
-Needs to match up with bottom up
  - But how do we know what to connect with what?
-Where does parameterization start?
-Does it apply to the root symbol?


When I instantiate a new RHS the variables need to be apparent

TODOs:
1) Need to restore parse holes 
2) Need way of deciding when recursion is good idea
    -Having the user verify the recursion would be a good idea
    -A good way to do this would be to 

'''
