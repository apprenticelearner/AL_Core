from nltk.tree import ProbabilisticTree
from nltk.grammar import induce_pcfg
from nltk.grammar import Production
from nltk import Nonterminal
import heapq

import pickle

def average(l):
    """
    Returns the average of a list of numbers.
    """
    return sum(l) / len(l)

class NeuralNetLearnedFeatures():

    def __init__(self):

        with open("/Users/cmaclell/Downloads/embeddings.csv", "r") as embin:
            embeddings = [[float(v) for v in line.strip().split(",")] for line in embin]

        with open("/Users/cmaclell/Downloads/data_307.txt", "r") as fin:
            self.nn_features = {line.split('\t')[0].lower().replace('"',""):
                                embeddings[i] for i,line in enumerate(fin)}

        self.nn_features['a'] = {}
        self.nn_features['an'] = {}
        self.nn_features['the'] = {}

    def get_features(self, attr_name, sequence):
        sent = "".join(sequence)
        if sent in self.nn_features:
            emb = self.nn_features["".join(sequence)]
            return {("f%i" % i, attr_name): v for i,v in enumerate(emb)}

        print("Unable to find: %s" % sent)
        print(self.nn_features.keys())
        return {}

class GrammarLearnedFeatures():

    def __init__(self):

        with open("/Users/cmaclell/Downloads/grammar_dict.pickle", "rb") as fin:
            self.features = pickle.load(fin)

    def get_features(self, attr_name, sequence):
        sent = "".join(sequence)
        if sent in self.features:
            f = self.features["".join(sequence)]
            return {("NT-count (%s)" % attr, attr_name):f[attr] for attr in f}

        print("Unable to find: %s" % sent)
        print(self.nn_features.keys())
        return {}

class GrammarLearner():

    def __init__(self):
        self.sequences = []
        self.pcfg_grammar = None
        self.gensym_counter = 0

    def nonterminal_gensym(self):
        """
        Returns a unique nonterminal symbol.
        """
        self.gensym_counter += 1
        return Nonterminal("NT%i" % self.gensym_counter)

    def ifit(self, sequence):
        """
        Adds the sequence to the previous training data
        and retrains.
        """
        self.sequences.append(sequence)
        return self.fit(self.sequences)

    def fit(self, sequences):
        """
        Learn a grammar given a set of sequences
        """
        self.sequences = sequences
        self.greedy_structure_hypothesizer(self.sequences)
        self.viterbi_training(self.sequences)
        return self.pcfg_grammar

    def generate_most_likely_grounding(self, nt):
        head_index = {}

        for p in self.pcfg_grammar.productions():
            if str(p.lhs()) not in head_index:
                head_index[str(p.lhs())] = []
            head_index[str(p.lhs())].append(p)

        initial = (0, [nt], "")
        h = []

        heapq.heappush(h, initial)

        while len(h) > 0:
            lp, nts, s = heapq.heappop(h)

            if len(nts) == 0:
                return s

            nt = str(nts[0])

            for p in head_index[str(nt)]:
                if len(p.rhs()) == 1:
                    new = (lp - p.logprob(), nts[1:], s + p.rhs()[0])
                else:
                    new = (lp - p.logprob(), list(p.rhs()) + nts[1:], s)
                heapq.heappush(h, new)

        return None

    def __str__(self):
        unary = []
        binary = []

        for p in self.pcfg_grammar.productions():
            if len(p.rhs()) == 1:
                unary.append((p.logprob(), p))
            else:
                binary.append((p.logprob(), p))

        unary.sort()
        binary.sort()

        s = ""
        for lb, p in unary:
            s += str(p) + "\n"
        for lp, p in binary:
            s += str(p) + "\n"

        return s

    def get_features(self, attr_name, sequence):
        """
        Returns a set of new grammar based features to describe the 
        sequence.
        """
        tree = self.parse(sequence)
        if tree is None:
            return {}

        features = {}
        nodes = [tree]
        while len(nodes) > 0:
            n = nodes.pop()
            if attr_name is None:
                f = n.label()
            else:
                f = (n.label(), attr_name)

            if f not in features:
                features[f] = 0
            features[f] += 1

            if len(n) == 2:
                nodes.append(n[0])
                nodes.append(n[1])

        return features

    def get_tree_features(self, attr_name, sequence):
        """
        Returns a set of new grammar based features to describe the 
        sequence.
        """
        tree = self.parse(sequence)
        if tree is None:
            return {}

        features = {}
        nodes = [(attr_name, tree)]
        while len(nodes) > 0:
            path, n = nodes.pop()
            features[path] = n.label()
            
            if len(n) == 2:
                nodes.append((('left', path), n[0]))
                nodes.append((('right', path), n[1]))

        return features

    def generate_terminal_productions(self, sequences):
        """
        Returns a set of rules for each of the terminal symbols. 
        """
        terminals = {symbol for seq in sequences for symbol in seq}
        productions = {Production(self.nonterminal_gensym(), [t]) for t in
                       terminals}
        return productions

    def update_sequences(self, sequences, productions):
        unary_index = {}
        binary_index = {}
        for p in productions:
            if len(p.rhs()) == 1:
                unary_index[p.rhs()[0]] = p
            elif len(p.rhs()) == 2:
                binary_index[tuple(p.rhs())] = p
            else:
                raise Exception("productions cannot have more than 2 symbols on rhs")

        sequences = [[unary_index[ele].lhs() if ele in unary_index else ele 
                      for ele in s] for s in sequences]
        new_sequences = []

        for s in sequences:
            start = 0
            while start < len(s)-1:
                curr = tuple(s[start:start+2])
                if curr in binary_index:
                    s = s[:start] + [binary_index[curr].lhs()] + s[start+2:]
                    if start > 1:
                        start -= 1
                else:
                    start += 1

            new_sequences.append(s)

        return new_sequences

    def get_recursive_production(self, sequences):
        """
        Finds a symbol that is repeated in one of the sequences and returns it.
        If no pair is repeated, then returns None.
        """
        count_threshold = 1
        avg_len_threshold = 2 

        patterns = {}

        for s in sequences:
            left = None
            last = None
            length = 0

            for sym in s + [None]:
                if sym != last:
                    if last not in patterns:
                        patterns[last] = {}

                    if left is not None:
                        if ('right', left) not in patterns[last]:
                            patterns[last][('right', left)] = {'count':0,
                                                               'lengths':[]}
                        patterns[last][('right', left)]['count'] += 1
                        patterns[last][('right', left)]['lengths'].append(length)

                    if sym is not None:
                        if ('left', sym) not in patterns[last]:
                            patterns[last][('left', sym)] = {'count':0,
                                                               'lengths':[]}
                        patterns[last][('left', sym)]['count'] += 1
                        patterns[last][('left', sym)]['lengths'].append(length)

                    left = last
                    last = sym
                    length = 1
                else:
                    length += 1

        prods = [(patterns[rs][(d,ts)]['count'], Production(ts, [rs, ts])) 
                 if d=="left" else 
                 (patterns[rs][(d,ts)]['count'], Production(ts, [ts, rs]))
                 for rs in patterns for d, ts in patterns[rs] if
                 patterns[rs][(d,ts)]['count'] >= count_threshold and 
                 average(patterns[rs][(d,ts)]['lengths']) >= avg_len_threshold]

        if len(prods) > 0:
            prods.sort(reverse=True)
            return prods[0][1]

        return None

    def get_most_frequent_pair(self, sequences):
        counts = {}
        for s in sequences:
            for start in range(len(s)-1):
                pair = s[start], s[start+1]
                if pair not in counts:
                    counts[pair] = 0
                counts[pair] += 1

        pairs = [(counts[p], i, p) for i,p in enumerate(counts)]
        pairs.sort(reverse=True)
        return pairs[0][2]

    def greedy_structure_hypothesizer(self, sequences):
        productions = self.generate_terminal_productions(sequences)
        parsed_seqs = self.update_sequences(sequences, productions)
        start_symbols = set()

        while len(parsed_seqs) > 0:
            new_prod = self.get_recursive_production(parsed_seqs)
            if new_prod is None:
                most_frequent_pair = self.get_most_frequent_pair(parsed_seqs)
                new_prod = Production(self.nonterminal_gensym(), most_frequent_pair)

            productions.add(new_prod)
            parsed_seqs = self.update_sequences(parsed_seqs, productions)
            start_symbols = start_symbols.union({s[0] for s in parsed_seqs if
                                                 len(s) == 1})
            parsed_seqs = [s for s in parsed_seqs if len(s) > 1]

        # TODO not sure this is the right way to handle start symbol
        # introduction, but hey it'll work for now.
        #productions = [Production(Nonterminal('S'), 
        #                          [Nonterminal('S') 
        #                           if nt in start_symbols else nt 
        #                           for nt in p.rhs()]) 
        #               if p.lhs() in start_symbols else 
        #               Production(p.lhs(),[Nonterminal('S') 
        #                           if nt in start_symbols else nt 
        #                           for nt in p.rhs()])  
        #               for p in productions]

        self.pcfg_grammar = induce_pcfg(Nonterminal("S"), productions)

    def parse(self, s):
        parser = CustomViterbiParser(self.pcfg_grammar)
        return parser.parse(s)

    def viterbi_training(self, sequences):
        """
        Update self.pcfg probabilities using viterbi training
        """
        last_ll = 0
        ll = float('-inf')
        tol = 0.000001

        while abs(last_ll - ll) > tol:
            last_ll = ll
            parser = CustomViterbiParser(self.pcfg_grammar)
            productions = []
            ll = 0
            for s in sequences:
                tree = parser.parse(s)
                if not tree:
                    raise Exception("Not parsable")
                ll += tree.logprob()
                productions += tree.productions()

            self.pcfg_grammar = induce_pcfg(self.pcfg_grammar.start(),
                                            productions)
            
class CustomViterbiParser:

    def __init__(self, grammar):
        # organize grammar productions in hashmaps for 
        # fast lookup
        self.unary_prods = {}
        self.binary_prods = {}
        for p in grammar.productions():
            if len(p.rhs()) == 1:
                if p.rhs()[0] not in self.unary_prods:
                    self.unary_prods[p.rhs()[0]] = []
                self.unary_prods[p.rhs()[0]].append(p)
            elif len(p.rhs()) == 2:
                if tuple(p.rhs()) not in self.binary_prods:
                    self.binary_prods[tuple(p.rhs())] = []
                self.binary_prods[tuple(p.rhs())].append(p)
            else:
                raise Exception("Grammar rules cannot have more than two symbols on RHS.")

    def parse(self, s):
        n = len(s)
        # inside chart is a tuple with nt:(left_loc, right_loc, prob)
        chart = {i:{j:{} for j in range(i,n)} for i in range(n)}

        # apply unary productions
        for i in range(n):
            token = s[i]
            for p in self.unary_prods[token]:
                if (p.lhs() not in chart[i][i] or 
                    p.logprob() > chart[i][i][p.lhs()][2]):
                    # no left and right expansions at the leaves so use None
                    chart[i][i][p.lhs()] = (None, None, p.logprob())

        # build up chart with binary productions
        for span in range(1,n):
            for left in range(n-span):
                right = left+span

                for split in range(left, right):
                    for left_nt in chart[left][split]:
                        for right_nt in chart[split+1][right]:
                            rhs = (left_nt, right_nt)

                            if rhs not in self.binary_prods:
                                continue

                            for p in self.binary_prods[rhs]:
                                prob = (chart[left][split][left_nt][2] +
                                        chart[split+1][right][right_nt][2] +
                                        p.logprob())

                                if (p.lhs() not in chart[left][right] or
                                    prob > chart[left][right][p.lhs()][2]):
                                    chart[left][right][p.lhs()] = ((left_nt,
                                                                    left,
                                                                    split),
                                                                   (right_nt,
                                                                    split+1,
                                                                    right),
                                                                   prob)

        top_nts = [(chart[0][n-1][nt][2], nt) for nt in chart[0][n-1]]

        if len(top_nts) > 0:
            top_nts.sort(reverse=True)
            return self.build_tree(s, chart, top_nts[0][1], 0, n-1)

        print("UNABLE TO PARSE")
        print(s)
        return None

    def build_tree(self, seq, chart, nt, left, right):
        """
        Given a sequence and a chart, builds the nltk Probabilistic Tree
        object.
        """
        if left == right:
            return ProbabilisticTree(nt.symbol(), [seq[left]],
                                     logprob=chart[left][right][nt][2])
        
        left_tree = self.build_tree(seq, chart, chart[left][right][nt][0][0],
                                    chart[left][right][nt][0][1], 
                                    chart[left][right][nt][0][2])
        right_tree = self.build_tree(seq, chart, chart[left][right][nt][1][0],
                                    chart[left][right][nt][1][1], 
                                    chart[left][right][nt][1][2])

        return ProbabilisticTree(nt.symbol(), [left_tree, right_tree],
                                 logprob=chart[left][right][nt][2]) 

what_learners = {'grammar': GrammarLearner}


if __name__ == "__main__":
    #from nltk.corpus import treebank
    import re

    gl = GrammarLearner()
    with open("/Users/cmaclell/Projects/simstudent/AuthoringTools/java/Projects/articleSelectionTutor/massproduction-templates/article_sentences.csv") as fin:
        lines = [line for line in fin]
    
    print(len(lines))

    words = [[c for c in w] for line in lines 
             for w in re.split(r'[ ,;.\?\-"]+', line.strip().lower()) 
             if w != ""]
    test_seqs = words + [[c for c in line.strip().lower()] for line in lines]


    #gl.fit(test_seqs)
    #with open("/Users/cmaclell/Downloads/article_grammar.pickle", 'wb') as fout:
    #    pickle.dump(gl, fout)
    #print("PICKLED")

    with open("/Users/cmaclell/Downloads/article_grammar.pickle", 'rb') as f:
        gl = pickle.load(f)

    #print(gl)

    #for w in test_seqs:
    #    fm = gl.get_features(None, w)
    #    for a in fm:
    #        print(a, gl.generate_most_likely_grounding(a))
    #    print({gl.generate_most_likely_grounding(a): fm[a] for a in fm})


    features = {}
    for line in lines:
        fm = gl.get_features(None, [c for c in line.strip().lower()])
        #for a in fm:
        #    print(a, gl.generate_most_likely_grounding(a))
        features[line.strip().lower()] = {gl.generate_most_likely_grounding(a):fm[a] for a in fm}

    with open("/Users/cmaclell/Downloads/grammar_dict.pickle", "wb") as fout:
        pickle.dump(features, fout)

    print(features)
#

    #gl = GrammarLearner()

    #with open("/Users/cmaclell/Projects/simstudent/AuthoringTools/java/Projects/articleSelectionTutor/massproduction-templates/article_sentences.txt") as fin:
    #    lines = [line for line in fin]

    #words = [[c for c in w] for line in lines for w in re.split(r'[ ,;.\?\-"]+', line.strip().lower()) if
    #         w != ""]
    #test_seqs = [[c for c in line.strip().lower()] for line in lines] #+ [words]

    ##test_seqs = [[c for c in " ".join([s.lower() for s in t])] for fileid in
    ##             treebank.fileids()[:2] for t in treebank.sents(fileid)]
    #print(len(test_seqs))
    #print(sum([len(s) for s in test_seqs])/len(test_seqs))

    #print(test_seqs)

    #gl.fit([['b','a','a','a']])
    #print(gl.parse(['b','a','a','a']))
    #gl.fit(test_seqs[:20])
    #print(test_seqs[19])
    #print(gl.parse(test_seqs[19]))
    #print(gl.pcfg_grammar)

    #import pickle

    #with open("/Users/cmaclell/Downloads/article_grammar.pickle", 'wb') as fout:
    #    pickle.dump(gl, fout)

    #print(gl.get_features(("value", '?foa0'), test_seqs[19]))



    #nnf = NeuralNetLearnedFeatures()
    #print(nnf.get_features('?foa0', lines[10].strip().replace('"',"")))
