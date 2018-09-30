import pickle
from random import random
# from pprint import pprint
# from random import shuffle
from itertools import product

import nltk
from nltk import ProbabilisticTree
from nltk.grammar import Production


class ViterbiParser(object):

    def __init__(self, grammar, trace=0):
        if not grammar.is_chomsky_normal_form():
            raise ValueError("Grammar must be in chomsky normal form.")

        self._grammar = grammar
        self._trace = trace

        # Build a custom index for more efficient parsing.
        self.unary_index = {}
        self.binary_index = {}
        for prod in grammar.productions():
            rhs = prod.rhs()
            if len(rhs) == 1:
                if rhs[0] not in self.unary_index:
                    self.unary_index[rhs[0]] = []
                self.unary_index[rhs[0]].append(prod)
            else:
                if rhs not in self.binary_index:
                    self.binary_index[rhs] = []
                self.binary_index[rhs].append(prod)

    def parse(self, tokens):

        tokens = list(tokens)
        self._grammar.check_coverage(tokens)

        # The most likely constituent table.  This table specifies the
        # most likely constituent for a given span and type.
        # Constituents can be either Trees or tokens.  For Trees,
        # the "type" is the Nonterminal for the tree's root node
        # value.  For Tokens, the "type" is the token's type.
        # The table is stored as a dictionary, since it is sparse.
        constituents = {}

        # Initialize the constituents dictionary with the words from
        # the text.
        if self._trace:
            print(('Inserting tokens into the most likely' +
                   ' constituents table...'))
        for index in range(len(tokens)):
            constituents[index, index+1] = {}
            token = tokens[index]
            if token not in self.unary_index:
                return
            for production in self.unary_index[token]:
                nt = production.lhs()
                symbol = production.lhs().symbol()
                ll = production.logprob()
                if (nt not in constituents[index, index+1] or
                        constituents[index, index+1][nt].logprob() < ll):
                    children = [token]
                    tree = ProbabilisticTree(symbol, children, logprob=ll)
                    constituents[index, index+1][nt] = tree
            if self._trace > 1:
                self._trace_lexical_insertion(token, index, len(tokens))

        # Consider each span of length 2, ..., n; Then for each way to
        # decompose this span, we check for rules that matching existing
        # sympols in the constituents table. Then we add matching rules.
        # If we cannot find a match, then we terminate early.
        # We start at 2 because we already parsed the unary rules.
        for length in range(2, len(tokens)+1):
            if self._trace:
                print(('Finding the most likely constituents' +
                       ' spanning %d text elements...' % length))
            for start in range(len(tokens)-length+1):
                span = (start, start+length)
                constituents[span] = {}
                # print("SPAN", span)
                for split in range(start+1, start+length):
                    left_idx = (start, split)
                    right_idx = (split, start+length)

                    left_symbols = []
                    if left_idx in constituents:
                        left_symbols = constituents[left_idx]
                    right_symbols = []
                    if right_idx in constituents:
                        right_symbols = constituents[right_idx]

                    # print("left", left_idx, left_symbols)
                    # print("right", right_idx, right_symbols)

                    for rhs in product(left_symbols, right_symbols):
                        # print(rhs)
                        if rhs not in self.binary_index:
                            continue
                        for production in self.binary_index[rhs]:
                            nt = production.lhs()
                            symbol = production.lhs().symbol()
                            children = [constituents[left_idx][rhs[0]],
                                        constituents[right_idx][rhs[1]]]
                            ll = (production.logprob() +
                                  constituents[left_idx][rhs[0]].logprob() +
                                  constituents[right_idx][rhs[1]].logprob())
                            if (nt not in constituents[span] or
                                    constituents[span][nt].logprob() < ll):
                                tree = ProbabilisticTree(symbol, children,
                                                         logprob=ll)
                                constituents[span][nt] = tree

                # print()

        # Return the tree that spans the entire text & have the right cat
        tree = None
        span = (0, len(tokens))
        # print(constituents[span])
        if self._grammar.start() in constituents[span]:
            tree = constituents[span][self._grammar.start()]

        if tree is not None:
            yield tree


class pCFG_Grammar(object):

    def __init__(self):
        self.nt_count = 0
        self.grammar = None
        self.start = nltk.Nonterminal('S')

    def gen_nt(self):
        self.nt_count += 1
        return nltk.Nonterminal("NT%i" % self.nt_count)

    def parse(self, sentences):
        if self.grammar is None:
            raise Exception("No Grammar, call learn_grammar first.")

        parser = ViterbiParser(self.grammar)

        for sent in sents:
            found = False
            for tree in parser.parse(sent):
                found = True
                yield tree
            if not found:
                raise Exception("Sentence not parsable")

    def learn_grammar(self, sentences):
        print()
        print("Inducing Structure...")
        self.induce_structure(sentences)
        print("done.")

        print()
        print("Initial Grammar:")
        print(self.grammar)

        print()
        print("Inducing Weights...")
        self.induce_weights(sentences)
        print("done.")

    def induce_weights(self, sentences):
        if self.grammar is None:
            raise Exception("Need to call induce_structure first")

        sentences = [[c for c in s] for s in sentences]

        log_prob_last = 0
        log_prob_curr = float('inf')

        while abs(log_prob_last - log_prob_curr) > 0.0001:
            log_prob_last = log_prob_curr

            parser = ViterbiParser(self.grammar)

            productions = []
            log_prob_curr = 0
            for i, sent in enumerate(sentences):
                print("parsing sentence %i of %i" % (i, len(sentences)))
                found = False
                for tree in parser.parse(sent):
                    found = True
                    log_prob_curr += tree.logprob()
                    productions += tree.productions()
                if not found:
                    print(sent)
                    raise Exception("Unable to parse sentence")

            # print("last log prog", log_prob_last)
            print("curr log prob", log_prob_curr)

            self.grammar = nltk.induce_pcfg(self.start, productions)

    def induce_structure(self, sentences):

        sentences = [[c for c in s] for s in sentences]

        start_symbols = set()
        productions = []
        prod_table = {}

        # group all digits together
        digit_terminals = set([str(i) for i in range(10)])

        # unary rules
        terminals = set()
        for s in sentences:
            terminals.update(s)
        for t in terminals:
            if t in digit_terminals:
                nt = nltk.Nonterminal("Digit")
            else:
                nt = nltk.Nonterminal("Unary%s" % self.gen_nt())
            p = Production(nt, [t])
            productions.append(p)
            prod_table[tuple(p.rhs())] = p.lhs()

        sentences = self.apply_unary_prod(sentences, prod_table)

        while len(sentences) > 0:
            if self.has_recursion(sentences):
                p = self.generate_recursive_prod(sentences)
            else:
                p = self.generate_most_frequent_prod(sentences)

            productions.append(p)
            prod_table[tuple(p.rhs())] = p.lhs()

            sentences = self.update_with_prod(sentences, prod_table)

            new_sentences = []
            for s in sentences:
                if len(s) == 1:
                    start_symbols.add(s[0])
                else:
                    new_sentences.append(s)

            sentences = new_sentences

        # generate the start productions
        for symbol in start_symbols:
            for p in productions:
                if p.lhs() == symbol:
                    productions.append(Production(self.start, p.rhs()))

        self.grammar = nltk.induce_pcfg(self.start, productions)

    def generate_most_frequent_prod(self, sentences):
        pairs = {}

        for s in sentences:
            for i in range(len(s)-1):
                pair = (s[i], s[i+1])
                if pair not in pairs:
                    pairs[pair] = 0
                pairs[pair] += 1

        pairs = [(pairs[p], random(), p) for p in pairs]
        pairs.sort(reverse=True)
        count, _, pair = pairs[0]
        nt = self.gen_nt()

        return Production(nt, list(pair))

    def generate_recursive_prod(self, sentences):
        pairs = {}
        for s in sentences:
            for i in range(len(s) - 1):
                if (s[i] == s[i+1] and
                        isinstance(s[i], nltk.Nonterminal)):
                    pair = (s[i], s[i+1])
                    if pair not in pairs:
                        pairs[pair] = 0
                    pairs[pair] += 1

        pairs = [(pairs[p], random(), p) for p in pairs]
        pairs.sort(reverse=True)
        count, _, pair = pairs[0]

        return Production(pair[0], list(pair))

    def apply_unary_prod(self, sentences, prod_table):
        for s in sentences:
            i = 0
            while i < len(s):
                if (s[i],) in prod_table:
                    s[i:i+1] = [prod_table[(s[i],)]]
                else:
                    i += 1
        return sentences

    def update_with_prod(self, sentences, prod_table):
        for s in sentences:
            # print(s)
            i = 0
            while i < len(s) and len(s) > 1:
                pair = tuple(s[i:i+2])
                if pair in prod_table:
                    s[i:i+2] = [prod_table[pair]]
                    i -= 1
                else:
                    i += 1
        return sentences

    def has_recursion(self, sentences):
        for s in sentences:
            for i in range(len(s) - 1):
                if (s[i] == s[i+1] and
                        isinstance(s[i], nltk.Nonterminal)):
                    return True
        return False


def get_what_learner(name):
    return WHAT_LEARNERS[name.lower.replace(' ', '').replace('_', '')]

WHAT_LEARNERS = {
    'viterbiparser':ViterbiParser,
    'pcfggrammar':pCFG_Grammar
}


if __name__ == "__main__":

    sentences = set()

    fdir = "/Users/cmaclell/Downloads/ds293_tx_2017_0424_102558/"
    fname = "ds293_tx_All_Data_876_2015_0804_091957.txt"

    with open(fdir + fname) as fin:
        key = None
        for line in fin:
            if key is None:
                key = {v: i for i, v in enumerate(line.split('\t'))}
                continue
            line = line.split('\t')
            eq = line[key['Step Name']]

            if '=' not in eq:
                continue

            eq = eq.split(' = ')
            # equation_strings.append(eq[0])
            sentences.add(eq[0].lower())
            # equation_strings.append(eq[1])
            sentences.add(eq[1].lower())

    fdir = "/Users/cmaclell/Dropbox/projects/simstudent/Authoring" \
           "Tools/java/Projects/articleSelectionTutor/massproduct" \
           "ion-templates/"
    fname = "article_sentences.txt"

    with open(fdir + fname, encoding="ISO-8859-1") as fin:
        key = None
        for line in fin:
            line = line.strip()
            print('adding', line.lower())
            sentences.add(line.lower())

    # sentences = list(sentences)
    # shuffle(sentences)
    # sentences = sentences[0:30]
    # pprint(sentences)

    # equation_strings = ["3x + 2", "-4x", "4x", "700", "-5", "-100"]

    sents = [[c for c in s] for s in sentences]
    # pprint(sents)

    gl = pCFG_Grammar()
    gl.learn_grammar(sents)
    # print(gl.grammar)

    print("PICKLING...")
    pickle.dump(gl.grammar, open("grammar.pickle", "wb"))
    print("Done.")

    # for tree in gl.parse(sents):
    #     print(tree)
    #     print(tree.logprob())
