from pprint import pprint
from numbers import Number
from subprocess import Popen
from subprocess import PIPE
from subprocess import STDOUT
import re

import pexpect

from ilp.base_ilp import BaseILP
from agents.utils import gen_varnames

class FoilClassifier(BaseILP):

    def __init__(self, closed_world=False, max_tuples=100000,
                 name="target_relation"):
        """
        The constructor for the Foil classifier. Closed world specifies whether
        only the positive examples should be used, and the complete set of
        implicit negatives is used. Max tuples specifies the maximum number of
        implicit negatives to use. If the number of implicit negatives exceeds
        the provided value then sampling will be used to randomly select a
        subset of negative examples.
        """
        # Keywords used for constructing foil file (try not to conflict).
        self.name = name

        # The prefix for the type names 
        self.type_prefix = "T"

        # whether or not to use closed world assumption
        self.closed_world = closed_world

        # the maximum number of tuples for foil
        self.max_tuples = max_tuples

        # initialize all class variables
        self.initialize()

    def initialize(self):
        """
        Initialize all of the variables used by the classifier.
        """
        # tuples for target relation
        self.pos = list()
        self.neg = list()

        # The values for each type.
        self.type_values = {}

        # the type identifiers for the target relation
        self.target_attrs = []

        # a mapping from target attrs to type shortnames
        self.attr_type = {}

        # Prolog string representation of learned rules
        self.rules = set()

    def __repr__(self):
        key = {v: self.target_attrs[i] 
               for i,v in enumerate(gen_varnames(end=len(self.target_attrs)))}
        #key = {chr(i + ord('A')): self.target_attrs[i] 
        #       for i in range(len(self.target_attrs))}

        for v in key:
            key[v] = self.unclean(key[v])

        rules = set()
        if len(self.rules) == 0:
            rules.add("FALSE")

        for r in self.rules:
            new = []
            if ":-" in r:
                new_r = r.split(" :- ")[1]
                new_r = new_r.split(", ")
                new = []
                for sub in new_r:
                    if "<>" in sub:
                        sub = "<>".join([self.unclean(e) if e not in key else key[e] 
                                         for e in sub.split("<>")])
                        new.append(sub)
                    else:
                        new.append(sub)
                        print("MAKE SURE THIS IS BEING HANDLED RIGHT")
                        raise Exception("What is this?: %s" % sub)
                r = r.split(" :- ")[0]

            sub = r[16:-1].split(",")
            for i,v in enumerate(gen_varnames(end=len(self.target_attrs))):
            #for i in range(len(self.target_attrs)):
                if sub[i] != v:
                    arg = self.unclean(self.target_attrs[i])
                    if sub[i] in key:
                        val = key[sub[i]]
                    else:
                        val = self.unclean(sub[i])
                    new.append("%s==%s" % (arg, val))
            # assume that foa0 must be nil
            new.append(("('value', '?foa0')==''"))
            new_r = " and ".join(new)
            rules.add(new_r)

        rules = ["(" + r + ")" for r in rules]
        rules = " or ".join(rules)

        return repr(rules)

    def fit(self, X, y):
        """
        Fit the FOIL classifier (i.e., learn a set of rules to cover the
        positive examples but not the negative examples. 

        X is a list of dictionaries of attribute values.
        y is a list of 0 or 1 class labels. 
        """
        # initialize all variables for learning
        self.initialize()

        # clean the input
        X = [{self.clean(a): self.clean(x[a]) for a in x} for x in X]

        # Collect attribute and value information
        for x in X:
            for attr in x:
                if attr not in self.type_values:
                    self.type_values[attr] = set()
                if isinstance(x[attr], Number):
                    self.type_values[attr].add('continuous')
                    if len(self.type_values[attr]) > 1:
                        raise Exception("Cannot mix numeric (continuous) and nominal types")
                else:
                    self.type_values[attr].add("*" + x[attr])

        # Build attr-type mappings
        self.target_attrs = list(set(self.type_values.keys()))
        self.attr_type = {a: self.type_prefix + str(i)
                           for i,a in enumerate(self.target_attrs)}

        # Build tuples
        for i, correct in enumerate(y):
            t = tuple([str(X[i][attr]) if attr in X[i] else '?' for attr in
                       self.target_attrs])
            if correct == 1:
                self.pos.append(t)
            else:
                self.neg.append(t)

        # construct foil data
        data = ""

        for tt in self.type_values:
            data += "#" + self.attr_type[tt] + ": "
            data += ", ".join(self.type_values[tt]) + ".\n"
        data += "\n"

        # target relation header
        data += self.name + "(" + ", ".join([self.attr_type[a] for a in
                                             self.target_attrs]) + ")\n" 

        if self.closed_world is False:
            # include all positive and negative examples
            for t in self.pos:
                data += ", ".join(t) + "\n"
            data += ";\n"
            for t in self.neg:
                data += ", ".join(t) + "\n"

        else:
            # only include positive examples
            for t in set(self.pos):
                data += ", ".join(t) + "\n"

        data += ".\n"

        #print(data)

        #print([self.unclean(e) for e in self.target_attrs])
        #for t in self.pos:
        #    print("+ : ", t)
        #for t in self.neg:
        #    print("- : ", t)

        if self.closed_world is True:
            # only need to do sophisticated sampling when using closed world.
            num_neg_tuples = 1.0
            for t in self.target_attrs:
                vals = set()
                for v in self.type_values[t]:
                    if v == "continuous":
                        raise Exception("Cannot use closed world with continuous types.")
                    vals.add(v)
                num_neg_tuples *= len(vals)

            num_pos_tuples = len(self.pos)
            max_neg_tuples = self.max_tuples - num_pos_tuples

            sample_size = min(10000, 10000 * ((max_neg_tuples-1) / num_neg_tuples))
            sample_size = int(sample_size) / 100

            #print("NUM_NEG: ", num_neg_tuples)
            #print("MAX_NEG: ", max_neg_tuples)
            #print("SAMPLE: ", sample_size)

            p = Popen(['ilp/FOIL6/foil6', '-m %i' % self.max_tuples, '-s %0.2f' %
                       sample_size], stdout=PIPE, stdin=PIPE, stderr=STDOUT)
        else:
            p = Popen(['ilp/FOIL6/foil6', '-m %i' % self.max_tuples],
                      stdout=PIPE, stdin=PIPE, stderr=STDOUT)

        # Send data to subprocess
        output = p.communicate(input=data.encode("utf-8"))[0]

        self.rules = set()

        found = False
            
        # Process result
        for line in output.decode().split("\n"):
            #print(line)
            if re.search("^Training Set Size will exceed tuple limit:", line):
                raise Exception("Tuple limit exceeded, should never happen.")
            matches = re.findall("^" + self.name + "\(" +
                                 ",".join(["(?P<arg%i>[^ ,()]+)" % i for i in
                                           range(len(self.target_attrs))]) +
                                 "\)", line)
            if matches:
                found = True
                self.rules.add(line[:-1])

        
        #print("POS: ", len(self.pos))
        #print("NEG: ", len(self.neg))
        if (self.closed_world is False and 
            not found and len(self.pos) > len(self.neg)):
            rule = self.name 
            rule += "(" + ",".join([v for v in
                                    gen_varnames(end=len(self.target_attrs))]) + ")"
            #rule += "(" + ",".join([chr(i + ord('A')) for i in
            #                        range(len(self.target_attrs))]) + ")"
            self.rules.add(rule)

        #print("LEARNED RULES")
        #print(self.rules)

    def predict(self, X):

        if len(self.rules) == 0:
            return [0 for x in X]

        # Construct prolog data
        data = ""

        for rule in self.rules:
            # replace with appropriate prolog operators.
            rule = re.sub(" (?P<first>\w+)<>(?P<second>\w+)", 
                          " dif(\g<first>, \g<second>)", rule)
            rule = rule.replace("<=", "=<")
            data += rule + ".\n"

        #print(data)

        outfile = open("foil_binding_lp.pl", 'w')
        outfile.write(data)
        outfile.close()

        #p = PopenSpawn('swipl -q -s foil_binding_lp.pl', encoding="utf-8",
        #               maxread=100000)
        p = pexpect.spawn('swipl -q -s foil_binding_lp.pl', timeout=None,
                          echo=False, encoding="utf-8", maxread=1000000,
                          searchwindowsize=None)
        p.expect("\?- ")

        X = [{self.clean(a): self.clean(x[a]) for a in x} for x in X]

        yh = []
        for x in X:
            e = tuple([str(x[attr]) if attr in x else '_' for attr in
                       self.target_attrs])

            #print(self.name + "(" + ", ".join(e) + ").")
            p.sendline(self.name + "(" + ", ".join(e) + ").")
            resp = p.expect(["false.", "true."])

            if resp == 0:
                yh.append(0)
            elif resp == 1:
                yh.append(1)

        return yh

    def unclean(self, x):
        if isinstance(x, tuple):
            return tuple(self.unclean(ele) for ele in x)
        elif isinstance(x, str):
            x = x.replace("rightparen", ')')
            x = x.replace("leftparen", '(')
            x = x.replace("multsign", "*")
            x = x.replace("equalsign", "=")
            x = x.replace("underscore", "_")
            x = x.replace("minussign", "-")
            x = x.replace("plussign", "+")
            x = x.replace("space", " ")
            x = x.replace("questionmark", "?")
            x = x.replace("period", ".")
            x = x.replace("apostrophe", "'")
            x = x.replace("quote", '"')
            x = x.replace("backslash", "\\")
            x = x[1:]
            if x == "nil":
                x == ""
            return x
        else:
            return x

    def clean(self, x):
        if isinstance(x, tuple):
            return tuple(self.clean(ele) for ele in x)
        elif isinstance(x, str) or isinstance(x, bool):
            if isinstance(x, bool):
                x = str(x)
            if x == "":
                x = "nil"
            x = x.replace("\\", "backslash")
            x = x.replace('"', "quote")
            x = x.replace("'", "apostrophe")
            x = x.replace(".", "period")
            x = x.replace("?", "questionmark")
            x = x.replace(" ", "space")
            x = x.replace("+", "plussign")
            x = x.replace("-", "minussign")
            x = x.replace("_", "underscore")
            x = x.replace("=", "equalsign")
            x = x.replace("*", "multsign")
            x = x.replace("(", "leftparen")
            x = x.replace(")", "rightparen")
            x = "a" + x
            return x
        else:
            return x
