from pprint import pprint
from numbers import Number
from subprocess import Popen
from subprocess import PIPE
from subprocess import STDOUT
import re

from ilp.base_ilp import BaseILP


class Foil(BaseILP):

    def __init__(self, bind_objects=True, closed_world=True, max_tuples=100000):
        """
        The constructor. 
        """
        # Keywords used for constructing foil file (try not to conflict).
        self.name = "target_relation"
        self.example_type = "E"
        self.example_name = "curr_example"
        self.example_keyword = "e"
        self.continuous_type = "C"
        self.object_type = "O"

        # whether or not to use closed world assumption
        self.bind_objects = bind_objects
        self.closed_world = closed_world
        self.max_tuples = max_tuples

        # initialize all class variables
        self.initialize()

    def initialize(self):

        # the target relation types
        self.target_types = None

        # tuples for target relation
        self.pos = set()
        self.neg = set()

        # the possible attr value types
        # the val type "continuous" is reserved for numerical types
        self.val_types = {}
        self.val_types[self.continuous_type] = "continuous"

        # the aux relations and their type tuples
        # also the pos tuples for aux relations
        self.aux_relations = {}
        self.pos_aux_tuples = {}

        # String representation of learned rules
        self.rules = ""

    def resolve_type(self, val):
        if isinstance(val, Number):
            return self.continuous_type
        elif isinstance(val, bool):
            raise Exception("I don't know what to do with bool types")
        else:
            for tt in self.val_types:
                if val in self.val_types[tt]:
                    return tt

        # if the val cannot be resolve, then assume it is an object type and
        # add it to object types
        if self.object_type not in self.val_types:
            self.val_types[self.object_type] = set()
        self.val_types[self.object_type].add(val)
        return self.object_type
        
    def fit(self, T, X, y):

        # initialize all variables for learning
        self.initialize()

        # clean the input
        T = [self.clean(t) for t in T]
        X = [{self.clean(a): self.clean(x[a]) for a in x} for x in X]

        self.val_types[self.example_type] = set()
        for i, correct in enumerate(y):
            example_id = "%s%i" % (self.example_keyword, i) 
            self.val_types[self.example_type].add(example_id)

            # Get target relation tuples
            if correct == 1:
                self.pos.add((example_id,) + T[i])
            else:
                self.neg.add((example_id,) + T[i])

            # Get all types and values
            for rel in X[i]:
                if (not isinstance(X[i][rel], bool) and not
                      isinstance(X[i][rel], Number)):
                    # if not a boolean or number than add appropriate 
                    # attr-val type info
                    if isinstance(rel, str):
                        if rel not in self.val_types:
                            self.val_types[rel] = set()
                        self.val_types[rel].add(X[i][rel])
                    elif isinstance(rel, tuple):
                        if rel[0] not in self.val_types:
                            self.val_types[rel[0]] = set()
                        self.val_types[rel[0]].add(X[i][rel])
                    else:
                        raise Exception("attribute not string or tuple.")
            
            # resolve target relation arg types
            target_types = tuple()
            target_types = (self.example_type,)
            for val in T[i]:
                target_types += (self.resolve_type(val),)

            if self.target_types is None:
                self.target_types = target_types
            else:
                if self.target_types != target_types:
                    raise Exception("All tuples should have same types")

            # accumulate aux relation information
            for rel in X[i]:
                if isinstance(rel, str):
                    if rel not in self.aux_relations:
                        self.aux_relations[rel] = (self.example_type,
                                                 self.resolve_type(X[i][rel]))
                        self.pos_aux_tuples[rel] = set()
                    self.pos_aux_tuples[rel].add((example_id, X[i][rel]))
                elif isinstance(rel, tuple):
                    tup = (self.example_type,)
                    for ele in rel[1:]:
                        if isinstance(ele, tuple):
                            if ele in X[i]:
                                # the ele tuple head should be the type
                                tup += ele[0] 
                            else:
                                raise Exception("don't know what to do with tuple element. No value is defined for tuple in state.")
                        else:
                            tup += (self.resolve_type(ele),)
                    tup += (self.resolve_type(X[i][rel]),)

                    if rel[0] not in self.aux_relations:
                        self.aux_relations[rel[0]] = tup
                        self.pos_aux_tuples[rel[0]] = set()
                    self.pos_aux_tuples[rel[0]].add((example_id,) + rel[1:] + (
                                                    X[i][rel],))
                else:
                    raise Exception("attribute not string or tuple.")

        if self.closed_world is True:
            # only need to do sophisticated sampling when using closed world.
            num_neg_tuples = 1.0
            for t in self.target_types:
                if t == self.continuous_type:
                    raise Exception("Are negative tuples computed for continuous attributes?")
                num_neg_tuples *= len(self.val_types[t])

            num_pos_tuples = len(self.pos)
            max_neg_tuples = self.max_tuples - num_pos_tuples

            sample_size = min(10000, 10000 * ((max_neg_tuples-1) / num_neg_tuples))
            sample_size = int(sample_size) / 100

            print("NUM_NEG: ", num_neg_tuples)
            print("MAX_NEG: ", max_neg_tuples)
            print("SAMPLE: ", sample_size)

            # Create subprocess
            p = Popen(['ilp/FOIL6/foil6', '-m %i' % self.max_tuples, '-s %0.2f' %
                       sample_size], stdout=PIPE, stdin=PIPE, stderr=STDOUT)
        else:
            p = Popen(['ilp/FOIL6/foil6', '-m %i' % self.max_tuples],
                      stdout=PIPE, stdin=PIPE, stderr=STDOUT)

        data = ""

        for tt in self.val_types:
            if tt == self.continuous_type:
                data += "#type" + tt + ": " + self.val_types[tt] + ".\n"
            elif (tt == self.example_type or 
                  (not self.bind_objects and tt == self.object_type)):
                data += "#type" + tt + ": " + ",".join(["%s" % v for v in
                                                    self.val_types[tt]]) + ".\n"
            else:
                data += "#type" + tt + ": " + ",".join(["*%s" % v for v in
                                                    self.val_types[tt]]) + ".\n"
        data += "\n"

        # target relation header
        data += self.name + "(" + ", ".join(["type" + tt for tt in
                                             self.target_types]) + ") " 
        data += "".join(["#" for i in range(len(self.target_types))]) + "\n"

        # positive examples
        for t in self.pos:
            data += ", ".join(t) + "\n"

        if self.closed_world is False:
            # negative examples
            data += ";\n"
            for t in self.neg:
                data += ", ".join(t) + "\n"

        data += ".\n"

        # aux relations
        for rel in self.aux_relations:
            data += "*" + rel + "(" + ", ".join(['type' + tt for tt in
                                                 self.aux_relations[rel]]) + ")\n"

            # the positive examples of relation
            for t in self.pos_aux_tuples[rel]:
                data += ", ".join(t) + "\n"

            data += ".\n"

        print(data)

        # Send data to subprocess
        output = p.communicate(input=data.encode("utf-8"))[0]

        self.rules = ""

        found = False
            
        # Process result
        for line in output.decode().split("\n"):
            print(line)
            if re.search("^Training Set Size will exceed tuple limit:", line):
                raise Exception("Tuple limit exceeded, should never happen.")
            #match = re.search('^' + self.name + ".+:-", line)
            matches = re.findall("^" + self.name + "\(" +
                                 ",".join(["(?P<arg%i>[^ ,()]+)" % i for i in
                                           range(len(self.target_types))]) + "\)", line)
            if matches:
                found = True
                args = matches[0]
                self.rules += re.sub("_[0-9]+", "_", 
                                     re.sub("not\((?P<content>[^)]+)\)", 
                                            "not \g<content>", line[:-1]))
                
                first_arg = False
                if " :- " not in line:
                    self.rules += " :- "
                    first_arg = True

                ground_atoms = set()
                for tt in self.val_types:
                    if tt != self.continuous_type:
                        ground_atoms = ground_atoms.union(self.val_types[tt])

                for i, arg in enumerate(args):
                    if arg not in ground_atoms:
                        if not first_arg:
                            self.rules += ", "
                        else:
                            first_arg = False
                        self.rules += "type" + self.target_types[i] + "(" + arg + ")"

                self.rules += ".\n"
        
        #print("POS: ", len(self.pos))
        #print("NEG: ", len(self.neg))
        if not self.closed_world and not found and len(self.pos) > len(self.neg):
            self.rules += self.name + "(" + ",".join([chr(i + ord('A')) for i in range(len(self.target_types))]) + ") :- "
            self.rules += ", ".join(["type" + t + "(" + chr(i +
                                                                    ord('A')) +
                                     ")" for i,t in enumerate(self.target_types)])
            self.rules += ".\n"

        print("LEARNED RULES")
        print(self.rules)

    def get_matches(self, X):

        if self.rules == "":
            return []

        X = {self.clean(a): self.clean(X[a]) for a in X}

        val_types = {}

        # get the type info
        for rel in X:
            if (not isinstance(X[rel], bool) and not
                  isinstance(X[rel], Number)):
                # if not a boolean or number than add appropriate 
                # attr-val type info
                if isinstance(rel, str):
                    if rel not in val_types:
                        val_types[rel] = set()
                    val_types[rel].add(X[rel])
                elif isinstance(rel, tuple):
                    if rel[0] not in val_types:
                        val_types[rel[0]] = set()
                    val_types[rel[0]].add(X[rel])
                else:
                    raise Exception("attribute not string or tuple.")

        # add type info to logic program
        data = ""
        data += "type" + self.example_type + "(" + self.example_name + ").\n"
        for tt in val_types:
            for val in val_types[tt]:
                data += "type" + tt + "(" + val + ").\n"

        # add aux relations to logic program
        for rel in X:
            if isinstance(rel, str):
                data += rel + "(" + self.example_name + ", " + X[rel] + ").\n"
            elif isinstance(rel, tuple):
                data += rel[0] + "(" + self.example_name + ", " + ", ".join(rel[1:]) + ", " + X[rel] + ").\n"
                for ele in rel[1:]:
                    if ele not in X:
                        data += "type" + self.object_type + "(" + ele + ").\n"
            else:
                raise Exception("attribute not string or tuple.")

        data += self.rules
        data += "#show %s/%i." % (self.name, len(self.target_types))

        p = Popen(['clingo'], stdout=PIPE, stdin=PIPE, stderr=STDOUT)
        output = p.communicate(input=data.encode("utf-8"))[0]

        print(data)
        print(output.decode())

        matches = re.findall(self.name + "\(" + self.example_name + "," + ",".join(["(?P<arg%i>[^ ,()]+)" % i for i in range(len(self.target_types)-1)]) + "\)", output.decode())

        return [self.unclean(m) for m in matches]

    def unclean(self, x):
        if isinstance(x, tuple):
            return tuple(self.unclean(ele) for ele in x)
        elif isinstance(x, str):
            x = x.replace("multsign", "*")
            x = x.replace("equalsign", "=")
            x = x.replace("underscore", "_")
            x = x.replace("minussign", "-")
            x = x.replace("plussign", "+")
            x = x.replace("space", " ")
            x = x.replace("questionmark", "?")
            x = x[1:]
            if x == "nil":
                x == ""
            return x
        else:
            return x

    def clean(self, x):
        if isinstance(x, tuple):
            return tuple(self.clean(ele) for ele in x)
        elif isinstance(x, str):
            if x == "":
                x = "nil"
            x = x.replace("?", "questionmark")
            x = x.replace(" ", "space")
            x = x.replace("+", "plussign")
            x = x.replace("-", "minussign")
            x = x.replace("_", "underscore")
            x = x.replace("=", "equalsign")
            x = x.replace("*", "multsign")
            x = "a" + x
            return x
        else:
            return x
