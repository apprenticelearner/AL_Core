from subprocess import Popen
from subprocess import PIPE
from subprocess import STDOUT
import re

from ilp.base_ilp import BaseILP


class Foil(BaseILP):

    def __init__(self, closed_world=True):

        # the name of the relation being learned
        self.name = "target_relation"
        self.example_name = "currr_ex"
        self.arity = 1
        self.types = ["E"]
        self.type_names = {}
        self.type_names["E"] = "exxx"
        self.type_names["O"] = "objj"
        self.type_names["V"] = "vall"

        # lists of positive and negative tuples
        self.pos = []
        self.neg = []

        # Examples
        self.examples = set()

        # Objects
        self.objects = set()

        # Values
        self.values = set()

        # the key is the name
        # the value is a list of tuples
        # aux relations only need positive examples, right?
        self.aux_relations = {}
        self.aux_rel_has_val = set()

        # whether or not to use closed world assumption
        self.closed_world = closed_world

        # rules for computing matches
        self.rules = ""

    def get_matches(self, X):

        if self.rules == "":
            return []

        X = {self.clean(a): self.clean(X[a]) for a in X}

        vals = set()
        objs = set()
        rels = set()

        for rel in X:
            if isinstance(X[rel], bool):
                rels.add((rel[0],) + (self.example_name,) + rel[1:])
            else:
                val = X[rel]
                if val == "":
                    val = "nil"
                rels.add((rel[0],) + (self.example_name,) + rel[1:] + (val,))
                vals.add(val)

            for v in rel[1:]:
                objs.add(v)

        data = self.type_names['E'] + "(" + self.example_name + ").\n"
        data += "\n".join([self.type_names['V'] + "(%s)." % v for v in vals]) + "\n"
        data += "\n".join([self.type_names['O'] + "(%s)." % v for v in objs]) + "\n"
        data += "\n".join([r[0] + "(" + ",".join(r[1:]) + ")." for r in rels]) + "\n"

        data += self.rules
        data += "#show %s/%i." % (self.name, self.arity)

        p = Popen(['clingo'], stdout=PIPE, stdin=PIPE, stderr=STDOUT)
        output = p.communicate(input=data.encode("utf-8"))[0]

        #print(data)
        #print(output.decode())

        matches = re.findall(self.name + "\(" + self.example_name + "," + ",".join(["(?P<arg%i>[^ ,()]+)" % i for i in range(self.arity-1)]) + "\)", output.decode())

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
            x = x.replace("sspp", " ")
            x = x.replace("qqmm", "?")
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
            x = x.replace("?", "qqmm")
            x = x.replace(" ", "sspp")
            x = x.replace("+", "plussign")
            x = x.replace("-", "minussign")
            x = x.replace("_", "underscore")
            x = x.replace("=", "equalsign")
            x = x.replace("*", "multsign")
            x = "a" + x
            return x
        else:
            return x
        
    def fit(self, T, X, y):

        # preprocess
        T = [self.clean(t) for t in T]
        X = [{self.clean(a): self.clean(x[a]) for a in x} for x in X]

        self.pos = []
        self.neg = []
        self.aux_relations = {}

        for i, correct in enumerate(y):
            example_id = "EXAMPLEID%i" % (i) 
            self.examples.add(example_id)

            # Get target tuple information
            if correct == 1:
                self.pos.append((example_id,) + T[i])
            else:
                self.neg.append((example_id,) + T[i])

            #for e in T[i]:
            #    self.values.add(e)

            # Get aux relation information
            for rel in X[i]:
                if rel[0] not in self.aux_relations:
                    self.aux_relations[rel[0]] = []

                if isinstance(X[i][rel], bool):
                    self.aux_relations[rel[0]].append((example_id,) + rel[1:])
                else:
                    val = X[i][rel]
                    if val == "":
                        val = "nil"
                    self.aux_relations[rel[0]].append((example_id,) + rel[1:] +
                                                     (val,))
                    self.aux_rel_has_val.add(rel[0])
                    self.values.add(val)

                # Add values
                for e in rel[1:]:
                    self.objects.add(e)

        self.types = ['E']
        for e in T[0]:
            if e in self.objects:
                self.types.append("O")
            elif e in self.values:
                self.types.append("V")
            else:
                raise Exception("tuple element is not an object or value")

        # Create subprocess
        p = Popen(['ilp/FOIL6/foil6', '-m100000000'], stdout=PIPE, stdin=PIPE, stderr=STDOUT)

        # construct file.
        # types and values
        data = "#E: " + ",".join(["%s" % v for v in self.examples]) + ".\n"
        data += "#O: " + ",".join(["%s" % v for v in self.objects]) + ".\n"
        data += "#V: " + ",".join(["*%s" % v for v in self.values]) + ".\n\n"

        # target relation header
        if len(self.pos) > 0 and len(self.pos[0]) > 0:
            self.arity = len(self.pos[0])
        elif len(self.neg) > 0 and len(self.neg[0]) > 0:
            self.arity = len(self.neg[0])
        data += self.name + "(" + ", ".join(self.types)
        data += ") " + "".join(["#" for i in range(self.arity)]) + "\n"

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
            arity = len(self.aux_relations[rel][0])
            if rel in self.aux_rel_has_val:
                data += "*" + rel + "(E, " + ", ".join(["O" for i in range(arity-2)]) + ", V"
            else:
                data += "*" + rel + "(E, " + ", ".join(["O" for i in range(arity-1)])
            # aux relations don't always need alll values to be bound
            data += ")\n"# " + "".join(["#" for i in range(arity)]) + "\n"

            # the positive examples of relation
            for t in self.aux_relations[rel]:
                data += ", ".join(t) + "\n"
            data += ".\n"

        #print(data)

        # Send data to subprocess
        output = p.communicate(input=data.encode("utf-8"))[0]

        self.rules = ""

        type_info = [self.type_names[t] for t in self.types]

        found = False
            
        # Process result
        for line in output.decode().split("\n"):
            print(line)
            #match = re.search('^' + self.name + ".+:-", line)
            matches = re.findall("^" + self.name + "\(" +
                                 ",".join(["(?P<arg%i>[^ ,()]+)" % i for i in
                                           range(self.arity)]) + "\)", line)
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

                ground_atoms = self.values.union(self.objects)
                for i, arg in enumerate(args):
                    if arg not in ground_atoms:
                        if not first_arg:
                            self.rules += ", "
                        else:
                            first_arg = False
                        self.rules += type_info[i] + "(" + arg + ")"

                self.rules += ".\n"
        
        print("POS: ", len(self.pos))
        print("NEG: ", len(self.neg))
        if not self.closed_world and not found and len(self.pos) > len(self.neg):
            self.rules += self.name + "(" + ",".join([chr(i + ord('A')) for i in range(len(self.types))]) + ") :- "
            self.rules += ", ".join([self.type_names[t] + "(" + chr(i +
                                                                    ord('A')) +
                                     ")" for i,t in enumerate(self.types)])
            self.rules += ".\n"

        print(self.rules)

