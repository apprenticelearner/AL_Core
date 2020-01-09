import ast
import hashlib
#import inspect
import re
import types

from experta import Fact

import apprentice.explain.inspect_patch as inspect

def rename_function_unique(func, suffix, reverse=False):
    s = inspect.getsource(func)
    s = s[s.find("def "):]
    tree = ast.parse(s)
    # tree.body[0].decorator_list = []
    args = list(inspect.signature(func).parameters.keys())[1:]
    for node in ast.walk(tree):
        if hasattr(node, 'id'):
            if node.id in args:
                node.id += suffix
        if hasattr(node, 'arg'):
            if node.arg in args:
                node.arg += suffix
    # print(ast.dump(tree))
    return get_func_from_ast(tree)


def rename_condition_unique(cond, suffix):
    # todo change this to dictionary mapping for reversing as well?
    if isinstance(cond, Fact):
        for v in cond.values():
            v.__bind__ += suffix
    else:
        [rename_condition_unique(sub, suffix) for sub in cond]


def rename_rule_unique(rule):
    suffix = '_' + hashlib.sha1(
        rule._wrapped.__name__.encode("UTF-8")).hexdigest()[:8]
    new_wrapped = rename_function_unique(rule._wrapped, suffix)
    rename_condition_unique(rule, suffix)
    rule._wrapped = None
    rule.__call__(new_wrapped)
    return rule


def parse_lambda(l):
    """ Attempts to parse lambda ast from source.
    Might break under certain conditions, e.g. if there are multiple
    lambdas defined in the same line/lines of source

    returns an ast assigning the lambda to the identifier "foo42",
    so the lambda can be retrieved from the module it is executed in
    """
    if hasattr(l, '_ast'):
        return l._ast

    s = inspect.getsource(l)
    s2 = s[s.find('lambda '):]
    oparens = 0
    for i, c in enumerate(s2):
        if c == '(':
            oparens += 1
        if c == ')':
            if oparens == 0:
                s2 = s2[:i]
                break
            else:
                oparens -= 1

    s2 = "foo42 = " + s2
    return ast.parse(s2)


def rename_lambda(l, mapping):
    """ takes a lambda, returns a lambda """
    tree = parse_lambda(l)
    tree2 = rename(mapping, tree)
    return get_func_from_ast(tree2)


def get_func_from_ast(tree):
    """ return functions/lambas from a module-level ast, i.e. function
    definition or lambda-assignments"""
    compiled = compile(tree, '', 'exec')
    module = types.ModuleType("testmodule")
    exec(compiled, module.__dict__)

    funcs = [x for x in module.__dict__.values() if
             isinstance(x, types.FunctionType)]

    if len(funcs) == 0:
        return None

    if len(funcs) == 1:
        ret = funcs[0]
        ret._ast = tree

    return ret


def parse(*funcs, drop_declare=True):
    """ *args: list of functions
        returns: list of asts
    """

    if len(funcs) > 1:
        return list(filter(lambda x: x is not None, [parse(f) for f in funcs]))

    source = inspect.getsource(funcs[0]).strip().split('\n')

    # drop comments
    source = [s.split('#')[0] for s in source]

    # drop decorators
    def_line = -1
    for i, s in enumerate(source):
        if 'def' in s and 'self' in s:
            assert def_line == -1
            def_line = i

    source = source[def_line:]

    # drop blank lines
    source = list(
        filter(lambda x: len(re.sub("[^a-zA-Z]", "", x)) > 0, source))

    # drop "declare" lines
    if drop_declare:
        source = [s for s in source if 'self.declare' not in s]
        # todo: make dropping declare more robust, add SAI support,
        #  last function logic

    # empty function
    if len(source) == 1:
        return None

    # unindent source
    extra_whitespace = source[0].find('def')
    if extra_whitespace > 0:
        source = [s[extra_whitespace:] for s in source]

    return ast.parse('\n'.join(source))


def rename(mapping, *asts):
    """ rename function ast variable names according to provided mapping """
    for tree in asts:
        for node in ast.walk(tree):
            if hasattr(node, 'id'):
                while node.id in mapping:
                    node.id = mapping[node.id]
            if hasattr(node, 'arg'):
                while node.arg in mapping:
                    node.arg = mapping[node.arg]

    if len(asts) == 1:
        return asts[0]

    return asts


def join(new_name, new_args, *asts):
    """ join multiple functions under a new signature, returns new function"""
    root = asts[0]

    root.body[0].name = new_name
    root.body[0].args.args = [ast.arg(a, None, lineno=1, col_offset=9) for a in
                              new_args]

    for graft in asts[1:]:
        root.body[0].body += graft.body[0].body

    return get_func_from_ast(root)


def dump(*asts):
    for x in asts:
        print(ast.dump(x))


def parseprint(foo, filename='<unknown>', mode="exec", **kwargs):
    """Parse the source and pretty-print the AST."""
    source = inspect.getsource(foo)
    node = ast.parse(source, filename, mode=mode)
    print(ast.dump(node, **kwargs))
