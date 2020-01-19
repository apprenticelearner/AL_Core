import ast
import hashlib
# import inspect
import re
import types

import apprentice.explain.inspect_patch as inspect
from experta import Fact


def dump(*asts):
    for x in asts:
        print(ast.dump(x))


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


def get_func_from_ast(tree, globs=None):
    """ return functions/lambas from a module-level ast, i.e. function
    definition or lambda-assignments"""
    compiled = compile(tree, '', 'exec')

    module = types.ModuleType("foo_module")
    exec(compiled, module.__dict__)

    funcs = [x for x in module.__dict__.values() if
             isinstance(x, types.FunctionType)]

    if globs is not None:
        for gid in ast_ids(tree):
            if isinstance(globs['__builtins__'], dict):
                if gid not in globs['__builtins__'].keys():
                    setattr(module, gid, globs[gid])
            elif gid not in dir(globs['__builtins__']):
                setattr(module, gid, globs[gid])

    if len(funcs) == 0:
        return None

    if len(funcs) == 1:
        ret = funcs[0]
        ret._ast = tree

    return ret


def parse(func, drop_declare=True):
    """ func: function
        returns: list of ast of function and its globals
    """

    # if len(funcs) > 1:
    # return list(filter(lambda x: x is not None, [parse(f) for f in funcs]))

    source = inspect.getsource(func).strip().split('\n')

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
        temp = []
        parens = 0
        self_declare = 'self.declare'
        for s in source:
            if self_declare in s:
                parens = 1
            if parens > 0:
                parens += s.count('(')
                parens -= s.count(')')
            else:
                temp.append(s)
        source = temp
        # source = [s for s in source if 'self.declare' not in s]
        # todo: make dropping declare more robust, add SAI support,
        #  last function logic

    # empty function
    if len(source) == 1:
        return None

    # unindent source
    extra_whitespace = source[0].find('def')
    if extra_whitespace > 0:
        source = [s[extra_whitespace:] for s in source]

    source = '\n'.join(source)
    # print(source)
    if hasattr(func, '_wrapped'):
        func = func._wrapped
    if hasattr(func, '__globals__'):
        globs = func.__globals__
    else:
        globs = {}
    return ast.parse(source), globs


def ast_ids(tree):
    """ recursively inspects ast for NEW id fields declared in the ast and
    yields them"""
    enc = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            for arg in node.args.args:
                enc.append(arg.arg)

        if hasattr(node, 'id'):
            if isinstance(node.ctx, ast.Store):
                enc.append(node.id)
            if isinstance(node.ctx, ast.Load) and node.id not in enc:
                yield node.id


def rename(mapping, tree):
    """ rename function ast variable names according to provided mapping """

    for node in ast.walk(tree):
        if hasattr(node, 'id'):
            while node.id in mapping:
                node.id = mapping[node.id]
        if hasattr(node, 'arg'):
            while node.arg in mapping:
                node.arg = mapping[node.arg]

    # remove duplicate arguments
    for node in ast.walk(tree):
        if type(node) is ast.arguments:
            unique = {}
            for arg in node.args:
                if arg.arg not in unique.keys():
                    unique[arg.arg] = arg

            node.args = list(unique.values())

    return tree


def join(new_name, new_args, globals, *asts):
    """ join multiple functions under a new signature, returns new function"""
    root = asts[0]

    root.body[0].name = new_name
    root.body[0].args.args = [ast.arg(a, None, lineno=1, col_offset=9) for a in
                              new_args]

    for graft in asts[1:]:
        root.body[0].body += graft.body[0].body

    return get_func_from_ast(root, globals)


# def parseprint(foo, filename='<unknown>', mode="exec", **kwargs):
#     """Parse the source and pretty-print the AST."""
#     source = inspect.getsource(foo)
#     node = ast.parse(source, filename, mode=mode)
#     print(ast.dump(node, **kwargs))

if __name__ == "__main__":
    class FooClass:
        att = 'FooAttribute'


    def foo1(self, arg1):
        f = FooClass()
        FooClass()
        print(f.att, arg1)


    def foo2(self, arg2):
        f = FooClass()
        FooClass()
        print(f.att, arg2)


    def testit(x):
        if x == 0:
            return None
        return x + 1, x + 2


    tree1, gl1 = parse(foo1)
    tree2, gl2 = parse(foo2)

    print(list(ast_ids(tree1)))

    fn1 = get_func_from_ast(tree1, gl1)
    fn2 = get_func_from_ast(tree2, gl2)

    fn1(None, 1)
    fn2(None, 2)
