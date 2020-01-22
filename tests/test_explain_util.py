from apprentice.explain.util import parse, ast_ids, get_func_from_ast


class FooClass:
    att = 'FooAttribute'

def test_1():
    # does not work in pytest because of globals.. ?
    def foo1(self, arg1):
        f = FooClass()
        FooClass()
        return (f.att, arg1)

    def foo2(self, arg2):
        f = FooClass()
        FooClass()
        return (f.att, arg2)

    tree1, gl1 = parse(foo1)
    tree2, gl2 = parse(foo2)

    l = list(ast_ids(tree1))
    assert l == ['FooClass', 'FooClass']

    fn1 = get_func_from_ast(tree1, gl1)
    fn2 = get_func_from_ast(tree2, gl2)

    assert fn1(None, 1) == ('FooAttribute', 1)
    assert fn2(None, 2) == ('FooAttribute', 2)

