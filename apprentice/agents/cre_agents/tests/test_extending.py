from apprentice.agents.cre_agents.extending import new_register_all, new_register_decorator, registries

def test_register():
    register_funcs = new_register_decorator("my_funcs")
    register_all_things = new_register_all("things", types=[dict], acceptor_funcs=[lambda x : isinstance(x,tuple)])

    @register_funcs
    def foo():
        return "foo"

    @register_funcs
    def foo():
        return "foo"

    assert registries['my_funcs']['foo'] == foo

    with register_all_things() as things:
        a = {1:0}
        b = {'q':0}
        c = ('eggo',)
        d = ['noodles']
        e = ['noodlez']

    assert len(things) == 3
    assert {1:0} in things 
    assert {'q':0} in things 
    assert ('eggo',) in things 

    print("things:", things)

    print(registries)

    with register_all_things as things:
        x = {7:0}
        b = {'q' : 0}
        z = ('eggo',)
        q = ['noodles']
        v = ['noodlez']

    print("things:", things)
    assert len(things) == 3
    assert {7:0} in things 
    assert {'q':0} in things 
    assert ('eggo',) in things 

    register_all_stuff = new_register_all("stuff", types=list)

    def reg():
        r = ["r"]
        q = ["Q"]
        register_all_stuff()   

    reg()
    print(registries)
    stuff_registry = registries['stuff']
    assert len(stuff_registry) == 2
    assert stuff_registry['r'] == ['r']
    assert stuff_registry['q'] == ["Q"]

    print(things)
    print(registries)

if __name__ == "__main__":
    test_register()
    

