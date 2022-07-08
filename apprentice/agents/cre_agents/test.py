from apprentice.agents.CRE_Agent.extending import new_register_all, new_register_decorator, registries

if __name__ == "__main__":
    register_poop = new_register_decorator("poop")
    register_all_things = new_register_all("things", types=[int,str], acceptor_funcs=[lambda x : isinstance(x,tuple)])

    @register_poop
    def foo():
        return "foo"

    print(registries)

    with register_all_things() as things:
        a = 1
        b = 'q'
        c = ('eggo',)
        d = ['noodles']
        e = ['noodlez']

    with register_all_things() as things:
        x = 1
        b = 'q'
        z = ('eggo',)
        q = ['noodles']
        v = ['noodlez']

    print(things)
    print(registries)
    

