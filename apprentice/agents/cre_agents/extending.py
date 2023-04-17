import inspect
import warnings

# -----------------------------------------------------------------------
# : Type registration 

registries = {}

class Registery():
    def __init__(self, name, full_descr=None):
        self.name = name
        self.full_descr = name if full_descr is None else full_descr
        self.reg_dict = {}

    def __contains__(self, name):
        return name in self.reg_dict

    def __getitem__(self, name):
        if(name not in self.reg_dict):
            raise ValueError(f"No {self.full_descr} registered with name {name!r}.")
        return self.reg_dict[name]

    def __iter__(self):
        return iter(self.reg_dict.values())

    def __setitem__(self, name, val):
        self.reg_dict[name] = val

    def __str__(self):
        return str(self.reg_dict)

    def __len__(self):
        return len(self.reg_dict)

    def __str__(self):
        return str(self.reg_dict)

    __repr__ = __str__

def _resolve_name(obj, name_resolver=None):
    if(name_resolver is not None):
        return name_resolver(obj)
    elif(hasattr(obj,'__name__')):
        return obj.__name__
    else:
        raise ValueError(f"Cannot resolve name during registration: {obj}")

def _register(obj, registry, name=None, name_resolver=None, 
                insert_func=None, full_descr=None, args=[], kwargs={}, stack_extra=0):
    name = _resolve_name(obj) if name is None else name
    regular_name = name.lower().replace("_", "")
    if(regular_name in registry):
        warnings.warn(f"Redefinition of {full_descr} '{name}'.", stacklevel=2+stack_extra)

    if(insert_func is not None):
        insert_func(registry, obj, name, *args, **kwargs)
    else:
        registry[regular_name] = obj
    return obj


def new_register_decorator(type_name, name_resolver=None, insert_func=None, full_descr=None):
    full_descr = type_name if full_descr is None else full_descr
    registry = registries[type_name] = registries.get(type_name, Registery(type_name, full_descr))
    def register_whatever(*args,name=None,**kwargs):
        if(len(args) >= 1):
            return _register(args[0], registry, name, name_resolver, 
            insert_func, full_descr, kwargs=kwargs, stack_extra=1)
        else:
            return lambda obj: _register(obj, registry, name, name_resolver, 
                insert_func, full_descr, kwargs=kwargs, stack_extra=2)
    return register_whatever

class RegisterAll():
    def __init__(self, type_name, types=[], acceptor_funcs=[],
                     name_resolver=None, insert_func=None, full_descr=None):
        self.type_name = type_name
        self.registry = registries[type_name] = registries.get(type_name, Registery(type_name, full_descr))
        self.types = types
        self.acceptor_funcs = acceptor_funcs
        self.insert_func = insert_func
        self.name_resolver = name_resolver
        self.full_descr = self.type_name if full_descr is None else full_descr

    def __call__(self, n_back=1, *args, **kwargs):
        frame = b_frame = inspect.currentframe()
        try:
            for n in range(n_back):
                b_frame = b_frame.f_back
            locs = {**b_frame.f_locals}
        finally:
            del frame

        for name, obj in locs.items():
            # Skip any builtins
            if(name[:2] == "__" and name[-2:] == "__"):
                continue

            if(isinstance(obj, self.types) or
               any([f(obj) for f in self.acceptor_funcs])):

                # The registered thing needs to be
                if(hasattr(self, 'enter_locs') and name in self.enter_locs):
                    # print(name, id(self.enter_locs[name]), id(obj))
                    if(id(self.enter_locs[name]) == id(obj)): 
                        continue
                _register(obj, self.registry, name, self.name_resolver,
                    self.insert_func, self.full_descr, args, kwargs, stack_extra=1)

                if(hasattr(self, 'collected')):
                    self.collected.append(obj)
        return self

    def __enter__(self):
        frame = inspect.currentframe()
        try:
            self.enter_locs = {**frame.f_back.f_locals}
        finally:
            del frame
        # print("enter_locs:")
        # print(list(self.enter_locs))
        # print()

        self.collected = []
        return self.collected

    def __exit__(self,*args):
        self.__call__(n_back=2)
        del self.collected
        del self.enter_locs


def new_register_all(type_name, types=[], acceptor_funcs=[],
                     name_resolver=None, insert_func=None, full_descr=None):
    if(not isinstance(types,(list,tuple))): types = [types]
    if(not isinstance(acceptor_funcs,(list,tuple))): acceptor_funcs = [acceptor_funcs]

    types = tuple(types)
    acceptor_funcs = tuple(acceptor_funcs)

    return RegisterAll(type_name, types, acceptor_funcs, name_resolver, insert_func, full_descr)
        



