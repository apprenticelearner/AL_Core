from apprentice.agents.cre_agents.extending import new_register_decorator, registries


def insert_transform(registry, func, name=None, **config):
    if(name is None):
        name = func.__name__
    # regular_name = name.lower().replace("_", "")
    registry[name] = (func, config)


register_transform = new_register_decorator('transform',insert_func=insert_transform)
transfrom_registry = registries['transform']
# Formats
#  - py_obj
#  - working_memory
#  - working_memory_flat
#  - working_memory_flat_feat

# class StateTransform:
#     def __init__(self, from_type, to_type, func):
#         self.from_type = from_type
#         self.to_type = to_type
#         self.func = func

# A dictionary of to_type : {**from_type : transform_func}
# registered_transforms = {}
from cre.utils import PrintElapse


# Usage Note: State is a class factory of itself
#  It makes a state class bound to an agent
#    state_cls = State(agent)
#  Transforms can be registered on it 
#    @state_cls.register_transform(...)
#    def my_transform(...):
#        ...
#  Individual state instances are created from that class
#    state = state_cls({"my_format" : ...})

class State():    
    # ------
    # : Class Methods

    # When State(agent) is called a new state_cls is created
    #  on which a set of transforms can be registered

    def __inst_new__(cls, state_formats={}):
        return object.__new__(cls)

    def __new__(cls, agent):
        scls = type('State', (State,),{
            # The agent and registered transforms are class attributes
            'agent' : agent,
            'transforms': {},

            # Use default __new__ in subclass 
            '__new__' : cls.__inst_new__
        })
        return scls

    @classmethod
    def register_transform(cls, *args, name=None, **kwargs):
        from apprentice.agents.cre_agents.extending import _register
        if(len(args) >= 1):
            return _register(args[0], cls.transforms, name, 
                insert_func=insert_transform, kwargs=kwargs)
        else:
            return lambda obj: _register(obj, cls.transforms, name, 
                insert_func=insert_transform,  kwargs=kwargs)        

    # ------
    # : Instance Methods

    def __init__(self, state_formats={}):
        formats = {}
        for key, val in state_formats.items():
            if(not isinstance(val, tuple)):
                formats[key] = (val, {}) 
            else:
                formats[key] = val
        self.state_formats = formats

    # TODO : Instead of cleaning up all old except is_incremental parts 
    #    should instead do something like join() on the incremental_parts
    def clear(self, keep_incr=True):
        state_formats = {}
        if(keep_incr):
            for k, (val, config) in self.state_formats.items():
                if(config.get("is_incremental", False)):
                    state_formats[k] = (val, config)
        self.state_formats = state_formats

    def get_transform(self, frmt):
        if(frmt in self.transforms):
            return self.transforms[frmt]
        elif(frmt in transfrom_registry):
            return transfrom_registry[frmt]
        return None, {}

    def get(self, frmt, return_config=False,**kwargs):
        _, config = self.get_transform(frmt)
        if( (frmt not in self.state_formats) or 
            kwargs.get('force_rebuild',False) or 
            config.get('is_incremental', False)):

            if(frmt not in transfrom_registry and
               frmt not in self.transforms):
                raise ValueError(f"No transform registered for producing state type {frmt!r}")
            transform, config = self.get_transform(frmt)
            try:
                for preq_frmt in config.get("prereqs",[]):
                    self.get(preq_frmt)
            except Exception as e:
                raise ValueError(f"Production of state type {frmt!r} failed on prerequisite {preq_frmt!r}\n\t{e}")

            self.state_formats[frmt] = (transform(self),config)

        val, config = self.state_formats[frmt]
        if(not return_config):
            return val
        else:
            return val, config


    def __getitem__(self, frmt):
        return self.state_format[frmt]

    def __contains__(self, frmt):
        return (frmt in self.state_formats)

    def set(self, frmt, value, **config):
        self.state_formats[frmt] = (value, config)


def encode_neighbors(objs, l_str='left', r_str="right", a_str="above", b_str="below", strip_attrs=["x", "y", "width", "height"]):
  # objs = list(_objs.values()) if(isinstance(_objs,dict)) else _objs
  objs_list = list(objs.values())

  rel_objs = []
  for i, obj in enumerate(objs):
    rel_objs.append({
      l_str : [],
      r_str : [], 
      a_str : [],
      b_str : [],
    })

  for i, a_obj in enumerate(objs_list):
    for j, b_obj in enumerate(objs_list):
      if(i != j):
        if(a_obj['y'] > b_obj['y'] and
           a_obj['x'] < b_obj['x'] + b_obj['width'] and
           a_obj['x'] + a_obj['width'] > b_obj['x']):
            dist = a_obj['y'] - b_obj['y'];
            rel_objs[i][a_str].append((dist, j));
            rel_objs[j][b_str].append((dist, i));

        if(a_obj['x'] < b_obj['x'] and
           a_obj['y'] + a_obj['height'] > b_obj['y'] and
           a_obj['y'] < b_obj['y'] + b_obj['height']):
            dist = b_obj['x'] - a_obj['x']
            rel_objs[i][r_str].append((dist, j));
            rel_objs[j][l_str].append((dist, i));

  strip_attrs_set = set(strip_attrs)
  out = {}   
  for (_id, obj), rel_obj in zip(objs.items(), rel_objs):
    # print(_id, obj["x"],obj["y"],obj["width"],obj["height"])
    new_obj = {k:v for k,v in obj.items() if k not in strip_attrs}
    new_obj[l_str] = objs_list[sorted(rel_obj[l_str])[0][1]]["id"] if len(rel_obj[l_str]) > 0 else ""
    new_obj[r_str] = objs_list[sorted(rel_obj[r_str])[0][1]]["id"] if len(rel_obj[r_str]) > 0 else ""
    new_obj[a_str] = objs_list[sorted(rel_obj[a_str])[0][1]]["id"] if len(rel_obj[a_str]) > 0 else ""
    new_obj[b_str] = objs_list[sorted(rel_obj[b_str])[0][1]]["id"] if len(rel_obj[b_str]) > 0 else ""
    out[_id] = new_obj

  # if(any([obj.get('value',"") != "" and obj.get('value',"")]))  
  # print()

  return out

if __name__ == "__main__":
    pass







    # print("HI")







'''
Notes:
What do we need to deal with?
1) A when learning mechanism can ask for any state format. 
    Which can in principle be encoded relative to a match.
2) A when learning mechanism can 

'''





