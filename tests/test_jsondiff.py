from apprentice.working_memory import ExpertaWorkingMemory
from jsondiff import replace


def test_update_1():
    initial = {replace: {'?ele-JCommTable5.R0C0': {'contentEditable': True,
                                                   'id': 'JCommTable5.R0C0',
                                                   'value': ''},
                         '?ele-JCommTable6.R0C0': {'contentEditable': True,
                                                   'id': 'JCommTable6.R0C0',
                                                   'value': ''},
                         '?ele-JCommTable8.R0C0': {'contentEditable': True,
                                                   'id': 'JCommTable8.R0C0',
                                                   'value': ''},
                         '?ele-ctatdiv68': {'id': 'ctatdiv68'},
                         }}

    dd = {'?ele-JCommTable6.R0C0': {'contentEditable': False, 'value': '2'}}

    wm = ExpertaWorkingMemory()
    wm.update(initial)

    f = wm.lookup['?ele-JCommTable6.R0C0']
    assert f['value'] == ''
    for f2 in wm.facts.values():
        if 'id' in f2:
            if f2['id'] == f['id']:
                wmf = f2
    assert f.as_dict() == wmf

    wm.update(dd)

    assert wm.lookup['?ele-JCommTable6.R0C0'].as_dict()['value'] == '2'


