
class ParameterSet(object):

    def __new__(cls, *modules, recurse = True, _in_list=None):
        if _in_list is not None:
            ob =  super().__new__(cls)
            ob.__init__(None, True, _in_list=_in_list)
            return ob
        ob_list = []
        c_mod = super().__new__(cls)
        c_mod.__init__(modules[0], recurse = recurse, _in_list = None)
        for i in range(1, len(modules)):
            next_mod = super().__new__(cls)
            next_mod.__init__(modules[i], recurse = recurse, _in_list = None)
            c_mod = c_mod + next_mod
        return c_mod


    def __init__(self, modules, recurse = True, _in_list=None):
        if _in_list is not None:
           self.param_list = _in_list
        else:
            self.param_list = list(modules.parameters(recurse))

    def __iter__(self):
        return self.param_list.__iter__()

    def __len__(self):
        return len(self.param_list)

    @staticmethod
    def _in_is( ob, it):
        for item in it:
            if item is ob:
                return True
        return False

    def __add__(self, other):
        out = ParameterSet(None, _in_list=self.param_list)
        for param in other.param_list:
            if not ParameterSet._in_is(param, self.param_list):
                out.param_list.append(param)
        return out

    def __sub__(self, other):
        out = ParameterSet(None, _in_list=[])
        for param in self.param_list:
            if not ParameterSet._in_is(param, other.param_list):
                out.param_list.append(param)
        return out