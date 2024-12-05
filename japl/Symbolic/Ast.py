from sympy.codegen.ast import FunctionCall
from sympy.codegen.ast import Token
from sympy.codegen.ast import String
from sympy.codegen.ast import Tuple
from sympy import ccode



class Dict(Token):
    __slots__ = _fields = ("kwpairs",)
    defaults = {"kwpairs": {}}

    @staticmethod
    def _construct_kwpairs(pairs):
        return dict(pairs)

    def __str__(self):
        return str(self.kwpairs)

    def __len__(self):
        return self.kwpairs.__len__()

    def items(self):
        return ((key, val) for key, val in self.kwpairs.items())

    def values(self):
        return (val for val in self.kwpairs.values())

    def keys(self):
        return (key for key in self.kwpairs.keys())

    def _ccode(self, *args, **kwargs):
        kwargs_str = ", ".join(["{" + f"\"{key}\", " + ccode(val) + "}"
                                for key, val in self.kwpairs.items()])
        kwargs_str = kwargs_str.strip(", ")
        kwargs_str = "{" + kwargs_str + "}"
        return kwargs_str


class CodegenFunction(FunctionCall):
    """ Represents a call to a function in the code.

    Parameters
    ==========

    name : str
    function_args : Tuple

    Examples
    ========

    >>> from sympy.codegen.ast import FunctionCall
    >>> from sympy import pycode
    >>> fcall = FunctionCall('foo', 'bar baz'.split())
    >>> print(pycode(fcall))
    foo(bar, baz)

    """
    __slots__ = _fields = ('name', 'function_args', 'function_kwargs')
    defaults = {"function_args": Tuple(), "function_kwargs": Dict()}

    _construct_name = String
    _construct_function_args = staticmethod(lambda args: Tuple(*args))
    _construct_function_kwargs = staticmethod(lambda kwargs: Dict(kwargs))


    @staticmethod
    def _dict_to_kwargs_str(dkwargs: dict):
        kwargs_list = []
        for key, val in dkwargs.items():
            kwargs_list += [f"{key}={val}"]
        return ", ".join(kwargs_list)


    def _ccode(self, *args, **kwargs):
        params_str = ""
        params_str = ", ".join([ccode(i) for i in self.function_args])
        if len(self.function_args) and len(self.function_kwargs):
            params_str += ", "
        kwargs_list = []
        for key, val in self.function_kwargs.items():
            kwargs_list += ["{" + f"\"{key}\", {val}" + "}"]
        if kwargs_list:
            params_str += "{" + ", ".join(kwargs_list) + "}"
        return f"{self.name}({params_str})"


    def _pythoncode(self, *args, **kwargs):
        params_str = ""
        params_str = ", ".join([ccode(i) for i in self.function_args])
        if len(self.function_args) and len(self.function_kwargs):
            params_str += ", "
        if (kwargs_str := self._dict_to_kwargs_str(self.function_kwargs)):
            params_str += f"{kwargs_str}"
        return f"{self.name}({params_str})"


# # TESTS
# class func(JaplFunction):
#     pass

# avar = Variable(a, type=double)
# bvar = Variable(b, type=double)
# cvar = Variable(c, type=double)

# # Dict
# assert ccode(Dict()) == "{}"
# assert ccode(Dict({})) == "{}"
# assert ccode(Dict({"x": 1, 'y': 2.0})) == "{{\"x\", 1}, {\"y\", 2.0}}"
# assert ccode(Dict({"x": a, 'y': bvar})) == "{{\"x\", a}, {\"y\", b}}"

# # Function
# assert ccode(Func("func")) == "func()"
# assert ccode(Func("func", ())) == "func()"
# assert ccode(Func("func", (avar,))) == "func(a)"
# assert ccode(Func("func", (avar, bvar))) == "func(a, b)"
# assert ccode(Func("func", (), {"x": 1, "y": 2})) == "func(x=1, y=2)"
# assert ccode(Func("func", (avar, bvar), {"x": 1, "y": 2})) == "func(a, b, x=1, y=2)"
