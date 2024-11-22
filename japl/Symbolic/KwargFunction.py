from typing import Any, Optional, Callable
from sympy import Function
from sympy import Float
from sympy.core.function import FunctionClass, UndefinedFunction
from sympy import Symbol
from sympy.core.cache import cacheit



class KwargFunction(Function):

    """This class inherits from sympy.Function and allows
    keyword arguments."""

    __slots__ = ("name", "kwargs")

    parent = ""
    no_keyword = False

    # @classmethod
    # def eval(cls, *args):
    #     """kwargs are not used in eval by default; augmenting here."""
    #     # return super().eval(*args)
    #     return None


    # def __new__(cls, name: str, kwargs: dict = {}):
    #     # create the object
    #     args = tuple([v for v in kwargs.values()])
    #     obj = super().__new__(cls, name, *args)
    #     # attatch kwargs to the object
    #     obj.name = name
    #     obj.kwargs = kwargs
    #     return obj


    def set_parent(self, parent: str):
        self.parent = parent
        self.name = f"{parent}.{self.name}"


    def __new__(cls, *args, **kwargs):
        # if args is tuple[tuple, ...], then __new__ is being
        # called from __setstate__ which passes keyword
        # arguments into *args as a tuple[tuple, ...]
        if len(args) and isinstance(args[0], tuple):
            kwargs = dict(args[0])
        # create the object
        kwargs = kwargs or {}
        args = tuple([v for v in kwargs.values()])
        obj = super().__new__(cls, *args)
        # attatch kwargs to the object
        if obj.parent:
            obj.name = f"{obj.parent}.{str(cls)}"
        else:
            obj.name = str(cls)
        obj.kwargs = kwargs
        return obj


    # def __setstate__(self, state: dict):
    #     kwargs = state.get("kwargs", {})
    #     self.name = state.get("name", str(self.__class__))
    #     self.kwargs = kwargs


    def __reduce__(self) -> str | tuple[Any, ...]:
        """defines serialization of class object"""
        # return (self.__class__, (self.name, self.kwargs), self.__getstate__())
        state = {"kwargs": self.kwargs}
        return (self.__class__, (tuple(self.kwargs.items()),), state)


    def __repr__(self) -> str:
        return self.__str__()


    def __str__(self) -> str:
        _kwargs = ", ".join([f"{k}={v}" for k, v in self.kwargs.items()])
    #     return f"{self.name}({_kwargs})"
        if self.parent:
            name = f"{self.parent}.{self.__class__}"
        else:
            name = self.name
        return f"{name}({_kwargs})"


    def _pythoncode(self, *args, **kwargs):
        """string representation of object when using sympy.pycode()
        for python code generation"""
        return self.__str__()


    def _ccode(self, *args, **kwargs):
        """string representation of object when using sympy.ccode()
        for c-code generate"""
        # _kwargs = []
        # for k, v in self.kwargs.items():
        #     _kwargs += [f"py::kw(\"{k}\"_a={v})"]
        # _kwargs = ", ".join(_kwargs)
        # args = tuple([v for v in self.kwargs.values()])
        # pass keyword args as a std::map
        args_str = ""

        if self.no_keyword:
            args_str = ", ".join([str(v) for v in self.kwargs.values()])
        else:
            for key, val in self.kwargs.items():
                args_str += "{" + f"\"{key}\", " + str(val) + "}, "
            args_str = "{" + args_str + "}"

        if self.parent:
            name = f"{self.parent}.{self.__class__}"
        else:
            name = self.name
        return f"{name}({args_str})"


    def _sympystr(self, printer):
        """string representation of object used in sympy string formatting."""
        return self.__str__()


    # def __call__(self, *args, **kwargs):
    #     args = tuple([v for v in kwargs.values()])
    #     obj = super().__new__(self.__class__, *args)
    #     obj.kwargs = kwargs
    #     obj.name = self.name
    #     return obj


    def func(self, *args):  # type:ignore
        """This overrides @property func. which is used to rebuild
        the object. This is used in sympy cse."""
        new_kwargs = {key: val for key, val in zip(self.kwargs.keys(), args)}
        # return self.__class__(self.name, new_kwargs)
        return self.__class__(**new_kwargs)


    def _eval_subs(self, old, new):  # type:ignore
        """this override method handles how this class interacts
        with sympy.subs"""
        # substitute self
        if self == old:
            return new

        # substitute function args
        new_kwargs = {
                key: val._subs(old, new) if hasattr(val, '_subs') else val
                for key, val in self.kwargs.items()
                }
        # if subs changes the args, return new instance to apply
        # changes
        if new_kwargs != self.kwargs:
            # return self.func(self.name, new_kwargs)
            # return self.__class__(self.name, new_kwargs)
            return self.__class__(**new_kwargs)

        # no subs applied
        return self


    def _xreplace(self, rule):
        """Helper for xreplace. Tracks whether a replacement actually occurred."""
        if self in rule:
            return rule[self], True
        elif rule:
            args = []
            changed = False
            for a in self.args:
                _xreplace = getattr(a, '_xreplace', None)
                if _xreplace is not None:
                    a_xr = _xreplace(rule)
                    args.append(a_xr[0])
                    changed |= a_xr[1]
                else:
                    args.append(a)
            args = tuple(args)
            if changed:
                new_kwargs = {key: val for key, val in zip(self.kwargs.keys(), args)}
                # return self.func(self.name, new_kwargs), True
                # return self.__class__(self.name, new_kwargs), True
                return self.__class__(**new_kwargs), True
        return self, False


    def _hashable_content(self):  # type:ignore
        """
        (Override)
        Return a tuple of information about self that can be used to
        compute the hash. If a class defines additional attributes,
        like ``name`` in Symbol, then this method should be updated
        accordingly to return such relevant attributes.

        Defining more than _hashable_content is necessary if __eq__ has
        been defined by a class. See note about this in Basic.__eq__."""
        return (self._args, self.name, tuple(self.kwargs))


    def __contains__(self, item):
        # check if item is a free symbol in the expression
        in_args = any(item == arg for arg in self.args)
        in_free_symbols = item in self.free_symbols
        return in_args or in_free_symbols


    def _eval_derivative(self, sym):
        return
