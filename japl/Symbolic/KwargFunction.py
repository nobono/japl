from typing import Any
from sympy import Function
from sympy import Symbol



class KwargFunction(Function):

    """This class inherits from sympy.Function and allows
    keyword arguments."""

    __slots__ = ("name", "kwargs")

    # @classmethod
    # def eval(cls, *args):
    #     """kwargs are not used in eval by default; augmenting here."""
    #     return super().eval(*args)


    def __new__(cls, name: str, kwargs: dict = {}):
        # create the object
        args = tuple([v for v in kwargs.values()])
        obj = super().__new__(cls, *args)
        # attatch kwargs to the object
        obj.name = name
        obj.kwargs = kwargs
        return obj


    def __reduce__(self) -> str | tuple[Any, ...]:
        """defines serialization of class object"""
        return (self.__class__, (self.name, self.kwargs), self.__getstate__())


    def __repr__(self) -> str:
        return self.__str__()


    def __str__(self) -> str:
        _kwargs = ", ".join([f"{k}={v}" for k, v in self.kwargs.items()])
        return f"{self.name}({_kwargs})"


    def _pythoncode(self, *args, **kwargs):
        """string representation of object when using sympy.pycode()
        for python code generation"""
        return self.__str__()


    def _ccode(self, *args, **kwargs):
        """string representation of object when using sympy.ccode()
        for c-code generate"""
        _kwargs = []
        for k, v in self.kwargs.items():
            _kwargs += [f"py::kw(\"{k}\"_a={v})"]
        _kwargs = ", ".join(_kwargs)
        return f"{self.name}({_kwargs})"


    def _sympystr(self, printer):
        """string representation of object used in sympy string formatting."""
        return self.__str__()


    def __call__(self, *args, **kwargs):
        args = tuple([v for v in kwargs.values()])
        obj = super().__new__(self.__class__, *args)
        obj.kwargs = kwargs
        obj.name = self.name
        return obj


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
            return self.func(self.name, new_kwargs)

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
                return self.func(self.name, new_kwargs), True
        return self, False
