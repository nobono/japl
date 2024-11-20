from sympy import Function
from sympy import Mul
from sympy import Add



class KwargFunction(Function):

    """This class inherits from sympy.Function and allows
    keyword arguments."""

    kwargs = {}

    @classmethod
    def eval(cls, *args, **kwargs):
        """kwargs are not used in eval by default; augmenting here."""
        return super().eval(*args)

    def __new__(cls, name, **kwargs):
        # create the object
        args = tuple([v for v in kwargs.values()])
        obj = super().__new__(cls, *args)
        # attatch kwargs to the object
        obj.name = name
        obj.kwargs = kwargs
        return obj


    def __repr__(self) -> str:
        return self.__str__()


    def __str__(self) -> str:
        _kwargs = ",".join([f"{k}={v}" for k, v in self.kwargs.items()])
        return f"{self.name}({_kwargs})"


    def _pythoncode(self, *args, **kwargs):
        return self.__str__()


    def _ccode(self, *args, **kwargs):
        return self.__str__()


    def __call__(self, **kwargs):
        args = tuple([v for v in kwargs.values()])
        obj = super().__new__(self.__class__, *args)
        obj.kwargs = kwargs
        obj.name = self.name
        return obj


    def __mul__(self, other):
        return Mul(self, other)


    def __add__(self, other):
        return Add(self, other)
