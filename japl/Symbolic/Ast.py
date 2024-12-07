from sympy.codegen.ast import FunctionCall
from sympy.codegen.ast import FunctionPrototype
from sympy.codegen.ast import FunctionDefinition
from sympy.codegen.ast import Declaration
from sympy.codegen.ast import Variable
from sympy.codegen.ast import Token
from sympy.codegen.ast import String
from sympy.codegen.ast import Tuple
from sympy.codegen.ast import Type
from sympy.codegen.ast import Node
from sympy.codegen.ast import untyped
from sympy.core.function import Function
# from sympy.codegen.ast import complex_
# from sympy import ccode
from sympy import pycode
from sympy import Float, Integer, Matrix
from sympy import MatrixSymbol
from sympy.printing.c import C99CodePrinter
from sympy.printing.c import value_const


class CodeGenPrinter(C99CodePrinter):

    def _print_Constructor(self, expr):
        params = expr.parameters
        # handle both Declaration and Variables passed
        if isinstance(expr.variable, Declaration):
            var = expr.variable.variable
        else:
            var = expr.variable

        if var.type == untyped:
            raise ValueError("C does not support untyped variables")

        elif isinstance(var, Variable):
            result = '{t} {s}({p})'.format(
                t=self._print(var.type),
                s=self._print(var.symbol),
                p=", ".join([self._print(p) for p in params])
            )
        else:
            raise NotImplementedError("Unknown type of var: %s" % type(var))
        # if params != None:  # Must be "!= None", cannot be "is not None" # noqa
        #     result += ' = %s' % self._print(params)
        return result


def ccode(expr, **kwargs):
    printer = CodeGenPrinter()
    return printer.doprint(expr, **kwargs)


class Constructor(Token):

    """Token for adding constructor to variable Declaration.
    Signature `Constructor(Variable, Tuple)`

    Arguments
    ---------
        variable:
            takes a Variable or Declaration
        parameters:
            the parameters of the constructor
    """

    __slots__ = _fields = ("variable", "parameters",)
    defaults = {"parameters": Tuple()}

    @staticmethod
    def _construct_variable(params):
        return params

    @staticmethod
    def _construct_parameters(params):
        return params


class KwargsToken(Token):

    """Token but accepts kwargs input"""

    def __new__(cls, *args, **kwargs):
        token_pops = ["exclude", "apply"]
        pops = []
        kwargs_passthrough = {}
        for k, v in kwargs.items():
            if k in token_pops:
                kwargs[k] = v
                pops += [k]
        for k in pops:
            kwargs.pop(k)
        if kwargs:
            args += (kwargs,)
        obj = super().__new__(cls, *args, **kwargs_passthrough)
        return obj


class Dict(Type):
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

    def _pythoncode(self, *args, **kwargs):
        kwargs_str = ", ".join([f"{key}: {pycode(val)}" for key, val in self.kwpairs.items()])
        kwargs_str = kwargs_str.strip(", ")
        kwargs_str = "{" + kwargs_str + "}"
        return kwargs_str


class Kwargs(KwargsToken):
    __slots__ = _fields = ("kwpairs", "type")
    defaults = {"kwpairs": {}, "type": Type("MAP<string, double>")}

    @staticmethod
    def _construct_kwpairs(pairs):
        return dict(pairs)

    @staticmethod
    def _construct_type(type_):
        return type_

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

    def _pythoncode(self, *args, **kwargs):
        kwargs_str = ", ".join([f"{key}={pycode(val)}" for key, val in self.kwpairs.items()])
        kwargs_str = kwargs_str.strip(", ")
        # kwargs_str = "{" + kwargs_str + "}"
        return kwargs_str


class CType(Type):

    """Token class but has modifier methods
    - as_vector
    - as_ndarray
    - ...etc
    """

    __slots__ = _fields = ("name",)
    defaults = {"name": "CType"}

    @staticmethod
    def _construct_name(name):
        return name


    def as_vector(self):
        return CType(f"vector<{self.name}>")


    def as_ndarray(self):
        return CType(f"py::array_t<{self.name}>")


    def as_map(self):
        return CType(f"map<string, {self.name}>")


    def as_const(self):
        return CType(f"const {self.name}")


    def as_ref(self):
        return CType(f"{self.name}&")


    def _ccode(self, *args, **kwargs):
        return self.name


    def _pythoncode(self, *args, **kwargs):
        return ""


class CTypes:
    bool = CType("bool")
    int = CType("int")
    float32 = CType("float")
    float64 = CType("double")
    void = CType("void")
    complex_ = CType("complex")


    @staticmethod
    def from_expr(expr) -> CType:
        """ Deduces type from an expression or a ``Symbol``.

        Parameters
        ==========

        expr : number or SymPy object
            The type will be deduced from type or properties.

        Examples
        ========

        >>> from sympy.codegen.ast import Type, integer, complex_
        >>> Type.from_expr(2) == integer
        True
        >>> from sympy import Symbol
        >>> Type.from_expr(Symbol('z', complex=True)) == complex_
        True
        >>> Type.from_expr(sum)  # doctest: +ELLIPSIS
        Traceback (most recent call last):
          ...
        ValueError: Could not deduce type from expr.

        Raises
        ======

        ValueError when type deduction fails.

        """
        if expr is None:
            return CTypes.void
        if isinstance(expr, Function):
            return expr.type
        if isinstance(expr, Kwargs) or isinstance(expr, Dict):
            return CTypes.float64.as_map()
        if isinstance(expr, MatrixSymbol):
            if expr.shape[0] > 1 and expr.shape[1] > 1:  # type:ignore
                raise Exception("Multidimensional Matrix shapes not yet supported.")
            return CTypes.float64.as_vector()
        if isinstance(expr, Matrix):
            return CTypes.float64.as_vector()
        if isinstance(expr, (float, Float)):
            return CTypes.float32
        if isinstance(expr, (int, Integer)) or getattr(expr, 'is_integer', False):
            return CTypes.int
        if getattr(expr, 'is_real', False):
            return CTypes.float64
        if isinstance(expr, complex) or getattr(expr, 'is_complex', False):
            return CTypes.complex_
        if (isinstance(expr, bool) or getattr(expr, 'is_Relational', False)
                or expr.assumptions0.get("boolean", False)):
            return CTypes.bool
        else:
            return CTypes.float64


class CodegenFunctionCall(FunctionCall, KwargsToken):
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
    defaults = {"function_args": Tuple(), "function_kwargs": Kwargs()}

    _construct_name = String
    _construct_function_args = staticmethod(lambda args: Tuple(*args))
    _construct_function_kwargs = staticmethod(lambda kwargs: Kwargs(kwargs))


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
        params_str = ", ".join([pycode(i) for i in self.function_args])
        if len(self.function_args) and len(self.function_kwargs):
            params_str += ", "
        if (kwargs_str := self._dict_to_kwargs_str(self.function_kwargs)):
            params_str += f"{kwargs_str}"
        return f"{self.name}({params_str})"


class CodeGenFunctionPrototype(FunctionPrototype):
    """ Represents a function prototype

    Allows the user to generate forward declaration in e.g. C/C++.

    Parameters
    ==========

    return_type : Type
    name : str
    parameters: iterable of Variable instances
    attrs : iterable of Attribute instances

    Examples
    ========

    >>> from sympy import ccode, symbols
    >>> from sympy.codegen.ast import real, FunctionPrototype
    >>> x, y = symbols('x y', real=True)
    >>> fp = FunctionPrototype(real, 'foo', [x, y])
    >>> ccode(fp)
    'double foo(double x, double y)'

    """

    __slots__ = ('return_type', 'name', 'parameters')
    _fields: tuple[str, ...] = __slots__ + Node._fields

    _construct_return_type = Type
    _construct_name = String

    @staticmethod
    def _construct_parameters(args):
        def _var(arg):
            if isinstance(arg, Declaration):
                return arg.variable
            elif isinstance(arg, Variable):
                return arg
            elif isinstance(arg, Kwargs):
                # for var in arg.kwpairs.values():
                #     return var
                # return Variable("a", type=Type("double"))
                return arg
            elif isinstance(arg, Dict):
                return arg
            else:
                return Variable.deduced(arg)
        return Tuple(*map(_var, args))

    @classmethod
    def from_FunctionDefinition(cls, func_def):
        if not isinstance(func_def, FunctionDefinition):
            raise TypeError("func_def is not an instance of FunctionDefinition")
        return cls(**func_def.kwargs(exclude=('body',)))


# class CodegenFunction:

#     def __init__(self, japl_function) -> None:
#         self.japl_function = japl_function
#         self.return_type = self.deduce_return_type()
#     #     f = CodegenFunctionProto(return_type=CTypes.double,
#     #                              name="func",
#     #                              parameters=[a, b])

#     @staticmethod
#     def deduce_return_type(expr) -> str:
#         # determine return shape
#         return_shape = Matrix([expr]).shape

#         if np.prod(return_shape) == 1:
#             primitive_type = CCodeGenerator._get_primitive_type(expr)  # type:ignore
#             type_str = primitive_type
#         else:
#             primitive_type = "double"
#             type_str = f"vector<{primitive_type}>"
#         return type_str


# # TESTS
# class func(JaplFunction):
#     pass

# avar = Variable(a, type=double)
# bvar = Variable(b, type=double)
# cvar = Variable(c, type=double)

# # Kwargs
# assert ccode(Kwargs()) == "{}"
# assert ccode(Kwargs({})) == "{}"
# assert ccode(Kwargs({"x": 1, 'y': 2.0})) == "{{\"x\", 1}, {\"y\", 2.0}}"
# assert ccode(Kwargs({"x": a, 'y': bvar})) == "{{\"x\", a}, {\"y\", b}}"

# # Function
# assert ccode(Func("func")) == "func()"
# assert ccode(Func("func", ())) == "func()"
# assert ccode(Func("func", (avar,))) == "func(a)"
# assert ccode(Func("func", (avar, bvar))) == "func(a, b)"
# assert ccode(Func("func", (), {"x": 1, "y": 2})) == "func(x=1, y=2)"
# assert ccode(Func("func", (avar, bvar), {"x": 1, "y": 2})) == "func(a, b, x=1, y=2)"
