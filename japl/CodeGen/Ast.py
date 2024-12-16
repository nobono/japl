from typing import Optional
from typing import Generator
from sympy.codegen.ast import numbered_symbols
from sympy.codegen.ast import FunctionCall
from sympy.codegen.ast import FunctionPrototype
from sympy.codegen.ast import FunctionDefinition
from sympy.codegen.ast import Declaration
from sympy.codegen.ast import CodeBlock
from sympy.codegen.ast import Variable
from sympy.codegen.ast import String
from sympy.codegen.ast import Token
from sympy.codegen.ast import Tuple
from sympy.codegen.ast import Type
from sympy.codegen.ast import Node
from sympy.codegen.ast import Expr
from sympy import true, false
from sympy.core.function import Function
from sympy.core.numbers import Number
from sympy import Float, Integer, Matrix
from sympy import Symbol
from sympy import MatrixSymbol
from japl.CodeGen.Globals import _STD_DUMMY_NAME
from japl.CodeGen.Util import is_empty_expr



__CODES__ = ("py", "c")


def get_lang_types(code_type: str):
    if code_type not in __CODES__:
        raise Exception(f"codegen for {code_type} not avaialable.")
    elif code_type == "py":
        # raise Exception("Type does not apply for generating python code.")
        return PyTypes
    elif code_type == "c":
        return CTypes


def convert_symbols_to_variables(params, code_type: str, dummy_symbol_gen: Generator) -> Variable|tuple:
    """This handles conversions between Symbolic types (Symbol, Matrix, ...etc)
    and converts them to Variable types necessary for code generation.
    Variable types contain name and type information."""
    ret = ()
    is_iterable_of_args = isinstance(params, (Tuple, list, tuple))
    if not is_iterable_of_args:
        params = [params]

    # create dummy symbols generator if none provided
    # this is so dummy variables can be generated within or
    # outside of another code-generating scope.
    # if dummy_symbol_gen is None:
    #     dummy_symbol_gen = numbered_symbols(prefix=_STD_DUMMY_NAME)

    for param in params:
        Types = get_lang_types(code_type)
        param_type = Types.from_expr(param).as_ref()

        if isinstance(param, (Number, int, float)):
            param_name = str(param)
        elif isinstance(param, Symbol):
            param_name = getattr(param, "name", None)
            if param_name is None:
                param_name = str(param)
        elif isinstance(param, MatrixSymbol):
            param_name = param.name
        else:
            param_name = next(dummy_symbol_gen)
        ret += (Variable(param_name, type=param_type),)

    # return shape same as input
    if len(ret) == 1:
        return ret[0]
    else:
        return ret


class JaplType(Type):
    __slots__ = _fields = ("name", "is_array")
    defaults = {"name": "JaplType", "is_array": false}

    _construct_name = String

    @staticmethod
    def _construct_is_array(val):
        return val


    def as_vector(self):
        return JaplType("")


    def as_ndarray(self):
        return JaplType("")


    def as_map(self):
        return JaplType("")


    def as_const(self):
        return JaplType("")


    def as_ref(self):
        return JaplType("")


    def _ccode(self, *args, **kwargs):
        return self.name


    def _pythoncode(self, *args, **kwargs):
        return ""


class JaplTypes:

    """This is a Types base class."""

    bool: JaplType
    int: JaplType
    float32: JaplType
    float64: JaplType
    void: JaplType
    complex_: JaplType

    @staticmethod
    def from_expr(expr):
        return JaplType()


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

    """Token but accepts kwargs input.
    inputs are symbolic but are converted to Variable types."""

    def __new__(cls, *args, **kwargs):
        # -----------------------------------------------------
        # NOTE: this overload allows passing of python kwargs
        #   "func(1, 2, a=a, b=b)"
        # that is also compatible with sympy Token
        # -----------------------------------------------------
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
        # group args not passed as tuple
        # this assumed first arg is name and everything
        # between name and kwargs is args tuple.
        # -----------------------------------------------------
        # NOTE: ensure args is passed as tuple of:
        #   (name, (*args), kwargs[dict])
        # -----------------------------------------------------
        if len(args) >= 3:
            if isinstance(args[1], tuple):
                _args = args
            else:
                _args = (args[0], tuple(args[1:-1]), args[-1])
        else:
            _args = args
        obj = super().__new__(cls, *_args, **kwargs_passthrough)
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


class Kwargs(KwargsToken):
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

    def to_variable(self, name: str, type: JaplType) -> Variable:
        """since Kwargs is not a Type we need a
        way to convert Kwargs to printable type (Variable)."""
        return Variable(name, type=type)


class PyType(JaplType):

    """Token class but has modifier methods
    - as_vector
    - as_ndarray
    - ...etc
    """

    def as_vector(self):
        return PyType("", is_array=true)  # type:ignore


    def as_ndarray(self):
        return PyType("", is_array=true)  # type:ignore


    def as_map(self):
        return PyType("")


    def as_const(self):
        return PyType("")


    def as_ref(self):
        return PyType("")


class CType(JaplType):

    """Token class but has modifier methods
    - as_vector
    - as_ndarray
    - ...etc
    """

    def as_vector(self):
        return CType(f"vector<{self.name}>", is_array=true)  # type:ignore


    def as_ndarray(self):
        return CType(f"py::array_t<{self.name}>", is_array=true)  # type:ignore


    def as_map(self):
        return CType(f"map<string, {self.name}>")


    def as_const(self):
        return CType(f"const {self.name}")


    def as_ref(self):
        return CType(f"{self.name}&")


class PyTypes(JaplTypes):
    bool = PyType("")
    int = PyType("")
    float32 = PyType("")
    float64 = PyType("")
    void = PyType("")
    complex_ = PyType("")


    @staticmethod
    def from_expr(expr):
        """ Deduces type from an expression or a ``Symbol``.
        """
        if expr is None:
            return PyTypes.void
        if isinstance(expr, Function):
            return expr.type
        if isinstance(expr, Kwargs) or isinstance(expr, Dict):
            return PyTypes.float64.as_map()
        if isinstance(expr, MatrixSymbol):
            if expr.shape[0] > 1 and expr.shape[1] > 1:  # type:ignore
                raise Exception("Multidimensional Matrix shapes not yet supported.")
            return PyTypes.float64.as_vector()
        if isinstance(expr, Matrix):
            return PyTypes.float64.as_vector()
        if isinstance(expr, (float, Float)):
            return PyTypes.float64
        if isinstance(expr, (int, Integer)) or getattr(expr, 'is_integer', False):
            return PyTypes.int
        if getattr(expr, 'is_real', False):
            return PyTypes.float64
        if isinstance(expr, complex) or getattr(expr, 'is_complex', False):
            return PyTypes.complex_
        if (isinstance(expr, bool) or getattr(expr, 'is_Relational', False)
                or expr.assumptions0.get("boolean", False)):
            return PyTypes.bool
        else:
            return PyTypes.float64


class CTypes(JaplTypes):
    bool = CType("bool")
    int = CType("int")
    float32 = CType("float")
    float64 = CType("double")
    void = CType("void")
    complex_ = CType("complex")


    @staticmethod
    def from_expr(expr):
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
        if is_empty_expr(expr):
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
            return CTypes.float64
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


class CodeGenFunctionCall(FunctionCall, KwargsToken):
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
        """
        returns string format of a dictionary as keyword args.

        for example:
            the dict: {'a': 1, 'b': 2} returns \"a=1, b=2\"
        """
        kwargs_list = []
        for key, val in dkwargs.items():
            kwargs_list += [f"{key}={val}"]
        return ", ".join(kwargs_list)


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



class CodeGenFunctionDefinition(FunctionDefinition):
    """ Represents a function definition in the code.

    Parameters
    ==========

    return_type : Type
    name : str
    parameters: iterable of Variable instances
    body : CodeBlock or iterable
    attrs : iterable of Attribute instances

    Examples
    ========

    >>> from sympy import ccode, symbols
    >>> from sympy.codegen.ast import real, FunctionPrototype
    >>> x, y = symbols('x y', real=True)
    >>> fp = FunctionPrototype(real, 'foo', [x, y])
    >>> ccode(fp)
    'double foo(double x, double y)'
    >>> from sympy.codegen.ast import FunctionDefinition, Return
    >>> body = [Return(x*y)]
    >>> fd = FunctionDefinition.from_FunctionPrototype(fp, body)
    >>> print(ccode(fd))
    double foo(double x, double y){
        return x*y;
    }
    """
    @staticmethod
    def _construct_return_type(arg):
        return arg

    @staticmethod
    def _construct_parameters(args):
        def _var(arg):
            if isinstance(arg, Declaration):
                return arg.variable
            elif isinstance(arg, Variable):
                return arg
            elif isinstance(arg, Kwargs):
                return arg
            elif isinstance(arg, Dict):
                return arg
            else:
                return Variable.deduced(arg)
        return Tuple(*map(_var, args))
