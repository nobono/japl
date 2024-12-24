import numpy as np
from typing import Optional
from typing import Generator
from sympy.codegen.ast import FunctionCall
from sympy.codegen.ast import FunctionPrototype
from sympy.codegen.ast import FunctionDefinition
from sympy.codegen.ast import Declaration
from sympy.codegen.ast import Variable
from sympy.codegen.ast import String
from sympy.codegen.ast import Token
from sympy.codegen.ast import Tuple
from sympy.codegen.ast import Type
from sympy.codegen.ast import Node
from sympy.codegen.ast import Expr
from sympy.tensor import Array
from sympy.tensor.array.expressions import ArraySymbol
from sympy import true, false
from sympy.core.function import Function
from sympy.core.numbers import Number
from sympy import Float, Integer, Matrix
from sympy import Symbol
from sympy import MatrixSymbol
from sympy.matrices import MutableDenseMatrix
from japl.CodeGen.Globals import _STD_DUMMY_NAME
from japl.CodeGen.Util import is_empty_expr



def get_lang_types(code_type: str):
    if code_type.lower() in ["py", "python"]:
        return PyTypes
    elif code_type.lower() in ["c", "cpp", "c++"]:
        return CTypes
    elif code_type.lower() in ["octave", "oct", "matlab", "m"]:
        return OctTypes
    else:
        raise Exception(f"codegen for {code_type} not avaialable.")


def flatten_matrix_to_array(expr: MatrixSymbol|Matrix, name: str):
    # --------------------------------------------------------------------
    # NOTE:
    # handle MatrixSymbol which should be flattened to ArraySymbol
    # if MatrixSymbol has shape (..., N, ...) then convert to ArraySymbol
    # of shape (N,).
    # this is so accessing flattened array paramter is correct because
    # accessing MatryxSymbol like: `a[0]` --> `a[0, 0]` and accessing
    # an ArraySymbol like: `a[0]` --> `a[0]`
    # --------------------------------------------------------------------
    shape = np.prod(expr.shape)  # type:ignore
    if isinstance(expr, MatrixSymbol):
        return ArraySymbol(name, (shape,))
    elif isinstance(expr, Matrix):  # type:ignore
        return Array(expr.flat())
    else:
        raise Exception("unhandled case.")


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
            param_var = Variable(param_name, type=param_type)
        elif isinstance(param, Symbol):
            param_name = getattr(param, "name", None)
            if param_name is None:
                param_name = str(param)
            param_var = Variable(param_name, type=param_type)
        elif isinstance(param, MatrixSymbol):
            # param_name = param.name
            param_var = Variable(param, type=param_type)
        elif isinstance(param, MutableDenseMatrix):
            param_name = next(dummy_symbol_gen)
            param_var = Variable(MatrixSymbol(param_name, *param.shape), type=param_type)
        else:
            param_name = next(dummy_symbol_gen)
            param_var = Variable(param_name, type=param_type)
        ret += (param_var,)

    # return shape same as input
    if len(ret) == 1:
        return ret[0]
    else:
        return ret


class JaplType(Type):
    __slots__ = _fields = ("name", "is_array", "is_map", "is_ref", "is_const")
    defaults = {"name": "JaplType",
                "is_array": false,
                "is_map": false,
                "is_ref": false,
                "is_const": false}

    _construct_name = String

    @staticmethod
    def _construct_is_array(val):
        return val


    @staticmethod
    def _construct_is_map(val):
        return val


    @staticmethod
    def _construct_is_ref(val):
        return val


    def as_vector(self, *args, **kwargs):
        return JaplType("", is_array=true)


    def as_ndarray(self, *args, **kwargs):
        return JaplType("", is_array=true)


    def as_map(self):
        return JaplType("", is_map=true)


    def as_const(self):
        return JaplType("", is_const=true)


    def as_ref(self):
        return JaplType("", is_ref=true)


    def _ccode(self, *args, **kwargs):
        return str(self.name)


    def _pythoncode(self, *args, **kwargs):
        return str(self.name)


    def _octave(self, *args, **kwargs):
        return str(self.name)


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


class OctType(JaplType):

    """Token class but has modifier methods
    - as_vector
    - ...etc
    """

    def as_vector(self, params: list|tuple = [], shape: list|tuple = []):
        if params:
            _params = ", ".join([str(i) for i in params])
            return OctType(f"zeros({_params})", is_array=true)  # type:ignore
        elif shape:
            _params = ", ".join([str(i) for i in shape])
            return OctType(f"zeros({_params})", is_array=true)  # type:ignore
        else:
            return OctType("()", is_array=true)  # type:ignore


    def as_map(self):
        return OctType("",
                      is_array=self.is_array,
                      is_map=true,
                      is_ref=self.is_ref,
                      is_const=self.is_const)


    def as_const(self):
        return OctType("",
                      is_array=self.is_array,
                      is_map=self.is_map,
                      is_ref=self.is_ref,
                      is_const=true)


    def as_ref(self):
        return OctType("",
                      is_array=self.is_array,
                      is_map=self.is_map,
                      is_ref=true,
                      is_const=self.is_const)


class PyType(JaplType):

    """Token class but has modifier methods
    - as_vector
    - as_ndarray
    - ...etc
    """

    # __slots__ = _fields = JaplType.__slots__ + ("type_hint",)
    # defaults = {**JaplType.defaults, "type_hint": JaplType("")}

    # @staticmethod
    # def _construct_type_hint(val):
    #     return val


    def as_vector(self, params: list|tuple = [], shape: list|tuple = []):
        if params:
            _params = ", ".join([str(i) for i in params])
            return PyType(f"np.empty({_params})", is_array=true)  # type:ignore
        elif shape:
            _params = ", ".join([str(i) for i in shape])
            _params = f"({_params})"
            return PyType(f"np.empty({_params})", is_array=true)  # type:ignore
        else:
            return PyType("np.empty()", is_array=true)  # type:ignore


    def as_ndarray(self, params: list|tuple = [], shape: list|tuple = []):
        if params:
            _params = ", ".join([str(i) for i in params])
            return PyType(f"np.empty({_params})", is_array=true)  # type:ignore
        elif shape:
            _params = ", ".join([str(i) for i in shape])
            _params = f"({_params})"
            return PyType(f"np.empty({_params})", is_array=true)  # type:ignore
        else:
            return PyType("np.empty()", is_array=true)  # type:ignore


    def as_map(self):
        return PyType("",
                      is_array=self.is_array,
                      is_map=true,
                      is_ref=self.is_ref,
                      is_const=self.is_const)


    def as_const(self):
        return PyType("",
                      is_array=self.is_array,
                      is_map=self.is_map,
                      is_ref=self.is_ref,
                      is_const=true)


    def as_ref(self):
        return PyType("",
                      is_array=self.is_array,
                      is_map=self.is_map,
                      is_ref=true,
                      is_const=self.is_const)


class CType(JaplType):

    """Token class but has modifier methods
    - as_vector
    - as_ndarray
    - ...etc
    """

    def as_vector(self, params: list|tuple = [], shape: list|tuple = []):
        # return CType(f"vector<{self.name}>", is_array=true)  # type:ignore
        if params:
            _params = ", ".join([str(i) for i in params])
            return CType(f"vector<{self.name}>({_params})", is_array=true)  # type:ignore
        elif shape:
            size = np.prod(shape)
            return CType(f"vector<{self.name}>({size})", is_array=true)  # type:ignore
        else:
            return CType(f"vector<{self.name}>", is_array=true)  # type:ignore


    def as_ndarray(self, params: list|tuple = [], shape: list|tuple = []):
        # return CType(f"py::array_t<{self.name}>", is_array=true)  # type:ignore
        if params:
            _params = ", ".join([str(i) for i in params])
            return CType(f"py::array_t<{self.name}>({_params})", is_array=true)  # type:ignore
        elif shape:
            size = np.prod(shape)
            return CType(f"py::array_t<{self.name}>({size})", is_array=true)  # type:ignore
        else:
            return CType(f"py::array_t<{self.name}>", is_array=true)  # type:ignore


    def as_map(self):
        return CType(f"map<string, {self.name}>",
                     is_array=self.is_array,
                     is_map=true,
                     is_ref=self.is_ref,
                     is_const=self.is_const)


    def as_const(self):
        return CType(f"const {self.name}",
                     is_array=self.is_array,
                     is_map=self.is_map,
                     is_ref=self.is_ref,
                     is_const=true)


    def as_ref(self):
        return CType(f"{self.name}&",
                     is_array=self.is_array,
                     is_map=self.is_map,
                     is_ref=true,
                     is_const=self.is_const)


class OctTypes(JaplTypes):
    bool = OctType("")
    int = OctType("")
    float32 = OctType("")
    float64 = OctType("")
    void = OctType("")
    complex_ = OctType("")


    @staticmethod
    def from_expr(expr):
        """ Deduces type from an expression or a ``Symbol``.
        """
        if is_empty_expr(expr):
            return OctTypes.void
        if isinstance(expr, Function):
            return expr.type
        if isinstance(expr, Kwargs) or isinstance(expr, Dict):
            return OctTypes.float64.as_map()
        if isinstance(expr, MatrixSymbol):
            if expr.shape[0] > 1 and expr.shape[1] > 1:  # type:ignore
                raise Exception("Multidimensional Matrix shapes not yet supported.")
            return OctTypes.float64.as_vector()
        if isinstance(expr, Matrix):
            return OctTypes.float64.as_vector()
        if isinstance(expr, (float, Float)):
            return OctTypes.float64
        if isinstance(expr, (int, Integer)) or getattr(expr, 'is_integer', False):
            return OctTypes.int
        if getattr(expr, 'is_real', False):
            return OctTypes.float64
        if isinstance(expr, complex) or getattr(expr, 'is_complex', False):
            return OctTypes.complex_
        if (isinstance(expr, bool) or getattr(expr, 'is_Relational', False)
                or expr.assumptions0.get("boolean", False)):
            return OctTypes.bool
        else:
            return OctTypes.float64


    @staticmethod
    def map_get(name: str, val):
        """helper / convenience method for accessing \"map\" types
        for different languages."""
        return f"{name}.{val}"


class PyTypes(JaplTypes):
    bool = PyType("bool")
    int = PyType("int")
    float32 = PyType("")
    float64 = PyType("float")
    void = PyType("None")
    complex_ = PyType("")


    @staticmethod
    def from_expr(expr):
        """ Deduces type from an expression or a ``Symbol``.
        """
        if is_empty_expr(expr):
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


    @staticmethod
    def map_get(name: str, val):
        """helper / convenience method for accessing \"map\" types
        for different languages."""
        return f"{(name)}[\"{val}\"]"


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


    @staticmethod
    def map_get(name: str, val):
        """helper / convenience method for accessing \"map\" types
        for different languages."""
        return f"{(name)}[\"{val}\"]"


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


    def __new__(cls, *args, **kwargs):
        obj = super().__new__(cls, *args, **kwargs)
        return obj


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

    __slots__ = ('return_type', 'name', 'parameters', 'is_static')
    _fields: tuple[str, ...] = __slots__ + Node._fields
    defaults = {**FunctionPrototype.defaults, "is_static": false}

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

    __slots__ = (*FunctionDefinition.__slots__, 'is_static')
    _fields: tuple[str, ...] = FunctionDefinition._fields + ('is_static',)
    defaults = {**FunctionDefinition.defaults, 'is_static': false}

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
