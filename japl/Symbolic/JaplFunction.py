import numpy as np
from typing import Any, Optional, Callable
from sympy import ccode, pycode
from sympy import Function
from sympy import Matrix
from sympy import Expr
from sympy.codegen.ast import FunctionDefinition
from sympy.codegen.ast import CodeBlock
from sympy.codegen.ast import FunctionPrototype
from sympy.codegen.ast import Variable
from sympy.codegen.ast import Type
from sympy.codegen.ast import Tuple
from japl.Util.Util import iter_type_check
from japl.Symbolic.Ast import CodegenFunctionCall
from japl.Symbolic.Ast import CodeGenFunctionPrototype
from japl.Symbolic.Ast import CType
from japl.Symbolic.Ast import CTypes
from japl.Symbolic.Ast import Kwargs



class CodeGenUtil:

    CODES = ("py", "c")

    @staticmethod
    def get_lang_types(code_type: str):
        if code_type not in CodeGenUtil.CODES:
            raise Exception(f"codegen for {code_type} not avaialable.")
        elif code_type == "py":
            raise Exception("Type does not apply for generating python code.")
        elif code_type == "c":
            return CTypes


class JaplFunction(Function):

    """This class inherits from sympy.Function and allows
    keyword arguments."""

    __slots__ = ("name",
                 "kwargs",  # function kwargs
                 "fargs",  # function args
                 "codegen_function_call",
                 "codegen_function_def",
                 "codegen_function_proto")

    parent = ""
    codegen_function_call: CodegenFunctionCall
    codegen_function_def: FunctionDefinition
    codegen_function_proto: FunctionPrototype

    # @classmethod
    # def eval(cls, *args):
    #     """kwargs are not used in eval by default; augmenting here."""
    #     # return super().eval(*args)
    #     return None

    def __new__(cls, *args, **kwargs):
        # if args is tuple[tuple, ...], then __new__ is being
        # called from __setstate__ which passes keyword
        # arguments into *args as a tuple[tuple, ...]
        all_args = ()
        found_args = ()
        found_kwargs = {}
        for arg in args:
            if iter_type_check(arg, tuple[tuple]):
                found_kwargs.update(dict(arg))
            elif isinstance(arg, tuple):
                found_args += arg
            else:
                found_args += (arg,)
        found_kwargs.update(kwargs)

        all_args += found_args
        all_args += tuple(found_kwargs.values())

        obj = super().__new__(cls, *all_args)

        # attatch kwargs to the object
        if obj.parent:
            obj.name = f"{obj.parent}.{str(cls)}"  # for "class.method" naming
        else:
            obj.name = str(cls)  # function name is name of class
        obj.kwargs = found_kwargs
        obj.fargs = found_args

        # codegen objects
        obj.codegen_function_call = CodegenFunctionCall(obj.name, found_args, found_kwargs)
        # obj.codegen_function_proto = FunctionPrototype(return_type=)
        return obj


    def set_body(self, body: CodeBlock|list|tuple):
        if isinstance(body, CodeBlock):
            pass
        elif isinstance(body, tuple) or isinstance(body, list):  # type:ignore
            pass
        else:
            raise Exception(f"could not set function body for {self.name}")


    def _get_parameter_variables(self, code_type: str) -> tuple[Variable, ...]:
        """converts parameter Symbols to Variables"""
        Types = CodeGenUtil.get_lang_types(code_type)
        kwarg_params = ()
        arg_params = ()
        if self.kwargs:
            kwarg_type = Types.float64.as_map().as_ref()
            kwarg_params = (Variable("kwargs", type=kwarg_type),)
        for param in self.fargs:
            param_type = Types.from_expr(param).as_ref()
            param_name = getattr(param, "name", None)
            if param_name is None:
                param_name = str(param)
            arg_params += (Variable(param_name, type=param_type),)
        return arg_params + kwarg_params


    def _build_proto(self, expr, code_type: str):
        Types = CodeGenUtil.get_lang_types(code_type)
        return_type = Types.from_expr(expr)
        # convert parameter Symbols to Variable
        parameters = self._get_parameter_variables(code_type)
        proto = CodeGenFunctionPrototype(return_type=return_type,
                                         name=self.name,
                                         parameters=parameters)
        self.codegen_function_proto = proto


    def __reduce__(self) -> str | tuple[Any, ...]:
        """defines serialization of class object"""
        state = {"kwargs": self.kwargs}
        return (self.__class__, (self.fargs, tuple(self.kwargs.items()),), state)


    def __repr__(self) -> str:
        return self.__str__()


    def __str__(self) -> str:
        return str(self.codegen_function_call)  # type:ignore


    def set_parent(self, parent: str):
        self.parent = parent
        self.name = f"{parent}.{self.name}"
        self.codegen_function_call.name = self.name


    def _pythoncode(self, *args, **kwargs):
        """string representation of object when using sympy.pycode()
        for python code generation"""
        return pycode(self.codegen_function_call)


    def _ccode(self, *args, **kwargs):
        """string representation of object when using sympy.ccode()
        for c-code generate"""
        return ccode(self.codegen_function_call)


    def _sympystr(self, printer):
        """string representation of object used in sympy string formatting."""
        return self.__str__()


    def func(self, *args):  # type:ignore
        """This overrides @property func. which is used to rebuild
        the object. This is used in sympy cse."""
        new_kwargs = {key: val for key, val in zip(self.kwargs.keys(), args)}
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

    # -----------------------------------------
    # Codegen Methods
    # -----------------------------------------

    # def _get_def_parameters(self)
