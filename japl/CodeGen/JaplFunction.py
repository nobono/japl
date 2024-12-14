import itertools
from typing import Any, Optional, Callable
from sympy import MatrixSymbol
from sympy import Function
from sympy import Matrix
from sympy import Expr
from sympy import Number
from sympy.codegen.ast import numbered_symbols
from sympy.codegen.ast import FunctionDefinition
from sympy.codegen.ast import FunctionPrototype
from sympy.codegen.ast import Declaration
from sympy.codegen.ast import CodeBlock
from sympy.codegen.ast import Assignment
from sympy.codegen.ast import Variable
from sympy.codegen.ast import Return
from sympy.codegen.ast import Symbol
from sympy.codegen.ast import Dummy
from sympy.codegen.ast import Type
from sympy.codegen.ast import Tuple
from sympy.matrices import MatrixExpr
from japl.Util.Util import iter_type_check
from japl.CodeGen.Ast import CodeGenFunctionCall
from japl.CodeGen.Ast import CodeGenFunctionPrototype
from japl.CodeGen.Ast import CodeGenFunctionDefinition
from japl.CodeGen.Ast import CType
from japl.CodeGen.Ast import CTypes
from japl.CodeGen.Ast import PyTypes
from japl.CodeGen.Ast import Kwargs
from japl.CodeGen.Globals import _STD_DUMMY_NAME
from japl.CodeGen.Globals import _STD_RETURN_NAME
from japl.CodeGen.Ast import get_lang_types
from japl.CodeGen.Ast import convert_symbols_to_variables
from japl.CodeGen.Util import is_empty_expr



class JaplFunction(Function):

    """This class inherits from sympy.Function and allows
    keyword arguments."""

    __slots__ = ("name",
                 "kwargs",  # function kwargs
                 "fargs",  # function args
                 # "expr",    # function expression
                 "function_call",
                 "function_def",
                 "function_proto",
                 "function_body")

    parent = ""
    class_name = ""
    description = ""
    function_call: CodeGenFunctionCall
    function_def: CodeGenFunctionDefinition
    function_proto: FunctionPrototype
    function_body: CodeBlock
    body = CodeBlock()
    expr: Expr|Matrix
    type = CTypes.float64

    std_return_name = _STD_RETURN_NAME
    std_dummy_name = _STD_DUMMY_NAME

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

        # printable members
        obj.kwargs = found_kwargs
        obj.fargs = found_args
        obj.function_call = CodeGenFunctionCall(obj.name, found_args, found_kwargs)
        obj.expr = Expr()
        return obj


    @staticmethod
    def _to_codeblock(arg: Expr|Matrix|CodeBlock|list|tuple) -> CodeBlock:
        """converts Symbolic expressions to codeblock. If an iterable of
        expressions are provided CodeBlock will be build recursively."""
        std_return_name = JaplFunction.std_return_name
        code_lines = []
        if isinstance(arg, MatrixExpr):  # captures MatrixSymbols / MatrixMul ...etc
            arg = arg.as_mutable()
        if isinstance(arg, Matrix):
            # for Matrix, declare return var and assign expressions.
            ret_symbol = MatrixSymbol(std_return_name, *arg.shape)
            return_type = CTypes.from_expr(ret_symbol)
            ret_var = Variable(std_return_name, type=return_type)
            code_lines += [ret_var.as_Declaration()]

            for idx, subexpr in zip(itertools.product(*[range(dim) for dim in arg.shape]), [*arg]):
                lhs = ret_symbol[idx]
                code_lines += [Assignment(lhs, subexpr)]

            # return ret_var
            code_lines += [Return(ret_var)]
            return CodeBlock(*code_lines)
        elif isinstance(arg, Expr):
            if is_empty_expr(arg):  # case arg is empty expression (i.e. None, Expr())
                return CodeBlock()
            elif isinstance(arg, Symbol):
                var = Variable(arg, type=CTypes.from_expr(arg)).as_Declaration()
                return CodeBlock(var)
            else:
                ret_var = Variable(std_return_name, type=CTypes.from_expr(arg))
                code_lines += [ret_var.as_Declaration()]
                code_lines += [Assignment(ret_var.symbol, arg)]
                code_lines += [Return(ret_var)]
                return CodeBlock(*code_lines)
        # TODO for iter types:
        # auto-apply return if non exist
        # auto-apply declarations if non exist
        elif isinstance(arg, tuple):
            # return CodeBlock(*arg)
            for item in arg:
                code_lines += [JaplFunction._to_codeblock(item)]
            return CodeBlock(*code_lines)
        elif isinstance(arg, list):
            # return CodeBlock(*arg)
            for item in arg:
                code_lines += [JaplFunction._to_codeblock(item)]
            return CodeBlock(*code_lines)
        elif isinstance(arg, CodeBlock):  # type:ignore
            return arg
        else:
            # raise Exception("Cannot conver expression to CodeBlock: unhandled case.")
            return arg


    def set_body(self, body: CodeBlock|Expr):
        # set body directly if CodeBlock.
        # otherwise, prime expr for body creation.
        if isinstance(body, CodeBlock):
            self.body = self._to_codeblock(body)
        else:
            self.expr = body


    def _get_parameter_variables(self, code_type: str) -> list:
        Types = get_lang_types(code_type)  # get the appropriate Types class

        # convert function args
        dummy_symbol_gen = numbered_symbols(prefix=self.std_dummy_name)
        parameters = []
        arg_parameters = convert_symbols_to_variables(self.fargs,
                                                      code_type=code_type,
                                                      dummy_symbol_gen=dummy_symbol_gen)
        if hasattr(arg_parameters, "__len__"):
            parameters += list(arg_parameters)  # convert to list
        else:
            parameters += [arg_parameters]

        # convert_symbols_to_variables passes through Literal types,
        # but for function prototype we want to use a dummy variable.
        # replace Literal types with Dummy variables.
        for i, param in enumerate(parameters):
            if isinstance(param.symbol, (Number, int, float)):
                dummy_symbol = next(dummy_symbol_gen)
                dummy_type = Types.from_expr(dummy_symbol).as_ref()
                parameters[i] = Variable(dummy_symbol, type=dummy_type)

        # convert function kwargs
        if self.kwargs:
            kwarg_name = next(dummy_symbol_gen).name
            kwarg_type = Types.float64.as_map().as_ref()
            kwarg_dummy_var = Kwargs(**self.kwargs).to_variable(kwarg_name, type=kwarg_type)
            parameters += [kwarg_dummy_var]
        return parameters


    def _build_proto(self, expr, code_type: str):
        """Builds function prototype"""
        Types = get_lang_types(code_type)
        return_type = Types.from_expr(expr)
        # convert parameter Symbols to Variable
        parameters = self._get_parameter_variables(code_type)
        proto = CodeGenFunctionPrototype(return_type=return_type,
                                         name=self.name,
                                         parameters=parameters)
        self.function_proto = proto


    def _build_def(self, expr, code_type: str):
        """Build function definition"""
        # func_proto = self.function_proto
        # func_def = FunctionDefinition.from_FunctionPrototype(func_proto=func_proto,
        #                                                      body=codeblock)
        Types = get_lang_types(code_type)
        return_type = Types.from_expr(expr)
        codeblock = self._to_codeblock(expr)
        func_name = self.get_def_name()
        parameters = self._get_parameter_variables(code_type)
        func_def = CodeGenFunctionDefinition(return_type=return_type,
                                             name=func_name,
                                             parameters=parameters,
                                             body=codeblock)
        self.function_def = func_def


    def get_def_name(self) -> str:
        return f"{self.class_name}::{self.name}" if self.class_name else self.name


    def get_proto(self):
        return self.function_proto


    def get_def(self):
        return self.function_def


    def _build_function(self, code_type: str):
        """dynamically builds JaplFunction prototype & definition
        using self.expr.

        This method calls _build_proto() and _build_def()."""
        self._build_proto(expr=self.expr, code_type=code_type)
        self._build_def(expr=self.expr, code_type=code_type)


    def _build(self, code_type: str):
        """method called by CodeGen.Builder used to build
        this AST object."""
        self._build_function(code_type=code_type)


    def __reduce__(self) -> str | tuple[Any, ...]:
        """defines serialization of class object"""
        state = {"kwargs": self.kwargs}
        return (self.__class__, (self.fargs, tuple(self.kwargs.items()),), state)


    def __repr__(self) -> str:
        return self.__str__()


    def __str__(self) -> str:
        return str(self.function_call)  # type:ignore


    def set_parent(self, parent: str):
        self.parent = parent
        self.name = f"{parent}.{self.name}"
        self.function_call.name = self.name


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
