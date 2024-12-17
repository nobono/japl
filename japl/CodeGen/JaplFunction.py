import itertools
import numpy as np
from typing import Any, Optional, Callable, Union
from sympy import MatrixSymbol
from sympy import Function
from sympy import Matrix
from sympy import Expr
from sympy import Number
from sympy import cse
from sympy.codegen.ast import numbered_symbols
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
from sympy.codegen.ast import NoneToken
from sympy.matrices import MatrixExpr
from japl.Util.Util import iter_type_check
from japl.CodeGen.Ast import CodeGenFunctionCall
from japl.CodeGen.Ast import CodeGenFunctionPrototype
from japl.CodeGen.Ast import CodeGenFunctionDefinition
from japl.CodeGen.Ast import JaplType
from japl.CodeGen.Ast import Kwargs
from japl.CodeGen.Globals import _STD_DUMMY_NAME
from japl.CodeGen.Globals import _STD_RETURN_NAME
from japl.CodeGen.Ast import get_lang_types
from japl.CodeGen.Ast import convert_symbols_to_variables
from japl.CodeGen.Util import is_empty_expr
from japl.CodeGen.Util import optimize_expression
from japl.BuildTools.BuildTools import parallel_subs



class JaplFunction(Function):

    """This class inherits from sympy.Function and allows
    keyword arguments."""

    __slots__ = ("name",
                 "function_kwargs",  # function kwargs
                 "function_args",  # function args
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
    return_type = JaplType()

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
        obj.function_kwargs = found_kwargs
        obj.function_args = found_args
        obj.function_call = CodeGenFunctionCall(obj.name, found_args, found_kwargs)

        # allow expr to be defined in class definition
        if hasattr(cls, "expr"):
            obj.expr = cls.expr
        else:
            obj.expr = Expr()
        return obj


    @staticmethod
    def get_std_args(function_args, code_type: str):
        """gets standard Model args [t, X, U, S, dt].
        standard args are defined dynamically in order to set
        the Variable type and respect the desired target language."""

        Types = get_lang_types(code_type)

        std_arg_names = ("t", "_X_arg", "_U_arg", "_S_arg", "dt")

        # ensure same number of params
        if len(function_args) != len(std_arg_names):
            raise Exception("exact standar function arguments must be specified"
                            "in order to use standard Model args in CodeGeneration."
                            "[t, state, input, static, dt].")

        # dynamically set std_arg types
        std_arg_symbols = (Symbol(std_arg_names[0]),
                           MatrixSymbol(std_arg_names[1], *function_args[1].shape),
                           MatrixSymbol(std_arg_names[2], *function_args[2].shape),
                           MatrixSymbol(std_arg_names[3], *function_args[3].shape),
                           Symbol(std_arg_names[4]))
        std_args = [Variable(std_arg_symbols[0], type=Types.float64.as_ref()),
                    Variable(std_arg_symbols[1], type=Types.float64.as_vector().as_ref()),
                    Variable(std_arg_symbols[2], type=Types.float64.as_vector().as_ref()),
                    Variable(std_arg_symbols[3], type=Types.float64.as_vector().as_ref()),
                    Variable(std_arg_symbols[4], type=Types.float64.as_ref())]

        return std_args


    @staticmethod
    def _to_codeblock(arg: Any, code_type: str) -> CodeBlock:
        """converts Symbolic expressions to codeblock. If an iterable of
        expressions are provided CodeBlock will be build recursively."""
        Types = get_lang_types(code_type=code_type)
        std_return_name = JaplFunction.std_return_name
        code_lines = []
        if isinstance(arg, MatrixExpr):  # captures MatrixSymbols / MatrixMul ...etc
            arg = arg.as_mutable()
        if isinstance(arg, Matrix):
            # for Matrix, declare return var and assign expressions.
            ret_symbol = MatrixSymbol(std_return_name, *arg.shape)
            return_type = Types.from_expr(ret_symbol)
            constructor = Types.float64.as_vector(shape=ret_symbol.shape)
            ret_var = Variable(std_return_name,
                               type=return_type,
                               value=constructor)
            code_lines += [ret_var.as_Declaration()]

            for idx, subexpr in zip(itertools.product(*[range(dim) for dim in arg.shape]), [*arg]):  # type:ignore # noqa
                lhs = ret_symbol[idx]
                code_lines += [Assignment(lhs, subexpr)]

            # return ret_var
            code_lines += [Return(ret_var)]
            return CodeBlock(*code_lines)
        elif isinstance(arg, Expr):
            if is_empty_expr(arg):  # case arg is empty expression (i.e. None, Expr())
                return CodeBlock()
            elif isinstance(arg, Symbol):
                var = Variable(arg, type=Types.from_expr(arg)).as_Declaration()
                return CodeBlock(var)
            else:
                ret_var = Variable(std_return_name, type=Types.from_expr(arg))
                if code_type != "py":  # python does not declare variables
                    code_lines += [Declaration(ret_var)]
                code_lines += [Assignment(ret_var.symbol, arg)]
                code_lines += [Return(ret_var)]
                return CodeBlock(*code_lines)
        # TODO for iter types:
        # auto-apply return if non exist
        # auto-apply declarations if non exist
        elif isinstance(arg, (tuple, list)):
            # return CodeBlock(*arg)
            for item in arg:
                code_block = JaplFunction._to_codeblock(item, code_type=code_type)
                code_lines += [code_block]
            return CodeBlock(*code_lines)
        elif isinstance(arg, CodeBlock):  # type:ignore
            return arg
        else:
            # raise Exception("Cannot conver expression to CodeBlock: unhandled case.")
            return arg


    def set_body(self, body: CodeBlock|Expr|Matrix, code_type: str):
        """sets function body. If expression provided,
        function body will be built when _build_function is called.
        Otherwise, function body is set directly."""
        if not isinstance(body, CodeBlock):
            self.expr = body
            self.body = self._to_codeblock(self.expr, code_type=code_type)
        else:
            self.body = body


    def _get_parameter_variables(self, code_type: str) -> list:
        Types = get_lang_types(code_type)  # get the appropriate Types class

        # convert function args
        dummy_symbol_gen = numbered_symbols(prefix=self.std_dummy_name)
        parameters = []
        arg_parameters = convert_symbols_to_variables(self.function_args,
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
        if self.function_kwargs:
            kwarg_name = next(dummy_symbol_gen).name
            kwarg_type = Types.float64.as_map().as_ref()
            kwarg_dummy_var = Kwargs(**self.function_kwargs).to_variable(kwarg_name, type=kwarg_type)
            parameters += [kwarg_dummy_var]
        return parameters


    def _build_proto(self, expr, code_type: str, use_std_args: bool = False):
        """Builds function prototype"""
        Types = get_lang_types(code_type)
        self.return_type = Types.from_expr(expr)
        # convert parameter Symbols to Variable
        parameters = self._get_parameter_variables(code_type)
        proto = CodeGenFunctionPrototype(return_type=self.return_type,
                                         name=self.name,
                                         parameters=parameters)
        self.function_proto = proto


    def _build_def(self,
                   expr,
                   code_type: str,
                   use_std_args: bool = False,
                   use_parallel: bool = True,
                   do_param_unpack: bool = True):
        """
        Builds function definition.

        Arguments
        =========
        expr:
            function symbolic expression. the function return type is derived from this.

        code_type: str
            target language for the code generation.

        use_std_args: bool
            set True to use japl.Model standardized arguments.

        use_parallel: bool
            set True to use parallel processing in expression optimization.

        do_param_unpack: bool
            set True to attempt:
                - the substitution of parameters expressions (for Matrix, map types)
                - the instantiation of symbols within Matrix & map type parameters
        """
        Types = get_lang_types(code_type)

        # ---------------------------------------------------------
        # optimize the expression
        # ---------------------------------------------------------
        if expr is None:
            expr = NoneToken()

        repl_assignments = []

        if use_parallel:
            replacements, expr = optimize_expression(expr)
        else:
            replacements, expr_simple = cse(expr)
            expr = expr_simple[0]  # type:ignore

        # create assignments from replacements tuple
        for lhs, rhs in replacements:
            Types = get_lang_types(code_type)
            lhs_var = Variable(lhs, type=Types.from_expr(lhs), value=rhs)
            repl_assignments += [Declaration(lhs_var)]
        # ---------------------------------------------------------

        if use_std_args:
            parameters = self.get_std_args(self.function_args, code_type)
        else:
            parameters = self._get_parameter_variables(code_type)

        # --------------------------------------------------------------------
        # get unpacks
        # pass iterable of symbols
        # pass iterable of expressions or MatrixSymbol
        #
        # NOTE: two scenarios of unpacking.
        # given an iterable of symbols, the array can be unpacked
        # and assigned to said symbols:
        #       given, Matrix([a, b])
        #       unpack:
        #           double a = _Dummy_arg[0]
        #           double b = _Dummy_arg[1]
        #
        # but given an iterable of expressions, direct instantiations
        # cannot be made.
        # --------------------------------------------------------------------
        arg_unpacks = []
        if do_param_unpack:
            expr, arg_unpacks = self._sub_array_of_expressions(target_expr=expr,
                                                               source_expr=parameters,
                                                               function_args=self.function_args,
                                                               function_kwargs=self.function_kwargs,
                                                               code_type=code_type)
        # --------------------------------------------------------------------

        self.return_type = Types.from_expr(expr)
        func_name = self.get_def_name()
        expr_codeblock = self._to_codeblock(expr, code_type=code_type)
        codeblock = CodeBlock(*arg_unpacks, *repl_assignments, *expr_codeblock.args)
        func_def = CodeGenFunctionDefinition(return_type=self.return_type,
                                             name=func_name,
                                             parameters=parameters,
                                             body=codeblock)
        self.function_def = func_def


    @staticmethod
    def _sub_array_of_expressions(target_expr,
                                  source_expr,
                                  function_args: list|tuple,
                                  function_kwargs: dict,
                                  code_type: str,
                                  do_unpack_symbols: bool = True):
        Types = get_lang_types(code_type)
        # --------------------------------------------------------------------
        # NOTE: experimental: special type accessor logic
        # For automatically accessing structures passed as function parameter.
        # and applying to the expression.
        #
        # this portion attempts to replace the expr with sub_expr of function parameters
        # that are matrix types.
        # --------------------------------------------------------------------
        # first process positional args
        arg_parameters = [i for i in source_expr if not i.type.is_map]
        kwarg_parameters = [i for i in source_expr if i.type.is_map]

        arg_unpacks = []

        # process for positional args
        # substitute function expr with expressions from args
        # arrays
        for (param, arg_symbol) in zip(arg_parameters, function_args):
            if param.type.is_array:
                subs_dict = {}
                arg_shape = [range(dim) for dim in arg_symbol.shape]
                for idx, sub_expr in zip(itertools.product(*arg_shape), arg_symbol):
                    # if MatrixElement contains only a Symbol, unpack this from the
                    # array-like arg:
                    #       "double x = _X_arg[0]"
                    if isinstance(sub_expr, Symbol) and do_unpack_symbols:
                        sub_expr_type = Types.from_expr(sub_expr).as_const()
                        arg_unpacks += [Declaration(Variable(sub_expr,
                                                             type=sub_expr_type,
                                                             value=param.symbol[idx]))]
                    else:
                        subs_dict.update({sub_expr: param.symbol[idx]})
                target_expr = target_expr.subs(subs_dict)

        # process for keyword-args
        # substitute function expr with expressions from kwargs
        # (map types: [dict, std::map])
        for i, arg_symbol in enumerate(function_kwargs.items()):
            i = min(i, len(kwarg_parameters) - 1)
            param = kwarg_parameters[i]
            if param.type.is_map:
                subs_dict = {}
                arg_name, arg_val = arg_symbol
                if not hasattr(arg_val, "__len__"):
                    arg_val = [arg_val]
                for sub_expr in arg_val:
                    map_symbol = Symbol(Types.map_get(param.symbol, arg_name))
                    # if MatrixElement contains only a Symbol, unpack this from the
                    # array-like arg:
                    #       "double x = _X_arg[0]"
                    if isinstance(sub_expr, Symbol) and do_unpack_symbols:
                        sub_expr_type = Types.from_expr(sub_expr).as_const()
                        arg_unpacks += [Declaration(Variable(sub_expr,
                                                             type=sub_expr_type,
                                                             value=map_symbol))]
                    else:
                        subs_dict.update({sub_expr: map_symbol})
                target_expr = target_expr.subs(subs_dict)

        return target_expr, arg_unpacks


    def get_def_name(self) -> str:
        return f"{self.class_name}::{self.name}" if self.class_name else self.name


    def get_proto(self):
        return self.function_proto


    def get_def(self):
        return self.function_def


    def _build_function(self, code_type: str,
                        use_parallel: bool = True,
                        use_std_args: bool = False,
                        do_param_unpack: bool = True):
        """dynamically builds JaplFunction prototype & definition
        using self.expr.

        This method calls _build_proto() and _build_def()."""
        self._build_proto(expr=self.expr,
                          code_type=code_type,
                          use_std_args=use_std_args)
        self._build_def(expr=self.expr,
                        code_type=code_type,
                        use_parallel=use_parallel,
                        use_std_args=use_std_args,
                        do_param_unpack=do_param_unpack)


    def _build(self, code_type: str):
        """method called by CodeGen.Builder used to build
        this AST object."""
        self._build_function(code_type=code_type)


    def __reduce__(self) -> str | tuple[Any, ...]:
        """defines serialization of class object"""
        state = {"kwargs": self.function_kwargs}
        return (self.__class__, (self.function_args, tuple(self.function_kwargs.items()),), state)


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
        new_kwargs = {key: val for key, val in zip(self.function_kwargs.keys(), args)}
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
                for key, val in self.function_kwargs.items()
                }
        # if subs changes the args, return new instance to apply
        # changes
        if new_kwargs != self.function_kwargs:
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
                new_kwargs = {key: val for key, val in zip(self.function_kwargs.keys(), args)}
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
        return (self._args, self.name, tuple(self.function_kwargs))


    def __contains__(self, item):
        # check if item is a free symbol in the expression
        in_args = any(item == arg for arg in self.args)
        in_free_symbols = item in self.free_symbols
        return in_args or in_free_symbols


    def _eval_derivative(self, sym):
        return
