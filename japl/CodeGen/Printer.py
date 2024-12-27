from sympy import Symbol
from sympy.printing.c import C99CodePrinter
from sympy.printing.pycode import PythonCodePrinter
from sympy.printing.octave import OctaveCodePrinter
from sympy.codegen.ast import FunctionDefinition
from sympy.codegen.ast import Assignment
from sympy.codegen.ast import Declaration
from sympy.codegen.ast import numbered_symbols
from sympy.codegen.ast import Variable
from sympy.codegen.ast import Comment
from sympy.codegen.ast import untyped
from sympy.core.function import Function
from japl.CodeGen.Ast import CTypes
from japl.CodeGen.Ast import get_lang_types
from japl.CodeGen.Ast import PyTypes
from japl.CodeGen.Ast import Kwargs
from japl.CodeGen.Ast import convert_symbols_to_variables
from japl.CodeGen.JaplFunction import JaplFunction
from japl.CodeGen.Globals import _STD_DUMMY_NAME
from japl.CodeGen.Globals import _STD_RETURN_NAME



class CCodeGenPrinter(C99CodePrinter):

    code_type: str = 'c'

    def _print_Function(self, expr):
        # ---------------------------------------------------------------
        # NOTE: CodeGen printer does not recognize _print_JaplFunction
        # JaplFunction print logic must be handled here.
        # ---------------------------------------------------------------
        if isinstance(expr, JaplFunction):
            return self._print_JaplFunction(expr)
        else:
            return super()._print_Function(expr)


    def _print_JaplFunction(self, expr: JaplFunction):
        # if here, printing the call of JaplFunction
        # ---------------------------------------------------------------
        # NOTE: this is not very clean but _get_parameter_variables must
        # know the codegen language type and thus parameters cannot be
        # defined in JaplFuction or CodeGenFunctionCall constructors.
        # ---------------------------------------------------------------
        return self._print_CodeGenFunctionCall(expr.function_call)


    def _print_CodeGenFunctionCall(self, expr):
        dummy_symbol_gen = numbered_symbols(prefix=_STD_DUMMY_NAME)
        parameters = convert_symbols_to_variables(expr.function_args,
                                                  code_type=self.code_type,
                                                  dummy_symbol_gen=dummy_symbol_gen)
        if not hasattr(parameters, "__len__"):
            parameters = [parameters]
        params_str = ", ".join([self._print(i.symbol) for i in parameters])  # type:ignore

        if len(expr.function_args) and len(expr.function_kwargs):
            params_str += ", "
        kwargs_list = []
        for key, val in expr.function_kwargs.items():
            kwargs_list += ["{" + f"\"{key}\", {self._print(val)}" + "}"]
        if kwargs_list:
            params_str += "{" + ", ".join(kwargs_list) + "}"
        return f"{expr.name}({params_str})"


    def _print_FunctionPrototype(self, expr):
        parameters = expr.parameters
        params_str = ", ".join([self._print(Declaration(i)) for i in parameters])  # type:ignore
        return_type_str = self._print(expr.return_type)

        # ----------------------------------------------------------------
        # NOTE: _print_FunctionDefinition uses this method to print itself
        # but "static" can only be in the function declaration
        # ----------------------------------------------------------------
        if not expr.is_static or isinstance(expr, FunctionDefinition):
            static_kw = ""
        else:
            static_kw = "static "

        return "%s%s %s(%s)" % (static_kw, return_type_str, expr.name, params_str)


    def _print_FunctionDefinition(self, expr):
        return "%s%s" % (self._print_FunctionPrototype(expr),
                         self._print_Scope(expr))


    def _print_ImmutableDenseMatrix(self, expr, name: str = ""):
        # NOTE: name keyword arg must be passed. This is typically
        # passed from the CodeGen Printer method "_print()".
        matrix_type = self._print(CTypes.from_expr(expr).as_vector().as_ref())
        return f"{matrix_type} {name}"


    def _print_MutableDenseMatrix(self, expr, name: str = ""):
        # NOTE: name keyword arg must be passed. This is typically
        # passed from the CodeGen Printer method "_print()".
        matrix_type = self._print(CTypes.from_expr(expr).as_vector().as_ref())
        return f"{matrix_type} {name}"


    def _print_Kwargs(self, expr):
        dummy_symbol_gen = numbered_symbols(prefix=_STD_DUMMY_NAME)
        kwargs_str = ""
        for key, val in expr.kwpairs.items():
            val_variable = convert_symbols_to_variables(val,
                                                        code_type=self.code_type,
                                                        dummy_symbol_gen=dummy_symbol_gen)
            kwargs_str += "{" + f"\"{key}\", " + self._print(val_variable) + "}, "
        kwargs_str = kwargs_str.strip(", ")
        kwargs_str = "{" + kwargs_str + "}"
        return kwargs_str


    def _print_Dict(self, expr):
        dict_str = ", ".join(["{" + f"\"{key}\", " + self._print(val) + "}"
                              for key, val in expr.kwpairs.items()])
        dict_str = dict_str.strip(", ")
        dict_str = "{" + dict_str + "}"
        return dict_str


class PyCodeGenPrinter(PythonCodePrinter):

    code_type: str = "py"
    tab = "    "

    def _print_Comment(self, expr):
        return f"# {str(expr)}"


    def _print_Piecewise(self, expr):
        # -----------------------------------------------------------------
        # handle case where Piecewise contains Assignments as the values:
        #
        # Assignments are created within _to_codeblock() but this will produce
        # invalid syntax for python code since assignments cannot be made within
        # conditional expressions.
        # -----------------------------------------------------------------
        flag_assign_case = False

        writes = []
        for i, (piece_expr, cond) in enumerate(expr.args):
            if isinstance(piece_expr, Assignment):
                flag_assign_case = True
                if cond == True:  # Default case (NOTE: comparison must be '==' not 'is') # noqa
                    writes.append("else:\n")
                    writes.append(self._indent_codestring(self._print(piece_expr)) + "\n")
                elif i == 0:
                    writes.append(f"if {cond}:\n")
                    writes.append(self._indent_codestring(self._print(piece_expr)) + "\n")
                else:
                    writes.append(f"elif {cond}:\n")
                    writes.append(self._indent_codestring(self._print(piece_expr)) + "\n")

        if flag_assign_case:
            return "".join(writes)
        else:
            return super()._print_Piecewise(expr)


    def _print_JaplClass(self, expr):
        Types = get_lang_types(self.code_type)
        name = expr.name
        parent = f"({expr.parent.text})" if expr.parent.text else ""
        writes = ["class %s%s:\n" % (name, parent)]
        for section_key, item in expr.members.items():
            if isinstance(item, Symbol):
                writes += [f"{self.tab}{section_key} = {item.name}\n"]
                writes += ["\n"]
            elif hasattr(item, "__len__"):
                section_comment = self.tab + self._print(Comment(section_key + "\n"))
                writes += [section_comment]
                for member in item:
                    if isinstance(member, JaplFunction):
                        member._build(code_type=self.code_type)
                        # --------------------------------------------------------
                        # NOTE: this is for python stubs but is not sufficient
                        # for generating python methods.
                        # --------------------------------------------------------
                        # function_proto_str = "def {name}{parameters}: ...".format(
                        #         name=self._print(member.name),
                        #         parameters=self._print(member.function_proto.parameters),
                        #         )
                        # writes += [self._indent_codestring(function_proto_str) + "\n"]
                        # --------------------------------------------------------
                    else:
                        type_hint = Types.from_expr(member)
                        writes += [f"{self.tab}{member.name}: {type_hint}\n"]
                        # raise Exception("unhandled case.")
                writes += ["\n"]
        return "".join(writes)


    def _print_Function(self, expr):
        # ---------------------------------------------------------------
        # NOTE: CodeGen printer does not recognize _print_JaplFunction
        # JaplFunction print logic must be handled here.
        #
        # super()._print_Function(expr) is not supported by PythonCodePrinter
        # ---------------------------------------------------------------
        if isinstance(expr, JaplFunction):
            return self._print_JaplFunction(expr)
        elif isinstance(expr, Function):
            print(expr)
            return super()._print_Function(expr)
        else:
            raise Exception(f"unhandled Function type. {expr.__class__}")


    def _print_JaplFunction(self, expr: JaplFunction):
        # if here, printing the call of JaplFunction
        # ---------------------------------------------------------------
        # NOTE: this is not very clean but _get_parameter_variables must
        # know the codegen language type and thus parameters cannot be
        # defined in JaplFuction or CodeGenFunctionCall constructors.
        # ---------------------------------------------------------------
        # parameters = expr._get_parameter_variables("py")  # first get params
        return self._print_CodeGenFunctionCall(expr.function_call)


    def _print_FunctionPrototype(self, expr):
        # NOTE: python has no prototypes
        return ""


    def _print_FunctionDefinition(self, fd):
        parameters = fd.parameters
        # convert param types to string
        if not hasattr(parameters, "__len__"):
            parameters = [parameters]
        params_str = ", ".join([self._print(i.symbol) for i in parameters])  # type:ignore

        body = '\n'.join((self._print(arg) for arg in fd.body))
        return "def {name}({parameters}):\n{body}".format(
            name=self._print(fd.name),
            parameters=params_str,
            body=self._indent_codestring(body)
        )


    def _print_CodeGenFunctionCall(self, expr):
        # get parameters and convert to printable types
        dummy_symbol_gen = numbered_symbols(prefix=_STD_DUMMY_NAME)
        parameters = convert_symbols_to_variables(expr.function_args,
                                                  code_type=self.code_type,
                                                  dummy_symbol_gen=dummy_symbol_gen)
        # convert param types to string
        if not hasattr(parameters, "__len__"):
            parameters = [parameters]
        params_str = ", ".join([self._print(i.symbol) for i in parameters])  # type:ignore

        # append function kwargs
        if len(expr.function_args) and len(expr.function_kwargs):
            params_str += ", "

        # convert function_kwargs [dict] to keyword args syntax.
        kwargs_writes = []
        for key, val in expr.function_kwargs.items():
            kwargs_writes += [f"{key}={self._print(val)}"]
        params_str += ", ".join(kwargs_writes)

        return "%s(%s)" % (expr.name, params_str)


    def _print_Kwargs(self, expr):
        kwargs_str = ", ".join([f"{key}={self._print(val)}" for key, val in expr.kwpairs.items()])
        kwargs_str = kwargs_str.strip(", ")
        return kwargs_str


    def _print_Dict(self, expr):
        kwargs_str = ", ".join([f"{key}: {self._print(val)}" for key, val in expr.kwpairs.items()])
        kwargs_str = kwargs_str.strip(", ")
        kwargs_str = "{" + kwargs_str + "}"
        return kwargs_str


class OctaveCodeGenPrinter(OctaveCodePrinter):

    code_type: str = "octave"
    tab = '\t'

    def _print_Declaration(self, expr):
        if expr.variable.value:
            lhs = expr.variable
            rhs = expr.variable.value
            return self._print(Assignment(lhs, rhs))
        else:
            return ""


    def _print_Return(self, expr):
        return ""


    def _print_Comment(self, expr):
        return f"% {str(expr)}"

    # def _print_JaplClass(self, expr):
    #     Types = get_lang_types(self.code_type)
    #     name = expr.name
    #     parent = f"({expr.parent.text})" if expr.parent.text else ""
    #     writes = ["class %s%s:\n" % (name, parent)]
    #     for section_key, iter in expr.members.items():
    #         if not len(iter):
    #             continue
    #         section_comment = self._print(Comment(section_key + "\n"))
    #         writes += [section_comment]
    #         for member in iter:
    #             type_hint = Types.from_expr(member)
    #             writes += [f"\t{member.name}: {type_hint}\n"]
    #         writes += ["\n"]
    #     return "".join(writes)


    def _print_Function(self, expr):
        # ---------------------------------------------------------------
        # NOTE: CodeGen printer does not recognize _print_JaplFunction
        # JaplFunction print logic must be handled here.
        #
        # super()._print_Function(expr) is not supported by PythonCodePrinter
        # ---------------------------------------------------------------
        return self._print_JaplFunction(expr)


    # def _print_CodeGenFunctionDefinition(self, expr):
    #     return "FUNC"


    def _print_JaplFunction(self, expr: JaplFunction):
        # if here, printing the call of JaplFunction
        # ---------------------------------------------------------------
        # NOTE: this is not very clean but _get_parameter_variables must
        # know the codegen language type and thus parameters cannot be
        # defined in JaplFuction or CodeGenFunctionCall constructors.
        # ---------------------------------------------------------------
        # parameters = expr._get_parameter_variables("py")  # first get params
        return self._print_CodeGenFunctionCall(expr.function_call)


    def _print_FunctionPrototype(self, expr):
        # NOTE: octave has no prototypes
        return ""


    def _print_FunctionDefinition(self, fd):
        parameters = fd.parameters
        # convert param types to string
        if not hasattr(parameters, "__len__"):
            parameters = [parameters]
        params_str = ", ".join([self._print(i.symbol) for i in parameters])  # type:ignore

        body = '\n'.join((self._print(arg) for arg in fd.body))
        return "function [{returns}] = {name}({parameters})\n{body}end".format(
            returns=_STD_RETURN_NAME,
            name=self._print(fd.name),
            parameters=params_str,
            body=self._indent_codestring(body)
        )


    def _print_CodeGenFunctionCall(self, expr):
        # get parameters and convert to printable types
        dummy_symbol_gen = numbered_symbols(prefix=_STD_DUMMY_NAME)
        parameters = convert_symbols_to_variables(expr.function_args,
                                                  code_type=self.code_type,
                                                  dummy_symbol_gen=dummy_symbol_gen)
        # convert param types to string
        if not hasattr(parameters, "__len__"):
            parameters = [parameters]
        params_str = ", ".join([self._print(i.symbol) for i in parameters])  # type:ignore

        # append function kwargs
        if len(expr.function_args) and len(expr.function_kwargs):
            params_str += ", "
        params_str += expr._dict_to_kwargs_str(expr.function_kwargs)

        return "%s(%s)" % (expr.name, params_str)


    # def _print_Kwargs(self, expr):
    #     kwargs_str = ", ".join([f"{key}={self._print(val)}" for key, val in expr.kwpairs.items()])
    #     kwargs_str = kwargs_str.strip(", ")
    #     return kwargs_str


    # def _print_Dict(self, expr):
    #     kwargs_str = ", ".join([f"{key}: {self._print(val)}" for key, val in expr.kwpairs.items()])
    #     kwargs_str = kwargs_str.strip(", ")
    #     kwargs_str = "{" + kwargs_str + "}"
    #     return kwargs_str


    def _indent_codestring(self, codestring):
        return '\n'.join([self.tab + line for line in codestring.split('\n')])


def ccode(expr, **settings):
    printer = CCodeGenPrinter(settings)
    return printer.doprint(expr)


def pycode(expr, **settings):
    printer = PyCodeGenPrinter(settings)
    return printer.doprint(expr)


def octave_code(expr, **settings):
    printer = OctaveCodeGenPrinter(settings)
    return printer.doprint(expr)
