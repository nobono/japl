import os
import dill
from typing import Callable, Optional
import numpy as np
from enum import Enum
# from scipy.sparse import csr_matrix
# from scipy.sparse._csr import csr_matrix as Tcsr_matrix
# from sympy.matrices.expressions.matexpr import MatrixElement
from sympy import Matrix, MatrixSymbol, Symbol, Expr, Function
from sympy import nan as sp_nan
from sympy import simplify
from sympy.codegen.ast import NoneToken

from japl.Model.StateRegister import StateRegister
from japl.Util.Desym import Desym

from japl.BuildTools.DirectUpdate import DirectUpdateSymbol
from japl.BuildTools.CCodeGenerator import CCodeGenerator
from japl.BuildTools import BuildTools

from japl.CodeGen.CodeGen import cache_py_function
from japl.CodeGen import FileBuilder
from japl.CodeGen import CFileBuilder
from japl.CodeGen import ModuleBuilder
from japl.CodeGen import CodeGenerator
from japl.CodeGen import JaplFunction
from japl.CodeGen import JaplClass
from japl.CodeGen import pycode

# ---------------------------------------------------

# TODO in this file:
# deprecate the following:
#   - Desym
#   - self.vars (used only to configure Desym)
#   - __process_direct_state_updates()
#   - dont need to hold expressions just the functions from buildtools?
#   - self._type?
#   - self.modules? (only used for Desym)
#   - self._iX_reference? (unused i think)
#   - self._sym_references? (only only for StateSpace)



class Model:

    """
    This class is a Model interface for SimObjects

    ---

    """

    state_vars: Matrix
    input_vars: Matrix
    static_vars: Matrix
    state_updates: Callable
    input_updates: Callable
    dynamics: Callable

    def __init__(self, **kwargs) -> None:
        self._dtype = np.float64
        self.state_register = StateRegister()
        self.input_register = StateRegister()
        self.static_register = StateRegister()
        self.dynamics_expr = NoneToken()
        self.modules: dict = {}
        self.dt_var = Symbol("")
        self.vars: tuple = ()
        self.state_dim = 0
        self.input_dim = 0
        self.static_dim = 0
        self.state_updates_expr: Matrix
        self.input_updates_expr: Matrix
        self.input_function: Callable
        self.pre_update_functions: list[Callable] = []
        self.post_update_functions: list[Callable] = []

        if not hasattr(self, "state_vars"):
            self.state_vars = Matrix([])
        if not hasattr(self, "input_vars"):
            self.input_vars = Matrix([])
        if not hasattr(self, "static_vars"):
            self.static_vars = Matrix([])

        # init registers
        self.set_state(self.state_vars)  # NOTE: will convert any Function to Symbol
        self.set_input(self.input_vars)  # NOTE: will convert any Function to Symbol
        self.set_static(self.static_vars)

        # data array dims
        self.state_dim = len(self.state_vars)
        self.input_dim = len(self.input_vars)
        self.static_dim = len(self.static_vars)

        # namespace dict for cached functions
        self._namespace = {}

        # proxy state array updated at each step
        # ***************************************
        # this is ony used during animation or
        # when ODE solver is used step-by-step
        # ***************************************
        self._iX_reference = np.array([])

        # info on what matrix slice hold what state ref.
        # this is only used with StateSpace model types
        # to allow sympy symbols within the A, B matrices.
        self._sym_references: list[dict] = []


    def has_input_function(self) -> bool:
        return hasattr(self, "input_function")


    def has_state_updates(self) -> bool:
        return hasattr(self, "state_updates")


    def has_input_updates(self) -> bool:
        return hasattr(self, "input_updates")


    def has_dynamics(self) -> bool:
        return hasattr(self, "dynamics")


    def has_input_updates_expr(self) -> bool:
        # must be (expr == None)
        # not (expr is None)
        return not (self.input_updates_expr == None)  # noqa


    def has_state_updates_expr(self) -> bool:
        # must be (expr == None)
        # not (expr is None)
        return not (self.state_updates_expr == None)  # noqa


    def has_dynamics_expr(self) -> bool:
        # must be (expr == None)
        # not (expr is None)
        return not (self.dynamics_expr == None)  # noqa


    @DeprecationWarning
    def __set_current_state(self, X: np.ndarray):
        # TODO this only used right now to access quaternion in
        # Plotter. maybe we can dispense with these somehow...
        """Setter for Model state reference array \"Model._iX_reference\".
        This method should only be called by Model.step()."""
        self._iX_reference = X


    def get_current_state(self) -> np.ndarray:
        # TODO this only used right now to access quaternion in
        # Plotter. maybe we can dispense with these somehow...
        """Getter for Model state reference array. Used to access the
        state array between time steps outside of the Sim class."""
        return self._iX_reference.copy()


    def cache_py_function(self, func: JaplFunction, use_parallel: bool = True) -> Callable:
        func._build("py", use_parallel=use_parallel)
        cache_py_function(func.function_def, namespace=self._namespace)
        return self._namespace[func.name]


    def get_cached_function(self, name: str) -> Callable:
        ret = self._namespace.get(name, None)
        if ret is None:
            raise Exception()
        else:
            return ret


    @classmethod
    def from_function(cls,
                      dt_var: Symbol,
                      state_vars: list|tuple|Matrix,
                      input_vars: list|tuple|Matrix,
                      static_vars: list|tuple|Matrix = [],
                      dynamics_func: Optional[Callable] = None,
                      state_update_func: Optional[Callable] = None,
                      input_update_func: Optional[Callable] = None) -> "Model":
        """This method initializes a Model from a callable function.
        The provided function must have the following signature:

            func(t, X, U, S, dt)

        where,

            - 't' is the sim-time
            - 'X' is the current state
            - 'U', is the current inputs
            - 'S' is the static variables array
            - 'dt' is the time step.

        -------------------------------------------------------------------
        -- Arguments
        -------------------------------------------------------------------
        -- dt_var - symbolic dt variable
        -- state_vars - iterable of symbolic state variables
        -- input_vars - iterable of symbolic input variables
        -- func - callable function which returns the state dynamics
        -------------------------------------------------------------------
        -- Returns
        -------------------------------------------------------------------
        -- model - the initialized Model
        -------------------------------------------------------------------
        """
        # TODO initialize model.state_dim somehow ...
        model = cls()
        model.set_state(state_vars)
        model.set_input(input_vars)
        model.set_static(static_vars)
        model.state_vars = model.state_register.get_vars()
        model.input_vars = model.input_register.get_vars()
        model.static_vars = model.static_register.get_vars()
        model.dt_var = dt_var
        model.vars = (Symbol("t"), model.state_vars, model.input_vars, model.static_vars, dt_var)
        model.state_dim = len(model.state_vars)
        model.input_dim = len(model.input_vars)
        model.static_dim = len(model.static_vars)
        if dynamics_func:
            model.dynamics = dynamics_func
        if state_update_func:
            model.state_updates = state_update_func
        if input_update_func:
            model.input_updates = input_update_func
        if (not dynamics_func) and (not state_update_func):
            raise Exception("Both dynamics_func and state_update_func cannot be undefined.")
        return model


    def __set_modules(self, modules: dict|list[dict]) -> dict:
        ret = {}
        if isinstance(modules, list) or isinstance(modules, tuple):
            for module in modules:
                ret.update(module)
        else:
            ret.update(modules)
        return ret


    @classmethod
    def from_expression(cls,
                        dt_var: Symbol,
                        state_vars: list|tuple|Matrix,
                        input_vars: list|tuple|Matrix,
                        dynamics_expr: Expr|Matrix|MatrixSymbol = Matrix([]),
                        static_vars: list|tuple|Matrix = [],
                        definitions: tuple = (),
                        modules: dict|list[dict] = {},
                        use_multiprocess_build: bool = True) -> "Model":
        """This method initializes a Model from a symbolic expression.
        a Sympy expression can be passed which then is lambdified
        (see Sympy.lambdify) with computational optimization (see Sympy.cse).

        -------------------------------------------------------------------
        **Arguments**

        ``dt_var`` : Symbol
        :   symbolic dt

        ``state_vars`` : Iterable[Symbol]
        :   iterable of symbolic state variables

        ``input_vars`` : Iterable[Symbol]
        :   iterable of symbolic input variables

        ``dynamics_expr`` : Expr
        :   Sympy symbolic dynamics expression

        ``static_vars`` : Iterable[Symbol]
        :   iterable of symbolic static variables

        ``modules`` : Optional
        :   pass custom library to Desym (see sympy.lambdify)

        -------------------------------------------------------------------
        **Returns**

        ``cls`` : Model
        :   the initialized Model

        > NOTE: static variables are symbolic variables which are initialized
        but not stored as part of the state or input arrays.

        -------------------------------------------------------------------
        **Examples**

        ```python
        >>> from sympy import symbols
        >>> dt, a, b, c, d, e = symbols("dt, a, b, c, d, e")
        >>> state = Matrix([a, b, c])
        >>> input = Matrix([d, e])
        >>> model = Model.from_expression(dt, state, input)
        ```
        -------------------------------------------------------------------
        """
        # first build model using provided definitions
        (state_vars,
         input_vars,
         dynamics_expr,
         static_vars,
         state_updates_expr,
         input_updates_expr) = BuildTools.build_model(Matrix(state_vars),
                                                      Matrix(input_vars),
                                                      Matrix(dynamics_expr),
                                                      definitions,
                                                      static=Matrix(static_vars),
                                                      use_multiprocess_build=use_multiprocess_build)
        model = cls()
        model.modules = model.__set_modules(modules)
        model.set_state(state_vars)  # NOTE: will convert any Function to Symbol
        model.set_input(input_vars)  # NOTE: will convert any Function to Symbol
        model.set_static(static_vars)
        model.state_vars = model.state_register.get_vars()
        model.input_vars = model.input_register.get_vars()
        model.static_vars = model.static_register.get_vars()
        model.dt_var = dt_var
        model.vars = (Symbol("t"), model.state_vars, model.input_vars, model.static_vars, dt_var)
        model.dynamics_expr = dynamics_expr
        model.state_updates_expr = state_updates_expr
        model.input_updates_expr = input_updates_expr
        model.state_dim = len(model.state_vars)
        model.input_dim = len(model.input_vars)
        model.static_dim = len(model.static_vars)
        return model


    def _pre_sim_checks(self) -> bool:
        # run state-register checks
        self.state_register._pre_sim_checks()
        self.input_register._pre_sim_checks()
        return True


    @DeprecationWarning
    def _get_sim_func_call_list(self) -> list[Callable]:
        """Returns list of Callables which simulate this model.
        List includes function calls from child models gathered recursively.

        -------------------------------------------------------------------
        **Update sequence**

        - pre_update_functions
        - (user) input_function
        >
        - input_updates
        - state_updates
        - dynamics

        - post_update_functions
        -------------------------------------------------------------------
        """
        calls = []
        calls += self.pre_update_functions
        if getattr(self, "input_function", None) is not None:
            calls += [self.input_function]
        if getattr(self, "input_updates", None) is not None:
            calls += [self.input_updates]
        if getattr(self, "state_updates", None) is not None:
            calls += [self.state_updates]
        if getattr(self, "dynamics", None) is not None:
            calls += [self.dynamics]
        calls += self.post_update_functions
        return calls


    def get_static_id(self, names: str|list[str]) -> int|list[int]:
        """This method get the sympy variable associated with the provided
        name. variables must first be added to the StateRegister. If a list
        of state names are provided, then a list of corresponding state ids
        will be returned.

        If sympy MatrixElement has been registered in the state e.g. 'x[i, j]',
        then the provided name 'x' will return all indices of that particular
        state.

        -------------------------------------------------------------------
        **Arguments**

        ``name`` : str | list[str]
        :   name of the symbolic state variable
            name or a list of symbolic state variable names

        **Returns**

        ``int | list[int]``
        :   the index of the state variable in the
            state array or list of indices.
        -------------------------------------------------------------------
        """
        return self.static_register.get_ids(names)


    def get_state_id(self, names: str|list[str]) -> int|list[int]:
        """This method get the sympy variable associated with the provided
        name. variables must first be added to the StateRegister. If a list
        of state names are provided, then a list of corresponding state ids
        will be returned.

        If sympy MatrixElement has been registered in the state e.g. 'x[i, j]',
        then the provided name 'x' will return all indices of that particular
        state.

        -------------------------------------------------------------------
        **Arguments**

        ``name``: str | list[str]
        :   name of the symbolic state variable
            name or a list of symbolic state variable names

        **Returns**

        ``int | list[int]``
        :   the index of the state variable in the
            state array or list of indices.
        -------------------------------------------------------------------
        """
        return self.state_register.get_ids(names)


    def get_input_id(self, names: str|list[str]) -> int|list[int]:
        """This method get the sympy variable associated with the provided
        name. variables must first be added to the StateRegister. If a list
        of state names are provided, then a list of corresponding input ids
        will be returned.

        If sympy MatrixElement has been registered in the input e.g. 'x[i, j]',
        then the provided name 'x' will return all indices of that particular
        input.

        -------------------------------------------------------------------
        **Arguments**

        ``name`` : str | list[str]
        :   name of the symbolic input variable
            name or a list of symbolic input variable names

        **Returns**

        ``int | list[int]``
        :   the index of the input variable in the
            input array or list of indices.
        -------------------------------------------------------------------
        """
        return self.input_register.get_ids(names)


    def set_state(self, state_vars: tuple|list|Matrix, labels: Optional[list|tuple] = None):
        """This method initializes the StateRegister attribute of the Model.

        -------------------------------------------------------------------
        **Arguments**

        ``state_vars`` : Iterable[Symbol]
        :   iterable of symbolic state variables

        ``labels`` : Optional[str]
        :   iterable of labels that may be used by the
            Plotter class. order labels must correspond to order
            of state_vars.

        **Returns**

        ``Symbol``
        :   the symbolic object of the state variable
        -------------------------------------------------------------------
        """
        return self.state_register.set(vars=state_vars, labels=labels)


    def set_input(self, input_vars: tuple|list|Matrix, labels: Optional[list|tuple] = None):
        """This method initializes the (inputs) StateRegister attribute of the Model.

        -------------------------------------------------------------------
        **Arguments**

        ``input_vars`` : Iterable[Symbol]
        :   iterable of symbolic input variables

        ``labels`` : Optional[str]
        :   iterable of labels that may be used by the
             Plotter class. order labels must correspond to order
             of input_vars.

        **Returns**

        ``Symbol``
        :   the symbolic object of the state variable
        -------------------------------------------------------------------
        """
        return self.input_register.set(vars=input_vars, labels=labels)


    def set_static(self, static_vars: tuple|list|Matrix, labels: Optional[list|tuple] = None):
        """This method initializes the StateRegister attribute of the Model.

        -------------------------------------------------------------------
        **Arguments**

        ``static_vars`` : Iterable[Symbol]
        :   iterable of symbolic state variables

        ``labels`` : (optional)
        :   iterable of labels that may be used by the
            Plotter class. order labels must correspond to order
            of state_vars.

        **Returns**

        ``Symbol``
        :   the symbolic object of the static variable
        -------------------------------------------------------------------
        """
        return self.static_register.set(vars=static_vars, labels=labels)


    def get_sym(self, name: str) -> Symbol:
        """This method gets the symbolic variable associated
        with the provided name.

        -------------------------------------------------------------------
        **Arguments**

        ``name`` : str
        :   name of the symbolic state variable

        **Returns**

        Symbol
        :   the symbolic object of the state variable
        -------------------------------------------------------------------
        """
        return self.state_register.get_sym(name)


    def set_input_function(self, func: Callable) -> None:
        """This method takes a function and inserts it before the
        Model's direct input updates. The outputs of this function
        feed directly into the models inputs.

        NOTE that if the Model has any defined direct input updates,
        the user's changes to the input array may be modified or
        over-written.

        -------------------------------------------------------------------
        **Arguments**

        ``func``
        :   Callable function with the signature:
                func(t, X, U, S, dt, ...) -> U
            where X is the state array, U is the input array,
            S is the static variable array.

            this function must return the input array U
            to have any affect on the model.
        -------------------------------------------------------------------
        """
        self.input_function = func


    def set_pre_update_functions(self, funcs: list[Callable]|Callable) -> None:
        """This method takes a function and inserts it before the
        Model's update step.

        -------------------------------------------------------------------
        **Arguments**

        funcs
        :   list of Callable functions with the signature:
                func(t, X, U, S, dt, ...)
            where X is the state array, U is the input array,
            S is the static variable array.
        -------------------------------------------------------------------
        """
        if isinstance(funcs, list):
            self.pre_update_functions += funcs
        else:
            self.pre_update_functions += [funcs]


    def set_post_update_functions(self, funcs: list[Callable]|Callable) -> None:
        """This method takes a function and inserts it after the
        Model's update step.

        -------------------------------------------------------------------
        **Arguments**

        funcs
        :   list of Callable functions with the signature:
                func(t, X, U, S, dt, ...)
            where X is the state array, U is the input array,
            S is the static variable array.
        -------------------------------------------------------------------
        """
        if isinstance(funcs, list):
            self.post_update_functions += funcs
        else:
            self.post_update_functions += [funcs]


    def _get_independent_symbols(self) -> list[Symbol]:
        """Returns the independent symbols of the model.
        (the symbols which must be initialized)"""
        # -----------------------------------------
        # NOTE: Experiemental:
        # try to identify independent symbols in expr
        # for model initialization.
        # -----------------------------------------
        free_symbols = self.state_updates_expr.free_symbols
        independent_symbols = [i for i in self.state_vars if i in free_symbols]  # sort by state position
        independent_symbols += self.static_vars
        return independent_symbols


    # def save(self, path: str, name: str):
    #     """This method saves a model to a .japl file."""
    #     ext = ".japl"
    #     save_path = os.path.join(path, name + ext)
    #     print("saving model to path:", path)
    #     # remove existing file if exists
    #     if os.path.isfile(save_path):
    #         os.remove(save_path)
    #     with open(save_path, 'ab') as file:
    #         data = (self.modules,
    #                 self.state_vars,
    #                 self.input_vars,
    #                 self.static_vars,
    #                 self.dt_var,
    #                 self.vars,
    #                 self.state_dim,
    #                 self.input_dim,
    #                 self.static_dim,
    #                 self.dynamics,
    #                 self.dynamics_expr,
    #                 self.state_updates_expr,
    #                 self.input_updates_expr,
    #                 self.state_updates,
    #                 self.input_updates,
    #                 self.user_input_function)
    #         dill.dump(data, file)


    # @classmethod
    # def from_file(cls, path: str, modules: dict|list[dict] = {}) -> "Model":
    #     """This method loads a Model from a .japl file. Models are saved
    #     as a tuple of class attributes. Loading a model from a file unpacks
    #     said attributes and initializes a Model object.

    #     NOTE:
    #         currently, modules must be passed to this method and reloaded
    #         into the model in order for Aero & MassProp data tables to work.
    #         This is because a symbolic model may be created with empty or
    #         temporary data tables that are baked into the model output file.
    #         data tables are then loaded at runtime.
    #     """
    #     with open(path, 'rb') as file:
    #         data = dill.load(file)
    #     obj = cls()
    #     (obj.modules,
    #      obj.state_vars,
    #      obj.input_vars,
    #      obj.static_vars,
    #      obj.dt_var,
    #      obj.vars,
    #      obj.state_dim,
    #      obj.input_dim,
    #      obj.static_dim,
    #      obj.dynamics,
    #      obj.dynamics_expr,
    #      obj.state_updates_expr,
    #      obj.input_updates_expr,
    #      obj.state_updates,
    #      obj.input_updates,
    #      obj.user_input_function) = data
    #     if obj._type == ModelType.Symbolic:
    #         # the init from .from_expression()
    #         # this is to re-init model with updated modules
    #         model = cls()
    #         model._type = ModelType.Symbolic
    #         model.modules = model.__set_modules(modules)
    #         model.set_state(obj.state_vars)  # NOTE: will convert any Function to Symbol
    #         model.set_input(obj.input_vars)  # NOTE: will convert any Function to Symbol
    #         model.set_static(obj.static_vars)
    #         model.state_vars = model.state_register.get_vars()
    #         model.input_vars = model.input_register.get_vars()
    #         model.static_vars = model.static_register.get_vars()
    #         model.dt_var = obj.dt_var
    #         model.vars = (Symbol("t"), model.state_vars, model.input_vars, model.static_vars, obj.dt_var)
    #         model.dynamics_expr = obj.dynamics_expr
    #         model.state_updates_expr = obj.state_updates_expr
    #         model.input_updates_expr = obj.input_updates_expr

    #         # if modules are being reloaded / updates, re-build the lambdify'd
    #         # functions
    #         if modules:
    #             model.state_updates = model.__process_direct_state_updates(obj.state_updates_expr)
    #             model.input_updates = model.__process_direct_state_updates(obj.input_updates_expr)
    #         else:
    #             model.state_updates = obj.state_updates
    #             model.input_updates = obj.input_updates
    #         model.state_dim = len(model.state_vars)
    #         model.input_dim = len(model.input_vars)
    #         model.static_dim = len(model.static_vars)

    #         # create lambdified function from symbolic expression
    #         # dyn_vars = (Symbol("t"),) + model.vars
    #         if modules:
    #             match obj.dynamics_expr.__class__():  # type:ignore
    #                 case Expr():
    #                     model.dynamics = model.__process_direct_state_updates(obj.dynamics_expr)
    #                 case Matrix():
    #                     model.dynamics = model.__process_direct_state_updates(obj.dynamics_expr)
    #                 case MatrixSymbol():
    #                     model.dynamics = model.__process_direct_state_updates(obj.dynamics_expr)
    #                 case _:
    #                     raise Exception("function provided is not Callable.")
    #         else:
    #             model.dynamics = obj.dynamics

    #         return model
    #     else:
    #         # initialize the state & input registers
    #         obj.set_state(obj.state_vars)
    #         obj.set_input(obj.input_vars)
    #         obj.set_static(obj.static_vars)
    #     return obj


    def cache_build(self, use_parallel: bool = True):
        """Builds core JaplFunctions (input_updates, state_updates, dynamics)
        and caches function in self._namespace. This is to provide Model functionality
        without having to use code generation to output a python module."""
        t = Symbol("t", real=True)
        dt = Symbol("dt", real=True)
        params = [t, self.state_vars, self.input_vars, self.static_vars, dt]

        if self.has_input_updates_expr():
            class input_updates(JaplFunction):  # noqa
                expr = self.input_updates_expr
            self.input_updates = self.cache_py_function(func=input_updates(*params),
                                                        use_parallel=use_parallel)

        if self.has_state_updates_expr():
            class state_updates(JaplFunction):  # noqa
                expr = self.state_updates_expr
            self.state_updates = self.cache_py_function(func=state_updates(*params),
                                                        use_parallel=use_parallel)

        if self.has_dynamics_expr():
            class dynamics(JaplFunction):  # noqa
                expr = self.dynamics_expr
            self.dynamics = self.cache_py_function(func=dynamics(*params),
                                                   use_parallel=use_parallel)


    def create_c_module(self, name: str, path: str = "./"):
        """
        Creates a c-lang module.

        -------------------------------------------------------------------
        **Arguments**

        ``name`` : str
        :   name of the c-module to be created

        ``path`` : Optional[str]
        :   the output path to save the module to. (default is current path `"./"`)
        -------------------------------------------------------------------
        """
        filename = f"{name}.cpp"
        t = Symbol("t", real=True)
        dt = Symbol("dt", real=True)
        params = [t, self.state_vars, self.input_vars, self.static_vars, dt]

        class dynamics(JaplFunction):  # noqa
            class_name = "Model"
            # is_static = True
            expr = self.dynamics_expr
        class state_updates(JaplFunction):  # noqa
            class_name = "Model"
            # is_static = True
            expr = self.state_updates_expr
        class input_updates(JaplFunction):  # noqa
            class_name = "Model"
            # is_static = True
            expr = self.input_updates_expr

        sim_methods = [dynamics(*params),
                       state_updates(*params),
                       input_updates(*params)]

        file_builder = CFileBuilder(filename, sim_methods)

        tab = "    "

        # settings symbolic arrays
        # ---------------------------------------------------------------------------
        state_var_names = [getattr(i, "name") for i in self.state_vars]
        input_var_names = [getattr(i, "name") for i in self.input_vars]  # type:ignore
        static_var_names = [getattr(i, "name") for i in self.static_vars]  # type:ignore
        if not state_var_names:
            state_vars_member = Symbol("Matrix([])")
        else:
            state_vars_member = Symbol(f"Matrix({state_var_names})")
        if not input_var_names:
            input_vars_member = Symbol("Matrix([])")
        else:
            input_vars_member = Symbol(f"Matrix({input_var_names})")
        if not static_var_names:
            static_vars_member = Symbol("Matrix([])")
        else:
            static_vars_member = Symbol(f"Matrix({static_var_names})")

        # Model class file
        # TODO JaplClass is sloppy
        # ---------------------------------------------------------------------------
        header = "\n".join(["from japl import Model as JaplModel",
                            f"from {name}.{name} import Model as CppModel",
                            "from sympy import Matrix",
                            "cpp_model = CppModel()",
                            "", "", ""])
        model_class = JaplClass(name, parent="JaplModel", members={"aerotable": Symbol("cpp_model.aerotable"),
                                                                   "atmosphere": Symbol("cpp_model.atmosphere"),
                                                                   "state_vars": state_vars_member,
                                                                   "input_vars": input_vars_member,
                                                                   "static_vars": static_vars_member})
        footer = "\n".join([f"{tab}dynamics = cpp_model.dynamics",
                            f"{tab}state_updates = cpp_model.state_updates",
                            f"{tab}input_updates = cpp_model.input_updates"])
        model_file_builder = FileBuilder("model.py", contents=[header, pycode(model_class), footer])

        # SimObject class file
        # ---------------------------------------------------------------------------
        header = "\n".join(["from japl import SimObject",
                            f"from {name}.model import {name} as _model",
                            "", "", ""])
        simobj_class = JaplClass(name, parent="SimObject", members={"state vars": self.state_vars,
                                                                    "input vars": self.input_vars,
                                                                    "static vars": self.static_vars,
                                                                    "model": Symbol("_model()")})
        simobj_file_builder = FileBuilder("simobj.py", contents=[header, pycode(simobj_class)])

        builder = ModuleBuilder(name, [file_builder])
        CodeGenerator.build_c_module(builder, other_builders=[model_file_builder, simobj_file_builder])
