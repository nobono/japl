from typing import Callable, Optional
import numpy as np
from enum import Enum
# from scipy.sparse import csr_matrix
# from scipy.sparse._csr import csr_matrix as Tcsr_matrix
# from sympy.matrices.expressions.matexpr import MatrixElement
from sympy import Matrix, MatrixSymbol, Symbol, Expr, Function
from sympy import nan as sp_nan
from sympy import simplify

from japl.Model.StateRegister import StateRegister
from japl.Util.Desym import Desym

from japl.BuildTools.DirectUpdate import DirectUpdateSymbol
from japl.BuildTools import BuildTools

# ---------------------------------------------------



class ModelType(Enum):
    NotSet = 0
    StateSpace = 1
    Function = 2
    Symbolic = 3


class Model:

    """This class is a Model interface for SimObjects"""

    def __init__(self, **kwargs) -> None:
        self._type = ModelType.NotSet
        self._dtype = np.float64
        self.state_register = StateRegister()
        self.input_register = StateRegister()
        self.dynamics_func: Optional[Callable] = None
        self.dynamics_expr = Expr(None)
        self.modules: dict = {}
        self.state_vars = Matrix([])
        self.input_vars = Matrix([])
        self.dt_var = Symbol("")
        self.vars: tuple = ()
        self.state_dim = 0
        self.input_dim = 0
        self.direct_state_update_func: Optional[Callable] = None
        self.direct_input_update_func: Optional[Callable] = None
        self.A = np.array([])
        self.B = np.array([])
        self.C = np.array([])
        self.D = np.array([])

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


    @classmethod
    def from_function(cls,
                      dt_var: Symbol,
                      state_vars: list|tuple|Matrix,
                      input_vars: list|tuple|Matrix,
                      dynamics_func: Optional[Callable] = None,
                      update_func: Optional[Callable] = None):
        """This method initializes a Model from a callable function.
        The provided function must have the following signature:

            func(t, X, U, dt)

        where, 't' is the sim-time 'X' is the current state, 'U', is
        the current inputs and 'dt' is the time step.

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
        model._type = ModelType.Function
        model.set_state(state_vars)
        model.set_input(input_vars)
        model.state_vars = model.state_register.get_vars()
        model.input_vars = model.input_register.get_vars()
        model.dt_var = dt_var
        model.vars = (model.state_vars, model.input_vars, dt_var)
        model.state_dim = len(model.state_vars)
        model.input_dim = len(model.input_vars)
        if dynamics_func:
            model.dynamics_func = dynamics_func
        if update_func:
            model.direct_state_update_func = update_func
        if (not dynamics_func) and (not update_func):
            raise Exception("Both dynamics_func and update_func cannot be undefined.")
        return model


    @classmethod
    def from_statespace(cls,
                        dt_var: Symbol,
                        state_vars: list|tuple|Matrix,
                        input_vars: list|tuple|Matrix,
                        A: np.ndarray|Matrix,
                        B: np.ndarray|Matrix,
                        C: Optional[np.ndarray|Matrix] = None,
                        D: Optional[np.ndarray|Matrix] = None):
        """This method initializes a Model from the matrices describing.
        a linear time-invariant system (LTI).

            X_dot = A*X + B*U

        where, 'X' is the current state, 'U', is the current inputs, 'A' is
        the design matrix, 'B' is the input matrix, and 'dt' is the time step.

        -------------------------------------------------------------------
        -- Arguments
        -------------------------------------------------------------------
        -- dt_var - symbolic dt e
        -- state_vars - iterable of symbolic state variables
        -- input_vars - iterable of symbolic input variables
        -- A - design matrix
        -- B - input matrix
        -- C - (optional) output matrix
        -- D - (optional) output matrix
        -------------------------------------------------------------------
        -- Returns
        -------------------------------------------------------------------
        -- model - the initialized Model
        -------------------------------------------------------------------
        """
        model = cls()
        model.set_state(state_vars)
        model.set_input(input_vars)
        model.state_vars = model.state_register.get_vars()
        model.input_vars = model.input_register.get_vars()
        model.dt_var = dt_var
        model.vars = (model.state_vars, input_vars, dt_var)
        model.dynamics_expr = A * model.state_vars + B * model.input_vars
        if isinstance(model.dynamics_expr, Expr) or isinstance(model.dynamics_expr, Matrix):
            model.dynamics_expr = simplify(model.dynamics_expr)
        model.dynamics_func = Desym(model.vars, model.dynamics_expr)  # type:ignore
        model.state_dim = A.shape[0]
        model.input_dim = B.shape[1]
        model.A = np.array(A)
        model.B = np.array(B)

        if C is not None:
            model.C = np.array(C)
        else:
            model.C = np.eye(model.A.shape[0])
        if D is not None:
            model.D = np.array(D)
        else:
            model.D = np.zeros_like(model.B)

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
                        dynamics_expr: Expr|Matrix|MatrixSymbol,
                        definitions: tuple = (),
                        modules: dict|list[dict] = {}):
        """This method initializes a Model from a symbolic expression.
        a Sympy expression can be passed which then is lambdified
        (see Sympy.lambdify) with computational optimization (see Sympy.cse).

        -------------------------------------------------------------------
        -- Arguments
        -------------------------------------------------------------------
        -- dt_var - symbolic dt e
        -- state_vars - iterable of symbolic state variables
        -- input_vars - iterable of symbolic input variables
        -- dynamics_expr - Sympy symbolic dynamics expression
        -- modules - pass custom library to Desym (see sympy.lambdify)
        -------------------------------------------------------------------
        -- Returns
        -------------------------------------------------------------------
        -- cls - the initialized Model
        -------------------------------------------------------------------
        """
        # first build model using provided definitions
        (state_vars,
         input_vars,
         dynamics_expr) = BuildTools.build_model(Matrix(state_vars),
                                                 Matrix(input_vars),
                                                 Matrix(dynamics_expr),
                                                 definitions)
        model = cls()
        model._type = ModelType.Symbolic
        model.modules = model.__set_modules(modules)
        model.set_state(state_vars)  # NOTE: will convert any Function to Symbol
        model.set_input(input_vars)  # NOTE: will convert any Function to Symbol
        model.state_vars = model.state_register.get_vars()
        model.input_vars = model.input_register.get_vars()
        model.dt_var = dt_var
        model.vars = (model.state_vars, model.input_vars, dt_var)
        model.dynamics_expr = dynamics_expr
        model.direct_state_update_func = model.__process_direct_state_updates(model.state_vars)
        model.direct_input_update_func = model.__process_direct_state_updates(model.input_vars)
        model.state_dim = len(model.state_vars)
        model.input_dim = len(model.input_vars)
        # create lambdified function from symbolic expression
        match dynamics_expr.__class__():  # type:ignore
            case Expr():
                model.dynamics_func = Desym(model.vars, dynamics_expr, modules=model.modules)
            case Matrix():
                model.dynamics_func = Desym(model.vars, dynamics_expr, modules=model.modules)
            case MatrixSymbol():
                model.dynamics_func = Desym(model.vars, dynamics_expr, modules=model.modules)
            case _:
                raise Exception("function provided is not Callable.")
        return model


    def _pre_sim_checks(self) -> bool:
        match self._type:
            case ModelType.StateSpace:
                if len(self.A.shape) < 2:
                    raise AssertionError("Matrix \"A\" must have shape of len 2")
                if len(self.B.shape) < 2:
                    raise AssertionError("Matrix \"B\" must have shape of len 2")
                if self._type == ModelType.StateSpace:
                    assert self.A.shape == self.C.shape
                    assert self.B.shape == self.D.shape
                    # assert len(self._X) == len(self.A)
            case ModelType.Function:
                pass
            case ModelType.Symbolic:
                pass

        # run state-register checks
        self.state_register._pre_sim_checks()
        self.input_register._pre_sim_checks()

        return True


    def __call__(self, *args) -> np.ndarray:
        """This method calls an update step to the model after the
        Model object has been initialized."""
        # NOTE: in certain situations dynamics_func may be undefined.
        # namely, when Model is built from_function() and dynamics_func
        # is left unspecified. Typically this results from a Model being
        # updated exclusively by direct / external updates.
        if self.dynamics_func:
            return self.dynamics_func(*args).flatten()
        else:
            return np.empty([])


    def step(self, X: np.ndarray, U: np.ndarray, dt: float) -> np.ndarray:
        """This method is the step method of Model over a single time step.

        -------------------------------------------------------------------
        -- Arguments
        -------------------------------------------------------------------
        -- X - state array for nstep of the model
        -- U - input array for nstep of the model
        -------------------------------------------------------------------
        -- Returns
        -------------------------------------------------------------------
        -- Xdot - derivative of the state (A*X + B*U)
        -------------------------------------------------------------------
        """

        self.__set_current_state(X)
        # self.__update_A_matrix_exprs(self.A, X)
        # return self.A @ X + self.B @ U
        # return self.dynamics_func(X, U).flatten()
        return self(X, U, dt)


    def get_state_id(self, names: str|list[str]) -> int|list[int]:
        """This method get the sympy variable associated with the provided
        name. variables must first be added to the StateRegister. If a list
        of state names are provided, then a list of corresponding state ids
        will be returned.

        If sympy MatrixElement has been registered in the state e.g. 'x[i, j]',
        then the provided name 'x' will return all indices of that particular
        state.

        -------------------------------------------------------------------
        -- Arguments
        -------------------------------------------------------------------
        -- name - (str | list[str]) name of the symbolic state variable
                name or a list of symbolic state variable names
        -------------------------------------------------------------------
        -- Returns
        -------------------------------------------------------------------
        -- (int | list[int]) - the index of the state variable in the
                state array or list of indices.
        -------------------------------------------------------------------
        """
        if isinstance(names, list):
            return [self.state_register[k]["id"] for k in names]
        else:
            return self.state_register[names]["id"]


    def get_input_id(self, names: str|list[str]) -> int|list[int]:
        """This method get the sympy variable associated with the provided
        name. variables must first be added to the StateRegister. If a list
        of state names are provided, then a list of corresponding input ids
        will be returned.

        If sympy MatrixElement has been registered in the input e.g. 'x[i, j]',
        then the provided name 'x' will return all indices of that particular
        input.

        -------------------------------------------------------------------
        -- Arguments
        -------------------------------------------------------------------
        -- name - (str | list[str]) name of the symbolic input variable
                name or a list of symbolic input variable names
        -------------------------------------------------------------------
        -- Returns
        -------------------------------------------------------------------
        -- (int | list[int]) - the index of the input variable in the
                input array or list of indices.
        -------------------------------------------------------------------
        """
        if isinstance(names, list):
            return [self.input_register[k]["id"] for k in names]
        else:
            return self.input_register[names]["id"]


    def set_state(self, state_vars: tuple|list|Matrix, labels: Optional[list|tuple] = None):
        """This method initializes the StateRegister attribute of the Model.

        -------------------------------------------------------------------
        -- Arguments
        -------------------------------------------------------------------
        -- state_vars - iterable of symbolic state variables
        -- labels - (optional) iterable of labels that may be used by the
                    Plotter class. order labels must correspond to order
                    of state_vars.
        -------------------------------------------------------------------
        -- Returns
        -------------------------------------------------------------------
        -- (Symbol) - the symbolic object of the state variable
        -------------------------------------------------------------------
        """
        return self.state_register.set(vars=state_vars, labels=labels)


    def set_input(self, input_vars: tuple|list|Matrix, labels: Optional[list|tuple] = None):
        """This method initializes the (inputs) StateRegister attribute of the Model.

        -------------------------------------------------------------------
        -- Arguments
        -------------------------------------------------------------------
        -- input_vars - iterable of symbolic input variables
        -- labels - (optional) iterable of labels that may be used by the
                    Plotter class. order labels must correspond to order
                    of input_vars.
        -------------------------------------------------------------------
        -- Returns
        -------------------------------------------------------------------
        -- (Symbol) - the symbolic object of the state variable
        -------------------------------------------------------------------
        """
        return self.input_register.set(vars=input_vars, labels=labels)

    def get_sym(self, name: str) -> Symbol:
        """This method gets the symbolic variable associated
        with the provided name.

        -------------------------------------------------------------------
        -- Arguments
        -------------------------------------------------------------------
        -- name - (str) name of the symbolic state variable
        -------------------------------------------------------------------
        -- Returns
        -------------------------------------------------------------------
        -- (Symbol) - the symbolic object of the state variable
        -------------------------------------------------------------------
        """
        return self.state_register.get_sym(name)


    def __process_direct_state_updates(self, state_vars: Matrix):
        """This method creates an update function from a symbolic state
        Matrix. Any DirectUpdate elements will be updated using its
        substitution expression, "sub_expr", while Symbol & Function
        elements result in NaN.

        -------------------------------------------------------------------
        -- Arguments
        -------------------------------------------------------------------
        -- state_vars - [Matrix|list]
        -------------------------------------------------------------------
        -- Returns
        -------------------------------------------------------------------
        -- (Callable) - lambdified sympy expression
        -------------------------------------------------------------------
        """
        state_updates = []
        for var in state_vars:  # type:ignore
            if isinstance(var, DirectUpdateSymbol):
                state_updates.append(var.sub_expr)
            elif isinstance(var, Symbol):
                state_updates.append(sp_nan)
            elif isinstance(var, Function):
                state_updates.append(sp_nan)
            else:
                raise Exception("unhandled case.")
        t = Symbol('t')  # 't' variable needed to adhear to func argument format
        update_func = Desym((t, *self.vars), Matrix(state_updates), modules=self.modules)
        return update_func
