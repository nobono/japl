# ---------------------------------------------------

from typing import Callable, Optional

# ---------------------------------------------------

import numpy as np

# ---------------------------------------------------

# from control.iosys import StateSpace

# ---------------------------------------------------

from enum import Enum

# ---------------------------------------------------

from scipy.sparse import csr_matrix
from scipy.sparse._csr import csr_matrix as Tcsr_matrix
from sympy import Matrix, MatrixSymbol, Mul, Pow, Symbol, Expr
from sympy.matrices.expressions.matexpr import MatrixElement
from sympy import simplify

from japl.Model.StateRegister import StateRegister
from japl.Util.Desym import Desym

from japl.Model.BuildTools.DirectUpdate import DirectUpdateSymbol

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
        self.dynamics_func: Callable = lambda *_: None
        self.direct_state_update_map: dict = {}
        self.state_dim = 0
        self.expr = Expr(None)
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


    # TODO these only used right now to access quaternion in
    # Plotter. maybe we can dispense with these somehow...
    def __set_current_state(self, X: np.ndarray):
        """Setter for Model state reference array \"Model._iX_reference\".
        This method should only be called by Model.step()."""
        self._iX_reference = X


    def get_current_state(self) -> np.ndarray:
        """Getter for Model state reference array. Used to access the
        state array between time steps outside of the Sim class."""
        return self._iX_reference.copy()


    @DeprecationWarning
    def ss(self,
           A: np.ndarray,
           B: np.ndarray,
           C: Optional[np.ndarray] = None,
           D: Optional[np.ndarray] = None) -> None:

        self._type = ModelType.StateSpace

        self.A = np.array(A)
        self.B = np.array(B)

        if C is not None:
            self.C = np.array(C)
        else:
            self.C = np.eye(self.A.shape[0])

        if D is not None:
            self.D = np.array(D)
        else:
            self.D = np.zeros_like(self.B)

        self.state_dim = self.A.shape[0]

        # handle user-defined state references in state-space matrices
        self.__handle_sym_expr_references()

        # reduce complexity of mat mult.
        # self.A = csr_matrix(self.A.astype(self._dtype))
        # self.B = csr_matrix(self.B.astype(self._dtype))
        # self.C = csr_matrix(self.C.astype(self._dtype))
        # self.D = csr_matrix(self.D.astype(self._dtype))

        # failing here, means matrix contains symbolic references still
        assert self.A.dtype in [float, int]
        assert self.B.dtype in [float, int]
        assert self.C.dtype in [float, int]
        assert self.D.dtype in [float, int]

        # create the step function
        self.dynamics_func = lambda X, U: self.A @ X + self.B @ U


    def from_function(self,
                      dt_var: Symbol,
                      state_vars: list|tuple|Matrix,
                      input_vars: list|tuple|Matrix,
                      func: Callable):
        """This method initializes a Model from a callable function.
        The provided function must have the following signature:

            func(X, U, dt)

        where, 'X' is the current state, 'U', is the current inputs and
        'dt' is the time step.

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
        -- self - the initialized Model
        -------------------------------------------------------------------
        """
        # TODO initialize self.state_dim somehow ...
        self._type = ModelType.Function
        self.set_state(state_vars)
        self.set_input(input_vars)
        self.state_vars = self.state_register.get_vars()
        self.input_vars = self.input_register.get_vars()
        self.dt_var = dt_var
        self.vars = (self.state_vars, self.input_vars, dt_var)
        self.state_dim = len(self.state_vars)
        self.input_dim = len(self.input_vars)
        self.dynamics_func = func
        assert isinstance(func, Callable)
        return self


    def from_statespace(self,
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
        -- self - the initialized Model
        -------------------------------------------------------------------
        """
        self.set_state(state_vars)
        self.set_input(input_vars)
        self.state_vars = self.state_register.get_vars()
        self.input_vars = self.input_register.get_vars()
        self.dt_var = dt_var
        self.vars = (self.state_vars, input_vars, dt_var)
        self.expr = A * self.state_vars + B * self.input_vars
        if isinstance(self.expr, Expr) or isinstance(self.expr, Matrix):
            self.expr = simplify(self.expr)
        self.dynamics_func = Desym(self.vars, self.expr) #type:ignore
        self.state_dim = A.shape[0]
        self.input_dim = B.shape[1]
        self.A = np.array(A)
        self.B = np.array(B)

        if C is not None:
            self.C = np.array(C)
        else:
            self.C = np.eye(self.A.shape[0])
        if D is not None:
            self.D = np.array(D)
        else:
            self.D = np.zeros_like(self.B)

        return self


    def from_expression(self,
                        dt_var: Symbol,
                        state_vars: list|tuple|Matrix,
                        input_vars: list|tuple|Matrix,
                        dynamics_expr: Expr|Matrix|MatrixSymbol,
                        modules: dict = {}):
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
        -- self - the initialized Model
        -------------------------------------------------------------------
        """
        self._type = ModelType.Symbolic
        self.modules = modules
        self.set_state(state_vars)
        self.set_input(input_vars)
        self.state_vars = self.state_register.get_vars()
        self.input_vars = self.input_register.get_vars()
        self.dt_var = dt_var
        self.vars = (self.state_vars, input_vars, dt_var)
        self.expr = dynamics_expr
        self.dynamics_expr = dynamics_expr
        self.direct_state_update_map = self.__process_direct_state_updates(self.state_vars)
        self.state_dim = len(self.state_vars)
        self.input_dim = len(input_vars)
        # create lambdified function from symbolic expression
        match dynamics_expr.__class__(): #type:ignore
            case Expr():
                self.dynamics_func = Desym(self.vars, dynamics_expr, modules=modules)
            case Matrix():
                self.dynamics_func = Desym(self.vars, dynamics_expr, modules=modules)
            case MatrixSymbol():
                self.dynamics_func = Desym(self.vars, dynamics_expr, modules=modules)
            case _:
                raise Exception("function provided is not Callable.")
        return self


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


    @DeprecationWarning
    def __get_symbolic_name(self, var: Symbol|MatrixSymbol|MatrixElement) -> str:
        if isinstance(var, Symbol):
            return var.name
        elif isinstance(var, MatrixSymbol):
            return var.name
        elif isinstance(var, MatrixElement):
            return var.symbol.name #type:ignore
        else:
            raise Exception("wrong arg type.")

    @DeprecationWarning
    def __handle_sym_expr_references(self) -> None:
        """
            This method handles the user-defined state references (strings) provided
        within the state space model matrices.

        Use of StateRegister allows for non-linear state-space matrices such as:
            A = [
                    [1, 0],
                    [0, 'xpos'],
                ]

        where the registered state corresponds to the following names:
            X = ['xpos', 'ypos', 'ypos']

        The general intended operations ecapsulated by this class are as follows:
            - check user-defined state-space matrix for any strings (state references).

            - store info of [i][j] indices of matrix, the state-references id of the sympy expression
                used, coefficents in expression and powers of sympy state var reference.

        The info gathered in the previous section will be applied in Model.step().
        """

        # TODO this only checks matrix A... expand to B (C? and D?)
        _len = self.A.shape[0]
        for i in range(_len):
            for j in range(_len):
                val = self.A[i, j]
                try:
                    self.A[i, j] = float(val) #type:ignore
                except Exception as _:

                    # split symbolic matrix bin as (coefficent, state-variable)
                    if isinstance(val, Expr):
                        coef_mul: tuple = val.as_coeff_Mul()
                        coef = self._dtype(coef_mul[0])
                        var: Symbol|MatrixElement = coef_mul[1] #type:ignore

                        # check if exponents on symbolic state variable exist
                        if isinstance(var, Pow):
                            exp = self._dtype(var.exp)
                        else:
                            exp = None

                        # var_name = self.__get_symbolic_name(var)
                        var_name = str(var)

                        # check if state reference is registered
                        if var_name not in self.state_register:
                            raise Exception(f"User defined state references \"{var_name}\"\
                                    in state-space matrix but \"{var_name}\" is not a registered state")
                        else:
                            # store state references in state-space matrix
                            state_id = self.state_register[var_name]["id"]
                            self._sym_references.append({"A_slice": (i, j), "state_id": state_id, "coef": coef, "exp": exp})
                            self.A[i:i+1, j:j+1] = coef

        # ensure Model matrices have type float
        # since sympy expressions are allowed in 
        # original definition.
        self.A = self.A.astype(float)


    # TODO this can be generalized to mats B, C, D...
    @DeprecationWarning
    def __update_A_matrix_exprs(self, A: np.ndarray|Tcsr_matrix, X: np.ndarray) -> None:
        """This method uses the info gathered in "__handle_sym_expr_references()"
        as well as the current state 'X' for the time step and applies the changes
        in matrix A.
        """
        for sim_ref in self._sym_references:
            A_slice = sim_ref["A_slice"]
            state_id = sim_ref["state_id"]
            if sim_ref["exp"] is not None:
                val = X[state_id]**sim_ref["exp"]
            else:
                val = X[state_id]
            A[A_slice[0], A_slice[1]] = sim_ref["coef"] * val


    def __call__(self, *args) -> np.ndarray:
        """This method calls an update step to the model after the
        Model object has been initialized."""
        return self.dynamics_func(*args).flatten()


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


    @DeprecationWarning
    def add_state(self, name: str, id: int, label: str = "") -> Symbol:
        """This method registers a SimObject state name and plotting label
        with a user-specified name. The purpose of this register is for ease
        of access to SimObject states without having to use the satte index
        number.

        -------------------------------------------------------------------
        -- Arguments
        -------------------------------------------------------------------
        -- name - user-specified name of state
        -- id - state index number
        -- label - (optional) string other than "name" that will be displayed
                    in plots / visualization
        -------------------------------------------------------------------
        """
        return self.state_register.add_state(name=name, id=id, label=label)


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


    # def __process_direct_state_updates(self, direct_state_update_dict: dict,
    #                                 modules: dict = {}) -> dict:
    #     """This method takes a direct_state_update_dict which maps direct state
    #     updates of the model instead of integrating the dynamics.

    #      - The input dict should be of format: {Symbol: [Expr|Callable]}
    #      - and will be converted to format:    {state_id: Callable}.

    #     -------------------------------------------------------------------
    #     -- Arguments
    #     -------------------------------------------------------------------
    #     -- direct_state_update_dict - (dict) dict mapping state_vars to expression
    #     -------------------------------------------------------------------
    #     -- Returns
    #     -------------------------------------------------------------------
    #     -- (dict) - mapping state_var index to callable function
    #     -------------------------------------------------------------------
    #     """
    #     ret = {}
    #     for var, expr in direct_state_update_dict.items():
    #         var_id: int = self.get_state_id(var) #type:ignore
    #         match expr.__class__:
    #             case Expr():
    #                 ret.update({var_id: Desym(self.vars, expr, modules=modules)})
    #             case Callable():
    #                 ret.update({var_id: expr})
    #             case _:
    #                 raise Exception("unhandled case.")
    #     return ret


    def __process_direct_state_updates(self, state_vars: Matrix) -> dict:
        """This method takes the state array / state_vars as input which
        finds any DirectUpdate elements which map direct state updates of
        the model instead of integrating the state dynamics.

         - The input dict should be of format: {Symbol: [Expr|Callable]}
         - and will be converted to format:    {state_id: Callable}.

        -------------------------------------------------------------------
        -- Arguments
        -------------------------------------------------------------------
        -- state_vars - [Matrix|list] 
        -------------------------------------------------------------------
        -- Returns
        -------------------------------------------------------------------
        -- (dict) - mapping state_var index to callable function
        -------------------------------------------------------------------
        """
        # ret = {}
        # for var, expr in direct_state_update.items():
        #     var_id: int = self.get_state_id(var) #type:ignore
        #     match expr.__class__:
        #         case Expr():
        #             ret.update({var_id: Desym(self.vars, expr, modules=modules)})
        #         case Callable():
        #             ret.update({var_id: expr})
        #         case _:
        #             raise Exception("unhandled case.")
        # return ret
        direct_state_update_map = {}
        for i, var in enumerate(state_vars): #type:ignore
            if isinstance(var, DirectUpdateSymbol):
                assert var.expr is not None
                t = Symbol('t') # 't' variable needed to adhear to func arument format
                func = Desym((t, *self.vars), var.expr, modules=self.modules)
                direct_state_update_map.update({i: func})
        return direct_state_update_map

