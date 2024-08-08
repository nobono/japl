from typing import Optional
from sympy import Matrix, Symbol, Function, Expr, symbols
from sympy import Derivative
from sympy.core.function import UndefinedFunction
from japl.Model.StateRegister import StateRegister
from japl.BuildTools.DirectUpdate import DirectUpdateSymbol



def create_error_header(msg: str, char: str = "-", char_len: int = 40) -> str:
    seg = char*char_len
    header = "\n\n" + seg + f"\n{msg}:\n" + seg + "\n"
    return header


def build_model(state: Matrix,
                input: Matrix,
                dynamics: Matrix,
                definitions: tuple = ()) -> tuple:
    """
    Notes:
        - state and input arrays must maintain their symbolic name.
    """
    # state & input array checks
    _check_var_array_types(state, "state")
    _check_var_array_types(input, "input")

    # handle formatting of provided definitions, state, input
    def_subs = _create_subs_from_definitions(definitions)

    # create subs from
    state_subs = _create_subs_from_array(state)
    input_subs = _create_subs_from_array(input)

    # apply definitions sub to state & input
    # via DirectUpdateSymbol
    _apply_definitions_to_array(state, def_subs)
    _apply_definitions_to_array(input, def_subs)

    # process DirectUpdate.expr with diff & state subs
    _apply_subs_to_direct_updates(state, [def_subs, state_subs, input_subs])
    _apply_subs_to_direct_updates(input, [def_subs, state_subs, input_subs])

    # NOTE: may not need this ####################################
    # now that subs applied to DirectUpdateSymbols,
    # update state & input subs again
    # state_sub = _create_subs_from_vars(state)
    # input_sub = _create_subs_from_vars(input)
    # NOTE: may not need this ####################################

    # do substitions
    dynamics = dynamics.subs(def_subs).doit()
    dynamics = dynamics.subs(state_subs).subs(input_subs)

    # TODO: this needs to be re-evaluated. May not be neccessary
    # or best emthod of checking
    _check_array_for_undefined_function(state, "state", state, force_symbol=True)
    _check_array_for_undefined_function(state, "state", input, force_symbol=True)
    _check_array_for_undefined_function(input, "input", state, force_symbol=True)
    _check_array_for_undefined_function(input, "input", input, force_symbol=True)

    # check for any undefined differential expresion in dynamics
    _check_dynamics_for_undefined_diffs(dynamics)
    _check_dynamics_for_undefined_function(dynamics, state)

    return (state, input, dynamics)


def _check_var_array_types(array, array_name: str = "array"):
    """This method checks to make sure variables defined in the 
    state & input arrays are of types Symbol, Function, or DirectUpdateSymbol.
    These types can be reduced to a Symbol which the Model class uses to
    register state & input variable indices."""
    for i, elem in enumerate(array):
        if isinstance(elem, Symbol):
            continue
        if isinstance(elem, Function):
            continue
        if isinstance(elem, DirectUpdateSymbol):
            continue
        else:
            raise Exception(f"\n\n{array_name}-id[{i}]: cannot register a variable for "
                    f"expression \n\n\t\"{elem}\":\n\n\tElements of the state array must be "
                    f"either Symbol, Function, or DirectUpdate. Add to the array a "
                    f"variable and assign to it the expression using the definitions tuple.")


def _check_dynamics_for_undefined_diffs(dynamics):
    """This method checks for any undefined Derivative types in the dynamics.
    undefined Derivatives indicate a missing substition in the definitions."""
    for i, elem in enumerate(dynamics):
        if isinstance(elem, Derivative):
            raise Exception(f"\n\ndynamics-id[{i}]: undefined differential expression "
                            f"\n\n\t\"{elem}\"\n\n\t in the dynamics array. Assign this expression in "
                            f"the definitions or update using DirectUpdate().")


def _check_dynamics_for_undefined_function(dynamics: Matrix, state: Matrix):
    seg = "-"*40
    error_header = "\n\n" + seg +\
                    "\nUndefined functions found in dynamics:" +\
                   "\n" + seg
    fail = False
    var_msg = []
    for irow, row in enumerate(dynamics): #type:ignore
        found = False
        for ivar, var in enumerate(state): #type:ignore
            if isinstance(var, DirectUpdateSymbol):
                if row.has(var.state_expr): #type:ignore
                    fail = True
                    found = True
                    var_msg += [f"state[{ivar}] undefined variable: {var.state_expr} "]
            elif isinstance(var, Function):
                if row.has(var): #type:ignore
                    fail = True
                    found = True
                    var_msg += [f"state[{ivar}] undefined variable: {var} "]

        if found:
            row_msg = f"\nof expression: {row} (dynamics row {irow})"
            var_msg += [row_msg]
    if fail:
        found_vars_str = "\n".join(var_msg)
        raise Exception(error_header + f"\n\n{found_vars_str}")


def _check_array_for_undefined_function(array: Matrix, array_name: str, array_compare,
                                        force_symbol: bool = False) -> None:
    error_msg = create_error_header(f"Undefined functions found in {array_name}")
    fail = False
    for irow, row in enumerate(array): #type:ignore
        found = False
        for ivar, var in enumerate(array_compare): #type:ignore
            if isinstance(var, DirectUpdateSymbol):
                if isinstance(row, DirectUpdateSymbol):
                    if row.sub_expr.has(var.state_expr): #type:ignore
                        fail = True
                        found = True
                        error_msg += f"\n{array_name}[{ivar}] undefined variable: {var.state_expr} "
                        if force_symbol:
                            row.sub_expr = StateRegister._extract_variable(row.sub_expr)
        if found:
            row_msg = f"of expression: {row} ({array_name} row {irow})"
            error_msg += row_msg
    if fail and not force_symbol:
        raise Exception(error_msg)


def _create_subs_from_definitions(sub: tuple|list|dict) -> dict:
    """This method is used to convert differntial definitions
    into a substitutable dict.
        - handles substitions passed as Matrix"""
    ret = {}
    if isinstance(sub, tuple) or isinstance(sub, list):
        # update a new dict with substition pairs.
        # if pair is Matrix or MatrixSymbol (N x 1),
        # update each element.
        for old, new in sub:
            if hasattr(old, "__len__") and hasattr(new, "__len__"):
                for elem_old, elem_new in zip(old, new): #type:ignore
                    ret[elem_old] = elem_new
            else:
                try:
                    ret[old] = new
                except Exception as e:
                    raise Exception(e, "\nunhandled case. old and new need to both have '__len__'.")
    else:
        ret = sub
    return ret


def _create_subs_from_array(array: tuple|list|Matrix) -> dict:
    """This method generates a 'subs' dict from provided
    symbolic variables (Symbol, Function, Matrix). This is
    used for substitution of variables into a sympy expression.
        - handles substitions passed as Matrix"""
    assert hasattr(array, "__len__")
    ret = {}
    # for each element get the name
    for var in array:
        # NOTE: unused?######################################
        # if hasattr(var, "__len__"): # if Matrix
        #     for elem in var: #type:ignore
        #         ret[elem] = StateRegister._extract_variable(elem)
        # NOTE: unused?######################################
        if isinstance(var, DirectUpdateSymbol):
            ret[var.state_expr] = var.sub_expr
        elif isinstance(var, Function):
            ret[var] = StateRegister._extract_variable(var)
        elif isinstance(var, Symbol):
            ret[var] = StateRegister._extract_variable(var)
        else:
            raise Exception("unhandled case.")
    return ret


def _apply_subs_to_direct_updates(state: Matrix, subs: dict|list[dict]) -> None:
    """applies substitions to DirectUpdateSymbol.expr which is the expression
    that is lambdified for direct state updates."""
    for var in state:
        if isinstance(var, DirectUpdateSymbol):
            if isinstance(subs, dict):
                var.sub_expr = var.sub_expr.subs(subs)   #type:ignore
            elif isinstance(subs, list):
                for sub in subs:
                    var.sub_expr = var.sub_expr.subs(sub)   #type:ignore
            else:
                raise Exception("unhandled case.")


def _apply_definitions_to_array(array: Matrix, subs: dict):
    for var, sub in subs.items():
        for ielem, elem in enumerate(array): #type:ignore
            if var == elem:
                if isinstance(elem, Symbol):
                    array[ielem] = DirectUpdateSymbol(f"{str(elem)}", state_expr=elem, sub_expr=sub)
                # NOTE: if a function is subbed for an array variable, it is made final here
                # and converted to a state Symbol. I don't know if this should be done here or
                # in a "check" method but let's roll with it.
                elif isinstance(elem, Function):
                    array[ielem] = DirectUpdateSymbol(f"{str(elem.name)}", state_expr=elem, sub_expr=sub) #type:ignore
                elif isinstance(elem, Expr):
                    array[ielem] = DirectUpdateSymbol(f"{str(elem.name)}", state_expr=elem, sub_expr=sub) #type:ignore
                else:
                    raise Exception("unhandled case.")

