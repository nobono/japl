import os
from sympy import Matrix, Symbol, Function, Expr, Number
from sympy import Derivative
from japl.Model.StateRegister import StateRegister
from japl.BuildTools.DirectUpdate import DirectUpdateSymbol
from pprint import pformat
import threading
from functools import partial



def create_error_header(msg: str, char: str = "-", char_len: int = 40) -> str:
    seg = char * char_len
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

    print("=" * 50)
    print("building model...")
    print("=" * 50)

    # state & input array checks
    _check_var_array_types(state, "state")
    _check_var_array_types(input, "input")

    # handle formatting of provided definitions, state, input
    def_subs = _create_subs_from_definitions(definitions)

    # create subs for state / input symbols
    state_subs = _create_symbol_subs(state)
    input_subs = _create_symbol_subs(input)

    print("applying state & input substitions to definitions...")
    # apply state & input subs to defs
    # NOTE: idea here is that Function in state
    # become Symbols with states_subs. This then
    # needs to be applied to the definitions provided.
    for key in def_subs.keys():
        # NOTE: also sub itself after subbing state & input
        # (order here is important) to deal with recursive
        # definitions
        _defs = def_subs[key].subs(state_subs).subs(input_subs).subs(def_subs)
        def_subs[key] = _defs

    # apply definitions sub to state & input
    # TODO: do you need this or not?
    # NOTE: This looks like its for putting a symbol/Matrix
    # in the state array then a definition for said symbol/Matrix
    # in defs_subs...
    # _apply_definitions_to_array(state, def_subs)
    # _apply_definitions_to_array(input, def_subs)

    all_subs = {}
    all_subs.update(def_subs)
    all_subs.update(state_subs)
    all_subs.update(input_subs)
    # all_subs = _apply_subs_to_dict(all_subs)

    # _apply_subs_to_direct_updates(state, state_subs)
    # _apply_subs_to_direct_updates(input, state_subs)
    # _apply_subs_to_direct_updates(state, input_subs)
    # _apply_subs_to_direct_updates(input, input_subs)

    ###################################

    # apply subs to dynamics
    print("applying initial dynamics substitions...")
    dynamics = dynamics.subs(all_subs).doit()

    # default symbols imposed by Sim
    t = Symbol("t")
    dt = Symbol("dt")

    ############################################################
    # convert any Functions left in state array to Symbols
    ############################################################
    print("applying state substitions...")
    state_function_subs = []
    state_function_set = state.atoms(Function)
    state_direct_update_set = state.atoms(DirectUpdateSymbol)
    # gather functions from direct updates' sub_expr
    for du in state_direct_update_set:
        state_function_set.update(du.sub_expr.atoms(Function))
    # filter for user-defined functions wrt. time (i.e.) with args (t)
    t_functions = [i for i in state_function_set if hasattr(i, "name") and (i.args == (t,))]
    for func in t_functions:
        state_function_subs += [(func, Symbol(func.name))]
    # subs for DirectUpdateSymbols
    for var in state:
        if isinstance(var, DirectUpdateSymbol):
            var.sub_expr = var.sub_expr.subs(state_function_subs)  # type:ignore
    state = state.subs(state_function_subs)

    ############################################################
    # convert any Functions left in state input to Symbols
    ############################################################
    print("applying input substitions...")
    input_function_subs = []
    input_function_set = input.atoms(Function)
    input_direct_update_set = input.atoms(DirectUpdateSymbol)
    # gather functions from direct updates' sub_expr
    for du in input_direct_update_set:
        input_function_set.update(du.sub_expr.atoms(Function))
    # filter for user-defined functions wrt. time (i.e.) with args (t)
    t_functions = [i for i in input_function_set if hasattr(i, "name") and (i.args == (t,))]
    for func in t_functions:
        input_function_subs += [(func, Symbol(func.name))]
    # subs for DirectUpdateSymbols
    for var in input:
        if isinstance(var, DirectUpdateSymbol):
            var.sub_expr = var.sub_expr.subs(input_function_subs)  # type:ignore
    input = input.subs(input_function_subs)

    ############################################################
    # convert any Functions left in dynamics to Symbols
    # replace any function that is a function of only 't': func(t)
    # NOTE:
    # - dont want to replace atmosphere_get(var(t))
    #   only replace "var(t)" with "var"
    ############################################################

    def sub_func(expr, subs: list, id: int, result: list):
        # result[id] = expr.subs(subs)
        for sub in subs:
            expr = expr.subs(sub)
        result[id] = expr
        # NOTE: verbose options
        # njobs_done = len([i for i in result if i is not None])
        # njobs = len(result)
        # complete = njobs_done / njobs
        # print("%.1f" % (complete * 100.0))

    dynamics_function_subs = []
    for row in dynamics:
        row_funcs = [arg for arg in row.atoms(Function)]  # type:ignore
        # only looking for user-defined funcs
        usr_row_funcs = [i for i in row_funcs if hasattr(i, "name") and (i.args == (t,))]
        for func in usr_row_funcs:
            dynamics_function_subs += [(func, Symbol(func.name))]
    # convert back to list
    # dynamics_function_subs = [i for i in dynamics_function_subs]

    print("starting threads for dynamics substitions...")
    # NOTE: dynamics matrix can be complex use threading
    # to speed up substitions
    thread_results = [None] * int(dynamics.shape[0])
    thread_funcs = [partial(sub_func, expr, [dynamics_function_subs, state_function_subs], i, thread_results)
                    for i, expr in enumerate(dynamics)]  # type:ignore
    threads = [threading.Thread(target=func) for func in thread_funcs]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    dynamics = Matrix(thread_results)

    # dynamics = dynamics.subs(dynamics_function_subs)

    ############################################################
    # check for any undefined differential expresion in dynamics
    ############################################################
    print("doing model checks...")
    dynamic_functions = [i for i in dynamics.atoms(Function) if hasattr(i, "name") and (i.args == (t,))]
    if dynamic_functions:
        raise Exception(f"Undefined Functions found in dynamics matrix:\n{dynamic_functions}")

    # gather symbols from direct updates
    # NOTE: conversion to str since sympy vars being hashable is unknown right now
    direct_update_symbols = set()
    direct_update_symbols.update(set([i.name for i in state_direct_update_set]))
    direct_update_symbols.update(set([i.name for i in input_direct_update_set]))

    # gather symbols from dynamics
    dynamics_symbols = set([i.name for i in dynamics.atoms(Symbol)])

    total_symbols = set()
    total_symbols.update(dynamics_symbols)
    total_symbols.update(direct_update_symbols)

    # subtract defined symbols in the state & input arrays
    state_symbols = [i.name for i in state]  # type:ignore
    input_symbols = [i.name for i in input]  # type:ignore
    undefined_symbols = total_symbols.difference(set(state_symbols)).difference(set(input_symbols))
    # ignore default symbols (t, dt)
    undefined_symbols = undefined_symbols.difference({t.name, dt.name})

    if undefined_symbols:
        wrap_str = f"\n\n{'=' * 50}\n"
        Warning(f"{wrap_str}\nUndefined Symbols found in model:\n{undefined_symbols}{wrap_str}")
        breakpoint()

    ############################################################
    # _check_dynamics_for_undefined_diffs(dynamics)
    # _check_dynamics_for_undefined_function(dynamics, state)

    # write_array(state, "./temp_state.py")
    # write_array(input, "./temp_input.py")
    # write_array(dynamics, "./temp_dynamics.py")

    print("=" * 50, end="\n\n")
    return (state, input, dynamics)


def write_array(array, out_path: str):
    # data = pformat(array)
    if os.path.exists(out_path):
        option = "r+"
    else:
        option = "a+"
    with open(out_path, option) as f:
        f.seek(0)
        for elem in array:
            if isinstance(elem, DirectUpdateSymbol):
                data = pformat(elem.sub_expr)
            else:
                data = pformat(elem)
            f.write(data)
            f.write('\n')


def _get_direct_updates(array):
    return [i for i in array if isinstance(i, DirectUpdateSymbol)]


def _get_sub_expr(array):
    return [i.sub_expr for i in array if isinstance(i, DirectUpdateSymbol)]


def _apply_subs_to_expr(expr, subs: dict|list[dict]):
    if isinstance(subs, dict):
        return expr.subs(subs)
    elif isinstance(subs, list):
        for sub in subs:
            expr.subs(sub)
        return expr


def _apply_subs_to_dict(subs: dict):
    for key, val in subs.items():
        subs[key] = _apply_subs_to_expr(val, subs)
    return subs


def _create_symbol_subs(array):
    """This method generates a 'subs' dict from provided array of
    symbolic variables which map Symbols, Functions, DirectUpdateSymbol
    to a basic sympy Symbol. This represents pulling values from the X, U
    arrays in the step update methods."""
    subs = {}
    for elem in array:
        if isinstance(elem, DirectUpdateSymbol):
            symbol = Symbol(f"{elem.state_expr.name}")  # type:ignore
            subs.update({elem.state_expr: symbol})
        elif isinstance(elem, Symbol):
            symbol = Symbol(elem.name)
            subs.update({elem: symbol})
        elif isinstance(elem, Function):
            symbol = Symbol(elem.name)  # type: ignore
            subs.update({elem: symbol})
        else:
            raise Exception(f"unhandled case. {elem}")
    return subs


def _get_array_var_names(array):
    names = []
    for _, elem in enumerate(array):
        if isinstance(elem, Symbol):
            names.append(elem.name)
            continue
        if isinstance(elem, Function):
            names.append(elem.name)  # type:ignore
            continue
        if isinstance(elem, DirectUpdateSymbol):
            _get_array_var_names(elem.state_expr)
            continue
        else:
            raise Exception("unhandled case.")
    return names


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
        if elem.has(Derivative):
            raise Exception(f"\n\ndynamics-id[{i}]: undefined differential expression "
                            f"\n\n\t\"{elem}\"\n\n\t in the dynamics array. Assign this expression in "
                            f"the definitions or update using DirectUpdate().")


def _check_dynamics_for_undefined_function(dynamics: Matrix, state: Matrix):
    seg = "-" * 40
    error_header = "\n\n" + seg + "\nUndefined functions found in dynamics:" + "\n" + seg
    fail = False
    var_msg = []
    for irow, row in enumerate(dynamics):  # type:ignore
        found = False
        for ivar, var in enumerate(state):  # type:ignore
            if isinstance(var, DirectUpdateSymbol):
                if isinstance(var.state_expr, Function):
                    if row.has(var.state_expr):  # type:ignore
                        fail = True
                        found = True
                        var_msg += [f"state[{ivar}] undefined variable: {var.state_expr} "]
            elif isinstance(var, Function):
                if row.has(var):  # type:ignore
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
    for irow, row in enumerate(array):  # type:ignore
        found = False
        for ivar, var in enumerate(array_compare):  # type:ignore
            if isinstance(var, DirectUpdateSymbol):
                if isinstance(row, DirectUpdateSymbol):
                    if row.sub_expr.__class__ in [Number, float, int]:
                        pass
                    elif row.sub_expr.__class__ in [Symbol, Function]:
                        if row.sub_expr.has(var.state_expr):  # type:ignore
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
                for elem_old, elem_new in zip(old, new):  # type:ignore
                    ret[elem_old] = elem_new
            else:
                try:
                    ret[old] = new
                except Exception as e:
                    raise Exception(e, "\nunhandled case. old and new need to both have '__len__'.")
    else:
        ret = sub
    return ret


def _apply_subs_to_direct_updates(state: Matrix, subs: dict|list[dict]) -> None:
    """applies substitions to DirectUpdateSymbol.expr which is the expression
    that is lambdified for direct state updates."""
    for var in state:
        if isinstance(var, DirectUpdateSymbol):
            if isinstance(subs, dict):
                var.sub_expr = var.sub_expr.subs(subs)    # type:ignore
            elif isinstance(subs, list):
                for sub in subs:
                    if var.sub_expr.__class__ in [Number, float, int]:
                        pass
                    else:
                        var.sub_expr = var.sub_expr.subs(sub)    # type:ignore
            else:
                raise Exception("unhandled case.")


def _apply_definitions_to_array(array: Matrix, subs: dict):
    """This method is for applying substitions to an array of
    symbolic variables. substitions are made by instantiating
    a DirectUpdateSymbol for the original symbol."""
    for var, sub in subs.items():
        for ielem, elem in enumerate(array):  # type:ignore
            if var == elem:
                if isinstance(elem, Symbol):
                    array[ielem] = DirectUpdateSymbol(f"{str(elem)}", state_expr=elem, sub_expr=sub)
                # NOTE: if a function is subbed for an array variable, it is made final here
                # and converted to a state Symbol. I don't know if this should be done here or
                # in a "check" method but let's roll with it.
                elif isinstance(elem, Function):
                    array[ielem] = DirectUpdateSymbol(f"{str(elem.name)}", state_expr=elem, sub_expr=sub)  # type:ignore
                elif isinstance(elem, Expr):
                    array[ielem] = DirectUpdateSymbol(f"{str(elem.name)}", state_expr=elem, sub_expr=sub)  # type:ignore
                else:
                    raise Exception("unhandled case.")
