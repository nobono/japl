import os
from typing import Any, Iterable
from sympy import Matrix, Symbol, Function, Expr, Number, nan
from sympy import Float
from sympy import Derivative, Piecewise
from sympy import cse
from sympy import symbols
from sympy.matrices import MatrixSymbol
from sympy.matrices.expressions.matexpr import MatrixElement
from japl.Model.StateRegister import StateRegister
from japl.BuildTools.DirectUpdate import DirectUpdateSymbol
from japl.Util.Util import iter_type_check
from pprint import pformat
from multiprocess import Pool  # type:ignore
from multiprocess import cpu_count  # type:ignore
from japl.Util.Util import flatten_list
import dill as pickle
from time import perf_counter
import numpy as np
from tqdm import tqdm
from collections import defaultdict

from sympy.codegen.ast import Assignment, Variable, Return



def dict_subs_func(key_expr: tuple[str, Any], subs: list) -> bytes:
    """used for making substitions into a dict.
    e.g. when subbing expressions of the definitions dict."""
    key, expr = pickle.loads(key_expr)
    for sub in subs:
        sub = pickle.loads(sub)
        expr = expr.xreplace(sub)  # .doit()
    return pickle.dumps({key: expr})


def array_subs_func(expr, subs: list[dict]) -> bytes:
    """used for making substitions to expressions of a
    Matrix."""
    expr = pickle.loads(expr)
    for sub in subs:
        sub = pickle.loads(sub)
        expr = expr.xreplace(sub)  # .doit()
    return pickle.dumps(expr)


def create_error_header(msg: str, char: str = "-", char_len: int = 40) -> str:
    seg = char * char_len
    header = "\n\n" + seg + f"\n{msg}:\n" + seg + "\n"
    return header


def parallel_subs(target_expr: Matrix|dict, subs: tuple[tuple, ...]|list[dict]|dict) -> Matrix|dict:
    """A parallel sympy subs using multiprocess.

    Arguments
    ---------
    target_expr:
        sympy Matrix or dict which is the target for substitutions

    subs:
        dict of substitutions (e.g. {expr: replacement_expr}) or list of
        substition dicts.

    Returns
    -------
    returns the Matrix or dict which was passed.
    """
    # convert any subs paramters to list[dict]
    if isinstance(subs, dict):
        subs = [subs]
    elif isinstance(subs, Iterable):  # type:ignore
        if iter_type_check(subs, list[tuple]):
            subs = [dict(subs)]  # type:ignore
        elif iter_type_check(subs, list[dict]):
            pass
        else:
            raise Exception("unhandled case. subs TypeError")

    if isinstance(target_expr, dict):
        subs_func = dict_subs_func
        with Pool(processes=cpu_count()) as pool:
            subs = [pickle.dumps(sub) for sub in subs]  # type:ignore
            args = [(pickle.dumps((key, expr)), subs) for key, expr in target_expr.items()]
            results = [pool.apply_async(subs_func, arg) for arg in args]
            results = [pickle.loads(ret.get()) for ret in results]
        for ret in results:
            target_expr.update(ret)
        return target_expr
    elif isinstance(target_expr, Matrix):  # type:ignore
        subs_func = array_subs_func
        with Pool(processes=cpu_count()) as pool:
            subs = [pickle.dumps(sub) for sub in subs]  # type:ignore
            args = [(pickle.dumps(expr), subs) for expr in target_expr]
            results = [pool.apply_async(subs_func, arg) for arg in args]
            results = [pickle.loads(ret.get()) for ret in results]
        return Matrix(results)
    else:
        raise Exception("unhandled case. target_expr TypeError")


def cse_async(expr, id, *args, **kwargs):
    expr = pickle.loads(expr)
    temp_vars = symbols(f"X{id}_temp0:1000")
    return pickle.dumps(cse(expr, temp_vars))


def parallel_cse(matrix: Matrix) -> tuple[list, Matrix]:
    with Pool(processes=cpu_count()) as pool:
        args = [(pickle.dumps(expr), id) for id, expr in enumerate(matrix)]  # type:ignore
        results = [pool.apply_async(cse_async, arg) for arg in args]
        results = [pickle.loads(ret.get()) for ret in results]
    # replacements_iter = [i[0] for i in results]
    # replacements = []
    # for res in replacements_iter:
    #     for pair in res:
    #         replacements += [pair]
    replacements = [i[0] for i in results]
    expr_simples = Matrix([i[1] for i in results])
    return (replacements, expr_simples)


def build_model(state: Matrix,
                input: Matrix,
                dynamics: Matrix = Matrix([]),
                definitions: tuple = (),
                static: Matrix = Matrix([]),
                use_multiprocess_build: bool = True) -> tuple:
    """
    Notes:
        - state and input arrays must maintain their symbolic name.
    """

    print("=" * 50)
    print("building model...")
    print("=" * 50)

    start_time = perf_counter()

    # default symbols imposed by Sim
    t = Symbol("t")
    dt = Symbol("dt")

    ############################################################
    # give MatrixElement attr "name"
    ############################################################
    for var in state:
        if isinstance(var, MatrixElement):
            setattr(var, "name", str(var))
        elif isinstance(var, DirectUpdateSymbol):
            if isinstance(var.state_expr, MatrixElement):
                setattr(var.state_expr, "name", str(var))

    for var in input:
        if isinstance(var, MatrixElement):
            setattr(var, "name", str(var))
        elif isinstance(var, DirectUpdateSymbol):
            if isinstance(var.state_expr, MatrixElement):
                setattr(var.state_expr, "name", str(var))

    for var in static:
        if isinstance(var, MatrixElement):
            setattr(var, "name", str(var))
        elif isinstance(var, DirectUpdateSymbol):
            if isinstance(var.state_expr, MatrixElement):
                setattr(var.state_expr, "name", str(var))

    ############################################################
    # checks
    ############################################################

    # state & input array checks
    _check_var_array_types(state, "state")
    _check_var_array_types(input, "input")
    _check_var_array_types(static, "static")

    # handle formatting of provided definitions, state, input
    def_subs = _create_subs_from_definitions(definitions)

    # NOTE: check Functions of time in state array
    # which have no definition. This is a user error.
    diff_functions = [i for i in dynamics.atoms(Derivative)]
    undefined_diff_funcs = [func for func in diff_functions if func not in def_subs]
    if len(undefined_diff_funcs):
        error_msg_wrapper("Undefined Function(s) found in state:"
                          f"\n\n{undefined_diff_funcs}."
                          "\n\nEither define this expression or update this state as a DirectUpdate.")

    ############################################################
    # 1: resolve state & input to Symbols
    # gather direct update: (state_var, sub_expr) for substition
    # also gather direct update (state_var (Function), state_var (Symbol))
    # for substition into defs
    ############################################################
    def _name_to_symbolic(var):
        """this function ensures correct symbolic
        type is returned. This is to allow MatrixElements
        to be used."""
        if isinstance(var, MatrixElement):
            return var
        else:
            name = getattr(var, "name")
            return Symbol(name)

    print("resolving state & input to Symbols...")
    state_subs = {i: Symbol(i.name) for i in state.atoms(Function)}
    state_subs.update({i.state_expr: _name_to_symbolic(i.state_expr) for i in state.atoms(DirectUpdateSymbol)})

    input_subs = {i: Symbol(i.name) for i in input.atoms(Function)}
    input_subs.update({i.state_expr: _name_to_symbolic(i.state_expr) for i in input.atoms(DirectUpdateSymbol)})

    static_subs = {i: Symbol(i.name) for i in static.atoms(Function)}

    ############################################################
    # 3: get state direct update array
    ############################################################
    print("building state direct updates array...")
    definition_state_vars = flatten_list([state for (state, _) in definitions])
    definition_subs = flatten_list([sub for (_, sub) in definitions])
    state_direct_updates = []
    for expr in state:
        if isinstance(expr, DirectUpdateSymbol):
            state_direct_updates += [expr.sub_expr]
            # ensure state_expr is not Function
            expr.state_expr = _name_to_symbolic(expr.state_expr)
        # detect definition assignments that should
        # be converted to DirectUpdates. These are
        # definitions where the variable being assigned
        # exists in the state array.
        elif expr in definition_state_vars:
            id = definition_state_vars.index(expr)
            sub_expr = definition_subs[id]
            state_direct_updates += [sub_expr]
        else:
            state_direct_updates += [nan]
    state_direct_updates = Matrix(state_direct_updates)

    ############################################################
    # 4: get input direct update array
    ############################################################
    print("building input direct updates array...")
    input_direct_updates = []
    for expr in input:
        if isinstance(expr, DirectUpdateSymbol):
            input_direct_updates += [expr.sub_expr]
            # ensure state_expr is not Function
            expr.state_expr = _name_to_symbolic(expr.state_expr)
        else:
            input_direct_updates += [nan]
    input_direct_updates = Matrix(input_direct_updates)

    ############################################################
    # 5: apply definition subs to itself for recursive definitions
    ############################################################
    # print("processing recursive definition substitions...")
    # st = perf_counter()
    # for k, v in def_subs.items():
    #     def_subs[k] = v.subs(def_subs)  # .subs(state_subs).subs(input_subs)
    # print(perf_counter() - st)

    ############################################################
    # 6: apply state & input resolved Symbols to definition subs
    ############################################################
    print("applying state & input variable substitions to definitions...")

    if use_multiprocess_build:
        with Pool(processes=cpu_count()) as pool:
            subs = [pickle.dumps(state_subs),
                    pickle.dumps(input_subs),
                    pickle.dumps(def_subs),
                    pickle.dumps(static_subs)]
            args = [(pickle.dumps((key, expr)), subs) for key, expr in def_subs.items()]
            results = [pool.apply_async(dict_subs_func, arg) for arg in args]
            results = [pickle.loads(ret.get()) for ret in results]
        for ret in results:
            def_subs.update(ret)
    else:
        for k, v in def_subs.items():
            def_subs[k] = v.subs(state_subs).subs(input_subs)

    ############################################################
    # 7: apply substitions to dynamics array
    ############################################################
    ##################
    # NOTE IN FUTURE try speed ups by attempting
    # to turn expr into polynomial where each term can be
    # substituted for individually and then the full
    # expression reconstructed.
    # - poly = expr.as_poly()
    # - poly.terms() # list[(exps, coeffs), (exps, coeffs)]
    # - poly.gens # generator which provides order of terms
    ##################
    if dynamics:
        print("applying substitions to dynamics...")
        if use_multiprocess_build:
            with Pool(processes=cpu_count()) as pool:
                subs = [pickle.dumps(def_subs),
                        pickle.dumps(state_subs),
                        pickle.dumps(input_subs),
                        pickle.dumps(static_subs)]
                args = [(pickle.dumps(expr), subs) for expr in dynamics]
                results = [pool.apply_async(array_subs_func, arg) for arg in args]
                results = [pickle.loads(ret.get()) for ret in results]
            dynamics = Matrix(results)
        else:
            dynamics = dynamics.subs(def_subs)
            dynamics = dynamics.subs(state_subs).subs(input_subs)
    else:
        dynamics = Matrix([np.nan] * len(state))

    ############################################################
    # 8: apply subs to direct updates
    ############################################################
    print("applying substitions to direct state updates...")
    if use_multiprocess_build:
        with Pool(processes=cpu_count()) as pool:
            subs = [pickle.dumps(def_subs),
                    pickle.dumps(state_subs),
                    pickle.dumps(input_subs),
                    pickle.dumps(static_subs)]
            args = [(pickle.dumps(expr), subs) for expr in state_direct_updates]
            results = [pool.apply_async(array_subs_func, arg) for arg in args]
            results = [pickle.loads(ret.get()) for ret in results]
        state_direct_updates = Matrix(results)
    else:
        state_direct_updates = state_direct_updates.subs(def_subs)
        state_direct_updates = state_direct_updates.subs(state_subs).subs(input_subs)
        state_direct_updates = state_direct_updates.subs(static_subs)

    print("applying substitions to direct input updates...")
    if use_multiprocess_build:
        with Pool(processes=cpu_count()) as pool:
            subs = [pickle.dumps(def_subs),
                    pickle.dumps(state_subs),
                    pickle.dumps(input_subs),
                    pickle.dumps(static_subs)]
            args = [(pickle.dumps(expr), subs) for expr in input_direct_updates]
            results = [pool.apply_async(array_subs_func, arg) for arg in args]
            results = [pickle.loads(ret.get()) for ret in results]
        input_direct_updates = Matrix(results)
    else:
        input_direct_updates = input_direct_updates.subs(def_subs)
        input_direct_updates = input_direct_updates.subs(state_subs).subs(input_subs)
        input_direct_updates = input_direct_updates.subs(static_subs)

    ############################################################
    # check for any undefined differential expresion in dynamics
    ############################################################
    print("checking model for missing definitions...")
    dynamic_functions = [i for i in dynamics.atoms(Function) if hasattr(i, "name") and (i.args == (t,))]
    if dynamic_functions:
        raise Exception(f"Undefined Functions found in dynamics matrix:\n{dynamic_functions}")

    # gather symbols from state & input arrays
    state_symbols = [i.name for i in state]  # type:ignore
    input_symbols = [i.name for i in input]  # type:ignore
    static_symbols = [i.name for i in static]  # type:ignore

    # gather any symbols used indirectly in state & input arrays
    state_direct_update_set = state_direct_updates.atoms(Symbol)
    input_direct_update_set = input_direct_updates.atoms(Symbol)

    # gather symbols from direct updates
    state_direct_update_set.update(state.atoms(DirectUpdateSymbol))
    input_direct_update_set.update(input.atoms(DirectUpdateSymbol))

    # NOTE: conversion to str since sympy
    # vars being hashable is unknown right now
    direct_update_symbols = set()
    direct_update_symbols.update(set([i.name for i in state_direct_update_set]))
    direct_update_symbols.update(set([i.name for i in input_direct_update_set]))

    # gather symbols from dynamics
    dynamics_symbols = set([i.name for i in dynamics.atoms(Symbol)])

    total_symbols = set()
    total_symbols.update(dynamics_symbols)
    total_symbols.update(direct_update_symbols)

    # subtract defined symbols in the state & input arrays
    undefined_symbols = total_symbols.difference(set(state_symbols)).difference(set(input_symbols))
    # subtract static symbols
    undefined_symbols = undefined_symbols.difference(set(static_symbols))
    # ignore default symbols (t, dt)
    undefined_symbols = undefined_symbols.difference({t.name, dt.name})

    if undefined_symbols:
        error_msg_wrapper(f"Undefined Symbols found in model:\n{undefined_symbols}")

    # write_array(state, "./temp_state.py")
    # write_array(input, "./temp_input.py")
    # write_array(dynamics, "./temp_dynamics.py")

    # replace Functions in state, input, static arrays to Symbols
    # ------------------------------------------------------------------
    # NOTE: this was added to avoid japl.CodeGen from dealing with
    # sympy Function. Not sure if this violates logic elsewhere, though.
    # ------------------------------------------------------------------
    for i, item in enumerate(state):  # type:ignore
        if isinstance(item, Function):
            state[i] = Symbol(getattr(item, "name"))
    for i, item in enumerate(input):  # type:ignore
        if isinstance(item, Function):
            input[i] = Symbol(getattr(item, "name"))
    for i, item in enumerate(static):  # type:ignore
        if isinstance(item, Function):
            static[i] = Symbol(getattr(item, "name"))

    exec_time = perf_counter() - start_time
    print("exec time: %.3f seconds" % exec_time)

    print("=" * 50, end="\n\n")
    return (state, input, dynamics, static,
            state_direct_updates, input_direct_updates)


def error_msg_wrapper(msg) -> str:
    wrap_str = f"\n\n{'=' * 50}\n"
    raise Exception(f"{wrap_str}\n{msg}{wrap_str}")


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


@DeprecationWarning
def _get_direct_updates(array):
    return [i for i in array if isinstance(i, DirectUpdateSymbol)]


@DeprecationWarning
def _get_sub_expr(array):
    return [i.sub_expr for i in array if isinstance(i, DirectUpdateSymbol)]


def _apply_subs_to_expr(expr, subs: dict|list[dict]):
    if isinstance(subs, dict):
        return expr.subs(subs)
    elif isinstance(subs, list):
        for sub in subs:
            expr.subs(sub)
        return expr


@DeprecationWarning
def _apply_subs_to_dict(subs: dict):
    for key, val in subs.items():
        subs[key] = _apply_subs_to_expr(val, subs)
    return subs


@DeprecationWarning
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
        elif isinstance(elem, Function):
            continue
        elif isinstance(elem, DirectUpdateSymbol):
            continue
        elif isinstance(elem, MatrixSymbol):
            continue
        elif isinstance(elem, MatrixElement):
            continue
        else:
            raise Exception(f"\n\n{array_name}-id[{i}]: cannot register a variable for "
                            f"expression \n\n\t\"{elem}\":\n\n\tElements of the state array must be "
                            f"either Symbol, Function, or DirectUpdate. Add to the array a "
                            f"variable and assign to it the expression using the definitions tuple.")


@DeprecationWarning
def _check_dynamics_for_undefined_diffs(dynamics):
    """This method checks for any undefined Derivative types in the dynamics.
    undefined Derivatives indicate a missing substition in the definitions."""
    for i, elem in enumerate(dynamics):
        if elem.has(Derivative):
            raise Exception(f"\n\ndynamics-id[{i}]: undefined differential expression "
                            f"\n\n\t\"{elem}\"\n\n\t in the dynamics array. Assign this expression in "
                            f"the definitions or update using DirectUpdate().")


@DeprecationWarning
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


@DeprecationWarning
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
                    # convert std number types to sympy Float
                    if isinstance(new, float) or isinstance(new, int):
                        new = Float(new)
                    ret[old] = new
                except Exception as e:
                    raise Exception(e, "\nunhandled case. old and new need to both have '__len__'.")
    else:
        ret = sub
    return ret


@DeprecationWarning
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


@DeprecationWarning
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



def _subs_prune(replacements, expr_simple, nchunk: int = 2_000,
                nchunk_subs: int = 500) -> tuple[dict, Matrix, int]:
    # unpack to single iterable
    reps = []
    for rep in replacements:
        if isinstance(rep, tuple) and (len(rep) == 2):
            reps += [rep]
        else:
            for r in rep:
                reps += [r]

    # condense redundant replacements
    dreps = {sub: rexpr for sub, rexpr in reps}
    dreps_pops = []
    new_subs = {}  # new subs for expression to take out redundant variables

    # precompute and group subs by replacement expression
    repl_to_subs = defaultdict(list)
    # for sub, rexpr in dreps.items():
    for sub, rexpr in reps:
        repl_to_subs[rexpr].append(sub)

    # iterate over grouped expressions
    for sub, rexpr in tqdm(reps, ncols=100, desc="Pruning"):
        # if rexpr appears more than once in dict, its redundant
        if len(repl_to_subs[rexpr]) > 1:
            redundant_vars = repl_to_subs[rexpr]
            # replace redundant vars with first found var
            if redundant_vars:
                keep_var = redundant_vars[0]
                for rvar in redundant_vars[1:]:
                    new_subs.update({rvar: keep_var})
                    dreps_pops += [rvar]

    for var in dreps_pops:
        if var in dreps:
            dreps.pop(var)  # type:ignore

    #########################
    # nchunk = 2_000
    remaining_chunk = [*dreps.items()]

    if len(remaining_chunk) > nchunk:
        chunked_dicts = []
        for i in range(0, len(remaining_chunk), nchunk):
            chunk = dict(remaining_chunk[i:i + nchunk])
            chunked_dicts += [chunk]

        chunked_new_subs = []
        # nchunk_subs = 500
        new_subs_list = [*new_subs.items()]
        for i in range(0, len(new_subs), nchunk_subs):
            chunk_new_subs = dict(new_subs_list[i:i + nchunk_subs])
            chunked_new_subs += [chunk_new_subs]

        # remaining_chunk = dict(remaining_chunk[nchunk:])
        inter_reps = {}
        for chunk in tqdm(chunked_dicts, ncols=70, desc="\tdict subs",
                          ascii=" ="):
            inter_reps.update(parallel_subs(chunk, chunked_new_subs))  # type:ignore
        dreps = inter_reps
    else:
        dreps = parallel_subs(dict(remaining_chunk), [new_subs])

    replacements = [*dreps.items()]  # type:ignore

    expr_simple = parallel_subs(expr_simple, [new_subs])
    return (replacements, expr_simple, len(dreps_pops))  # type:ignore


@DeprecationWarning
def to_pycode(func_name: str,
              expr: Expr|Matrix,
              state_vars: list|Matrix,
              input_vars: list|Matrix,
              static_vars: list|Matrix = [],
              filepath: str = "",
              use_cse: bool = True,
              imports: list[str] = [],
              header_insert: str = "",
              intermediates: list[tuple] = [],
              extra_params: dict = {}):
    if not filepath:
        filepath = func_name + ".py"
    # NOTE: in development
    from sympy.codegen.ast import FunctionDefinition
    from sympy.printing.pycode import PythonCodePrinter
    from sympy.codegen.ast import real, float64
    from sympy import IndexedBase
    from sympy import pycode
    from sympy import cse
    from japl.BuildTools.CodeGeneratorBase import CodeGeneratorBase
    from textwrap import dedent
    import re

    replacements = ()
    if not use_cse:
        expr_simple = expr
    else:
        # optimize expression
        replacements, expr_simple = cse(expr)
        expr_simple = expr_simple[0]  # type:ignore

        # replacements, expr_simple = parallel_cse(expr)
        # # must further optimize and make substitutions
        # # between indices of expr
        # NCHUNK = 10_000
        # NCHUNK_SUBS = 5_000
        # MAX_PRUNE_ITER = 10
        # for _ in range(MAX_PRUNE_ITER):
        #     replacements, expr_simple, nredundant = _subs_prune(replacements, expr_simple,
        #                                                         nchunk=NCHUNK, nchunk_subs=NCHUNK_SUBS)
        #     if nredundant == 0:
        #         break

    ##########################################################################
    # Model uses standardized arg format: (t, X, U, S, dt)
    vars: list = [Symbol("t"), state_vars, input_vars, static_vars, Symbol("dt")]
    standard_args = ["t", "_X_stdarg", "_U_stdarg", "_S_stdarg", "dt"]
    assert len(vars) == len(standard_args)
    parameters = [Symbol(i, real=True) for i in standard_args]

    # user-specified extra fucntion params
    ##################################
    # NOTE this is all temporary
    ##################################
    for i, (param_name, param) in enumerate(extra_params.items()):
        # vars.insert(-1, param)
        parameters.insert(-1, Symbol(param_name, real=True))
        if hasattr(param, "__len__"):
            vars.insert(-1, param)
            standard_args.insert(-1, param_name)
    ##################################

    # unpack array-type parameters
    unpack_assignments = []
    for param, param_name in zip(vars, standard_args):
        if hasattr(param, "__len__") or hasattr(param, "shape"):
            for i, item in enumerate(flatten_list(param)):  # type:ignore
                if isinstance(item, MatrixElement):
                    # NOTE this if-stmt is temporary and can be removed.
                    # this is for temp solution for extra_parameters
                    pass
                else:
                    item_name = CodeGeneratorBase._get_expr_name(item)
                    iter_param = IndexedBase(param_name, shape=(len(param),))
                    unpack_assignments += [Assignment(Symbol(item_name, real=True), iter_param[i])]
    ##########################################################################

    mat_body = [Return(expr_simple)]  # type:ignore

    if not use_cse:
        body = unpack_assignments + mat_body
    else:

        # Build the assignments for common subexpressions
        # assignments = [Assignment(var, rep) for var, rep in replacements]
        # body = unpack_assignments + assignments + mat_body

        assignments = []
        for var, rep in replacements:
            if isinstance(rep, Piecewise):
                # NOTE: Piecewise assignment needs augmentation
                expr_str = re.sub(r"\#.*?\n", "", str(pycode(rep, strict=False)))
                assignments += [Assignment(var, Symbol(expr_str))]
            else:
                assignments += [Assignment(var, rep)]

        intermed_assigns = []
        for name, expr in intermediates:
            temp_name = re.sub(r"\[*.\]", "_", name)
            _reps, _expr = cse(expr, symbols(f"temp_{temp_name}0:500"))
            _expr = _expr[0]  # type:ignore
            for rep in _reps:
                intermed_assigns += [Assignment(*rep)]
            expr_str = re.sub(r"\#.*?\n", "", str(pycode(_expr, strict=False)))
            expr_str = expr_str.replace("(t)", "")
            intermed_assigns += [Assignment(Symbol(name), Symbol(expr_str))]

        body = unpack_assignments + intermed_assigns + assignments + mat_body

    # Define the full Python function
    func_def = FunctionDefinition(
        return_type=real,
        name=func_name,
        parameters=parameters,  # parameters for the function
        body=body
    )

    code_str = pycode(func_def, strict=False)
    # replace sympy Matrix types with numpy types
    code_str = code_str.replace("ImmutableDenseMatrix", "np.array")
    code_str = code_str.replace("MutableDenseMatrix", "np.array")
    code_str = code_str.replace("math.sin", "np.sin")
    code_str = code_str.replace("math.cos", "np.cos")
    code_str = code_str.replace("math.asin", "np.arcsin")
    code_str = code_str.replace("math.acos", "np.arccos")
    code_str = code_str.replace("math.tan", "np.tan")
    code_str = code_str.replace("math.atan", "np.arctan")
    code_str = code_str.replace("math.atan2", "np.arctan2")
    code_str = code_str.replace("math.sqrt", "np.sqrt")
    code_str = code_str.replace("math.nan", "np.nan")
    code_str = code_str.replace("Abs", "np.abs")

    # handle Piecewise inside expression
    # Regular expression to match arbitrary strings (allowing for different characters)
    # NOTE: this only works for Piecewise with a condition and a default condition
    pattern = r"Piecewise\(\(\s*([^,]+)\s*,\s*([^,]+)\s*\),\s*\(\s*([^,]+)\s*,\s*([^,]+)\s*\)\)"
    code_str = re.sub(pattern, r"(\1) if (\2) else (\3)", code_str)

    state_vars_dict = {id: CodeGeneratorBase._get_expr_name(var)
                       for id, var in enumerate(flatten_list(state_vars))}  # type:ignore
    input_vars_dict = {id: CodeGeneratorBase._get_expr_name(var)
                       for id, var in enumerate(flatten_list(input_vars))}  # type:ignore
    static_vars_dict = {id: CodeGeneratorBase._get_expr_name(var)
                        for id, var in enumerate(flatten_list(static_vars))}  # type:ignore

    import_header = ""
    for i in imports:
        import_header += i + "\n"

    code_header = (f"""\
            import numpy as np
            from sympy import symbols
            import math

            nan = np.nan
            sqrt = np.sqrt
            sin = np.sin
            cos = np.cos
            tan = np.tan

            {header_insert}

            def get_dt_var():
                return symbols("dt")

            def get_state_vars():
                state_vars = {str(state_vars_dict)}
                return symbols([*state_vars.values()])

            def get_input_vars():
                input_vars = {str(input_vars_dict)}
                return symbols([*input_vars.values()])

            def get_static_vars():
                static_vars = {str(static_vars_dict)}
                return symbols([*static_vars.values()])
            """)

    if os.path.isfile(filepath):
        os.remove(filepath)

    with open(filepath, "a+") as f:
        f.write(import_header)
        f.write(dedent(code_header))
        f.write(str(code_str))
