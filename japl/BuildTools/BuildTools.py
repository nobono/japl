from typing import Optional
from sympy import Matrix, Symbol
from japl.Model.StateRegister import StateRegister
from japl.BuildTools.DirectUpdate import DirectUpdateSymbol



def build_model(state: Matrix,
                input: Matrix,
                dynamics: Matrix,
                definitions: tuple = ()) -> tuple:
    # handle formatting of provided definitions, state, input
    diff_sub = _create_subs_from_definitions(definitions)
    state_sub = _create_subs_from_vars(state)
    input_sub = _create_subs_from_vars(input)
    # process DirectUpdate.expr with diff & state subs
    _apply_subs_to_direct_updates(state, [diff_sub, state_sub, input_sub])
    _apply_subs_to_direct_updates(input, [diff_sub, state_sub, input_sub])
    # now that subs applied to DirectUpdateSymbols,
    # update state & input subs again
    state_sub = _create_subs_from_vars(state)
    input_sub = _create_subs_from_vars(input)
    # do substitions
    dynamics = dynamics.subs(diff_sub).doit()
    dynamics = dynamics.subs(state_sub).subs(input_sub)

    return (state, input, dynamics)


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


def _create_subs_from_vars(sub: tuple|list|Matrix) -> dict:
    """This method generates a 'subs' dict from provided
    symbolic variables (Symbol, Function, Matrix). This is
    used for substition of variables into a sympy expression.
        - handles substitions passed as Matrix"""
    assert hasattr(sub, "__len__")
    ret = {}
    # for each element get the name
    for var in sub:
        if hasattr(var, "__len__"): # if Matrix
            for elem in var: #type:ignore
                ret[elem] = StateRegister._extract_variable(elem)
        elif isinstance(var, DirectUpdateSymbol):
            assert var.state_expr is not None
            ret[var.state_expr] = var.sub_expr
        else:
            ret[var] = StateRegister._extract_variable(var)
    return ret


def _apply_subs_to_direct_updates(state: Matrix, subs: dict|list[dict]) -> None:
    """applies substitions to DirectUpdateSymbol.expr which is the expression
    that is lambdified for direct state updates."""
    for var in state:
        if isinstance(var, DirectUpdateSymbol):
            assert var.sub_expr is not None     #type:ignore
            if isinstance(subs, dict):
                var.sub_expr = var.sub_expr.subs(subs)   #type:ignore
            elif isinstance(subs, list):
                for sub in subs:
                    var.sub_expr = var.sub_expr.subs(sub)   #type:ignore
            else:
                raise Exception("unhandled case.")

