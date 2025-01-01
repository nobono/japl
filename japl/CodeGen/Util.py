import os
import shutil
from tqdm import tqdm
from collections import defaultdict
from sympy import cse
from sympy import Matrix
from sympy.codegen.ast import NoneToken
from sympy.codegen.ast import Token
from sympy.codegen.ast import Expr
from japl.BuildTools.BuildTools import parallel_subs



def copy_dir(source_dir, target_dir) -> None:
    """
    Recursively copies all directories and files from source_dir to target_dir.

    Parameters:
    -----------
        source_dir (str): The source directory to copy from.
        target_dir (str): The target directory to copy to.

    Raises:
    -------
        ValueError: If source_dir does not exist or is not a directory.
    """
    if not os.path.isdir(source_dir):
        raise ValueError(f"Source directory '{source_dir}' does not exist or is not a directory.")

    # Ensure the target directory exists
    os.makedirs(target_dir, exist_ok=True)

    for item in os.listdir(source_dir):
        source_item = os.path.join(source_dir, item)
        target_item = os.path.join(target_dir, item)

        if os.path.isdir(source_item):
            # Recursively copy directories
            copy_dir(source_item, target_item)
        else:
            # Copy files
            shutil.copy2(source_item, target_item)


def is_empty_expr(expr):
    return ((expr is None)
            or expr == Expr()
            or isinstance(expr, NoneToken))


def subs_prune(replacements, expr_simple) -> tuple[dict, Matrix, int]:
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
    nchunk = 2_000
    remaining_chunk = [*dreps.items()]

    if len(remaining_chunk) > nchunk:
        chunked_dicts = []
        for i in range(0, len(remaining_chunk), nchunk):
            chunk = dict(remaining_chunk[i:i + nchunk])
            chunked_dicts += [chunk]

        chunked_new_subs = []
        nchunk_subs = 500
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


def optimize_expression(expr: Expr|Matrix|Token,
                        # params: list|tuple|Matrix,
                        # return_name: str = "_Ret_arg",
                        use_cse: bool = True,
                        # is_symmetric: bool = False,
                        by_reference: dict = {}):
    MAX_PRUNE_ITER = 10

    replacements = []
    expr_simple = expr
    if use_cse:
        # old method
        # expr_replacements, expr_simple = cse(expr, symbols("X_temp0:1000"), optimizations='basic')

        # NOTE: handles MatrixSymbols in expression.
        # wrapping in Matrix() simplifies the form of
        # any matrix operation expressions.
        if expr.is_Matrix:
            expr = Matrix(expr)

            ######################################################
            # optimize pass-by-reference exprs
            ######################################################
            # add by_reference expressions to expr and
            # do cse optimization to get substitutions.
            # then split main expr & pass-by-reference
            # expression again for individual processing
            by_ref_nadds = len(by_reference)
            if by_ref_nadds > 0:
                by_ref_expr = Matrix([*by_reference.values()])
                expr = Matrix([*expr, *by_ref_expr])
            ######################################################

            replacements, expr_simple = cse(expr)
            expr_simple = expr_simple[0]  # type:ignore

            # must further optimize and make substitutions
            # between indices of expr
            for _ in range(MAX_PRUNE_ITER):
                replacements, expr_simple, nredundant = subs_prune(replacements, expr_simple)
                if nredundant == 0:
                    break

            ######################################################
            # optimize pass-by-reference exprs
            ######################################################
            if by_ref_nadds > 0:
                expr_simple = expr_simple[:-by_ref_nadds]
                for i, (k, v) in enumerate(by_reference.items()):
                    by_reference[k] = expr_simple[-by_ref_nadds:][i]  # type:ignore
            ######################################################

            # remove added reference expr which were
            # added for cse()
            if by_ref_nadds > 0:
                expr = Matrix(expr[:-by_ref_nadds])

    return replacements, expr_simple
