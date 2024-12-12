import os
import shutil
from sympy.codegen.ast import numbered_symbols
from japl.CodeGen.Globals import _STD_DUMMY_NAME



__numbered_dummy_symbol_gen = numbered_symbols(prefix=_STD_DUMMY_NAME)


def get_dummy_symbol():
    return next(__numbered_dummy_symbol_gen)


def reset_dummy_symbol_gen():
    global __numbered_dummy_symbol_gen
    __numbered_dummy_symbol_gen = numbered_symbols(prefix=_STD_DUMMY_NAME)


# def ccode(expr, **kwargs):
#     printer = CCodeGenPrinter()
#     return printer.doprint(expr, **kwargs)


# def pycode(expr, **kwargs):
#     printer = PyCodeGenPrinter()
#     return printer.doprint(expr, **kwargs)


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
