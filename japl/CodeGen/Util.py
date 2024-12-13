import os
import shutil
from sympy.codegen.ast import Expr



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
    return (expr is None) or (expr == Expr())
