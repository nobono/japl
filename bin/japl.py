#!/usr/bin/env python

import os
import sys
import importlib.util
# import curses
import argparse
from textwrap import dedent
from pathlib import Path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

__JAPL_EXT_MODULE_INIT_HEADER__ = "#__japl_extension_module__"
__JAPL_MODEL_SOURCE_HEADER__ = "#__japl_model_source__"

__IGNORE_DIRS__ = [".git", "build", "bin", "src", "include", "libs", "tests", "typings",
                   "__pycache__"]


# def main(stdscr):
#     header = """\
#             ========================================
#                             JAPL
#                 Just Another Prototyping Layer
#             ========================================
#             """
#     # Keep existing content visible
#     stdscr.leaveok(True)

#     # stdscr.clear()

#     curses.curs_set(0)

#     # Display some text at row 5, column 10
#     stdscr.addstr(0, 10, "======================================")

#     # Display instructions
#     stdscr.addstr(7, 10, "Press 'q' to quit.")

#     # Refresh the screen to show changes
#     stdscr.refresh()

#     # Main loop to wait for user input
#     while True:
#         key = stdscr.getch()

#         # Quit on 'q'
#         if key == ord('q'):
#             break


def get_root_dir():
    # Find the module spec for the japl package
    spec = importlib.util.find_spec("japl")
    if spec is None or spec.origin is None:
        raise RuntimeError("japl package is not installed or cannot be found.")
    # Get the root directory of the japl package
    root_dir = os.path.dirname(spec.origin)
    return root_dir


def file_is_model_source(root: str, file: str):
    # look for model source files in dir
    # look for JAPL model source header
    first_line = ""
    is_pyfile = file.split('.')[-1] == "py"
    not_dot_file = not file[0] == '.'
    if not_dot_file and is_pyfile:
        with open(os.path.join(root, file)) as f:
            first_line = f.readline()
        first_line = first_line.lower().replace(" ", "").strip("\n")
        if first_line == __JAPL_MODEL_SOURCE_HEADER__:
            return True
        else:
            return False


def dir_is_ext_module(root: str, files: list):
    init_file = "__init__.py"
    build_file = "build.py"
    # for root, dirs, files in os.walk(path):
    if (build_file in files) and (init_file in files):  # look for build.py & __init__.py in dir
        with open(os.path.join(root, init_file)) as f:  # look for JAPL ext_module header
            first_line = f.readline()
        first_line = first_line.lower().replace(" ", "").strip("\n")
        if first_line == __JAPL_EXT_MODULE_INIT_HEADER__:
            return True
        else:
            return False


def find_ext_models(start_dir: str = '.', max_depth: int = 20) -> dict:
    """finds already build japl Models which may or may not be already compiled."""
    found_models = {}
    ignores = __IGNORE_DIRS__

    # Get the length of the start directory path for depth calculation
    start_depth = start_dir.rstrip(os.path.sep).count(os.path.sep)

    for root, dirs, files in os.walk(start_dir):
        # skip ignored dirs
        for ignore in ignores:
            if ignore in dirs:
                dirs.remove(ignore)

        # Calculate the current depth by counting the separators in the path
        current_depth = root.count(os.path.sep) - start_depth
        if current_depth >= max_depth:
            # return early
            return found_models

        if dir_is_ext_module(root, files):
            model_name = os.path.basename(root)
            model_path = os.path.join(root)
            found_models[model_name] = model_path

    return found_models


def find_src_models(start_dir: str = '.', max_depth: int = 20) -> dict:
    """finds source code which is intended to generate japl Models."""
    found_models = {}
    ignores = __IGNORE_DIRS__

    # Get the length of the start directory path for depth calculation
    start_depth = start_dir.rstrip(os.path.sep).count(os.path.sep)

    for st_dir in [start_dir, get_root_dir()]:
        for root, dirs, files in os.walk(st_dir):
            # skip ignored dirs
            for ignore in ignores:
                if ignore in dirs:
                    dirs.remove(ignore)

            # Calculate the current depth by counting the separators in the path
            current_depth = root.count(os.path.sep) - start_depth
            if current_depth >= max_depth:
                # return early
                return found_models

            for file in files:
                if file_is_model_source(root, file):
                    model_name = os.path.basename(file.split('.')[0])
                    model_path = os.path.join(root, file)
                    found_models[model_name] = model_path

    return found_models



def build_model(found_models: dict, **kwargs):
    model_id = kwargs.get("id")
    model_name = kwargs.get("name")
    target_lang = kwargs.get("target_lang", "-py")

    if model_id is not None:
        src_path = Path([*found_models.values()][model_id])
    elif model_name is not None:
        src_path = Path(found_models[model_name])
    else:
        raise Exception("unhandled case.")
    # import_first = '.'.join(src_path.parts[:-1])
    # import_second = src_path.parts[-1].split('.')[0]
    # load_module_str = f"from {import_first} import {import_second}"
    os.system(f"python {src_path} {target_lang}")  # run model source file


def show(found_models: dict):
    row_format_str = "{:<10} {:<25} {:<25}"
    print("-" * 50)
    print(row_format_str.format("id", "name", "path"))
    print("-" * 50)
    for id, (name, dir) in enumerate(found_models.items()):
        print(row_format_str.format(id, name, dir))


def main():
    parser = argparse.ArgumentParser(description="")
    subparsers = parser.add_subparsers(dest="command", help="Available Commands")
    list_parser = subparsers.add_parser("list", help="List of available models")
    build_parser = subparsers.add_parser("build", help="Build an available model")

    build_parser.add_argument("id",
                              default=None,
                              type=int,
                              nargs="?",
                              help="id of available models to build")
    build_parser.add_argument("-p", "--path",
                              default=".",
                              type=str,
                              nargs="?",
                              help="Name of available models to build")
    build_parser.add_argument("-py", "--python",
                              action="store_true",
                              help="target language")
    build_parser.add_argument("-c", "--c",
                              action="store_true",
                              help="target language")
    build_parser.add_argument("-n", "--name",
                              default=None,
                              type=str,
                              nargs="?",
                              help="name (filename) of available models to build")
    list_parser.add_argument("path",
                             default=".",
                             nargs="?",
                             help="Root path to begin search for model directories")


    japl_header = """\
            =============================================
                                JAPL
                    Just Another Prototyping Layer
            =============================================
            """

    # Parse the arguments
    args = parser.parse_args()

    # Check which command was given and call the corresponding function
    match args.command:
        case "list":
            found_models = find_ext_models(args.path)
            show(found_models)

        case "build":
            found_models = find_src_models(args.path)
            show(found_models)
            if (args.id is not None) or (args.name is not None):

                # choose target language to build
                if args.python:
                    target_lang = "--py"
                elif args.c:
                    target_lang = "-c"
                else:
                    target_lang = "-py"

                kwargs = {"id": args.id, "target_lang": target_lang}
                build_model(found_models, **kwargs)

        case _:
            print(dedent(japl_header))
            parser.print_help()  # Show help if no command is provided


if __name__ == "__main__":
    # curses.wrapper(main)
    main()
