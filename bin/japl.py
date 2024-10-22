#!/usr/bin/env python

import os
import subprocess
# import curses
import argparse
from textwrap import dedent

__JAPL_EXT_MODULE_INIT_HEADER = "#__japl_extension_module__"



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



def find_models(start_dir: str = '.', max_depth: int = 20) -> tuple:
    init_file = "__init__.py"
    build_file = "build.py"
    model_paths = []
    model_names = []

    # Get the length of the start directory path for depth calculation
    start_depth = start_dir.rstrip(os.path.sep).count(os.path.sep)

    for root, dirs, files in os.walk(start_dir):

        # Calculate the current depth by counting the separators in the path
        current_depth = root.count(os.path.sep) - start_depth
        if current_depth >= max_depth:
            # return early
            return (model_names, model_paths)

        # look for build.py & __init__.py in dir
        if (build_file in files) and (init_file in files):
            # look for JAPL ext_module header
            first_line = ""
            with open(os.path.join(os.path.dirname(root), init_file)) as f:
                first_line = f.readline()
                first_line = first_line.lower().replace(" ", "").strip("\n")
            if first_line == __JAPL_EXT_MODULE_INIT_HEADER:
                model_names += [os.path.dirname(root)]
                model_paths += [os.path.join(root)]
    return (model_names, model_paths)



def build_model(dir: str):
    # if dir:
    #     build_file_path = os.path.join(dir, "build.py")
    #     dir_exists = os.path.isdir(dir)
    #     build_file_exists = os.path.isfile(build_file_path)

    #     if dir_exists and build_file_exists:
    #         result = subprocess.run(["python", build_file_path],
    #                                 capture_output=True, text=True, check=True)
    # else:
    #     pass
    pass


def main():
    parser = argparse.ArgumentParser(description="")
    subparsers = parser.add_subparsers(dest="command", help="Available Commands")
    list_parser = subparsers.add_parser("list", help="List of available models")
    build_parser = subparsers.add_parser("build", help="Build an available model")

    build_parser.add_argument("path",
                              default="",
                              nargs="?",
                              help="Name of available models to build")
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
            row_format_str = "{:<10} {:<15} {:<20}"
            model_names, dirs = find_models(args.path)
            print("-" * 50)
            print(row_format_str.format("id", "name", "path"))
            print("-" * 50)
            for id, (name, dir) in enumerate(zip(model_names, dirs)):
                print(row_format_str.format(id, name, dir))

        case "build":
            build_model(dir=args.path)

        case _:
            print(dedent(japl_header))
            parser.print_help()  # Show help if no command is provided


if __name__ == "__main__":
    # curses.wrapper(main)
    main()
