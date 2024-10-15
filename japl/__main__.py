import os
import subprocess
import curses
import argparse
from textwrap import dedent



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



def find_models(dir: str = '.') -> tuple:
    filename = "build.py"
    model_paths = []
    model_names = []
    for root, dirs, files in os.walk(dir):
        if filename in files:
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
