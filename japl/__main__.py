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



def build_model(dir: str):
    build_file_path = os.path.join(dir, "build.py")
    dir_exists = os.path.isdir(dir)
    build_file_exists = os.path.isfile(build_file_path)

    if dir_exists and build_file_exists:
        ret = subprocess.run(["python", build_file_path],
                             capture_output=True, text=True, check=True)


def list_function(dir: str = '.'):
    dirs = subprocess.run(["find", dir, "-name", "build.py"],
                          capture_output=True, text=True, check=True)
    for dir in dirs.stdout.split("\n"):  # type:ignore
        dir = os.path.dirname(dir)
        filename = os.path.split(dir)
        if len(filename):
            print(filename[-1])
    # ret = os.system("find . -name build.py")
    # print(res.stdout)


def main():
    parser = argparse.ArgumentParser(description="")
    subparsers = parser.add_subparsers(dest="command",
                                       help="Available Commands")
    list_parser = subparsers.add_parser("list",
                                        help="List of available models")
    build_parser = subparsers.add_parser("build",
                                        help="List of available models")

    # Parse the arguments
    args = parser.parse_args()

    # list_args = list_parser.parse_args()

    # Check which command was given and call the corresponding function
    match args.command:
        case "list":
            list_function()
        # case "build":
        #     build_model()
        case _:
            parser.print_help()  # Show help if no command is provided

if __name__ == "__main__":
    # curses.wrapper(main)
    main()
