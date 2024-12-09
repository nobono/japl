from io import TextIOWrapper
import os
import shutil
from typing import Any, Optional
from sympy.codegen.ast import FunctionCall
from sympy.codegen.ast import FunctionPrototype
from sympy.codegen.ast import FunctionDefinition
from sympy.codegen.ast import Token
from sympy.codegen.ast import String
from sympy.codegen.ast import Tuple
from sympy.codegen.ast import Type
from sympy.codegen.ast import Node
from sympy.codegen.ast import Basic
from sympy.core.function import Function
from sympy import Float, Integer, Matrix
from sympy import MatrixSymbol
from sympy.printing.c import value_const
from japl.CodeGen.JaplFunction import JaplFunction
from japl.CodeGen.Util import ccode
from pathlib import Path



class Builder:

    name: str
    data: dict = {}
    code_type: str
    writes: list = []

    def __init__(self, *args, **kwargs) -> None:
        pass


    def add_item(self, name: str, ast_nodes: list|tuple|Basic):
        if not hasattr(ast_nodes, "__len__"):
            ast_nodes = [ast_nodes]
        self.name = name
        self.data[name] = ast_nodes


    def build(self, code_type: str):
        for name, ast_nodes in self.data.items():
            for node in ast_nodes:
                self.writes += [node._build(code_type=code_type)]


    def dumps(self, file):
        """Optional method to write build strings to file."""
        pass


class FileBuilder(Builder):


    def __init__(self, name: str, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.check_filename(name)
        self.name = str(name)


    @staticmethod
    def check_filename(filename: str):
        if '.' not in filename:
            raise Exception("filename misssing extension.")


    def add_item(self, name: str, ast_nodes: list|tuple|Basic):
        super().add_item(name, ast_nodes)


    def build(self, code_type: str):
        for name, ast_nodes in self.data.items():
            for node in ast_nodes:
                if isinstance(node, JaplFunction):
                    node._build(code_type=code_type)
                    self.writes += [str(ccode(node.get_def()))]
                else:
                    raise Exception("unhandled case.")


    def dumps(self, file):
        for line in self.writes:
            file.write(line + "\n")


class CFileBuilder(FileBuilder):
    code_type = "c"


class CodeGenerator:


    @staticmethod
    def build_ext_module(code_type: str):
        pass


    @staticmethod
    def build_file(builder: FileBuilder, file: Optional[Any] = None):
        if not hasattr(builder, "code_type"):
            raise Exception("Builder class has no defined output language "
                            f"ensure attribute \"code_type\" is defined.")
        code_type = builder.code_type
        builder.build(code_type)
        if file:
            builder.dumps(file)
            file.close()
        else:
            file = CodeGenerator.create_file(name=builder.name, path="./")
            builder.dumps(file)
            file.close()


    @staticmethod
    def create_file(name: str, path: str):
        file_path = Path(path, name)
        if os.path.exists(file_path) and os.path.isfile(file_path):
            input_str = f"{file_path} already exists. Overwrite? (y/n):"
            if input(input_str).strip().lower() == "y":
                shutil.rmtree(file_path)
                return open(file_path, "a+")
            else:
                print("exiting.")
                quit()
        else:
            raise Exception(f"path {file_path} does not exist.")
