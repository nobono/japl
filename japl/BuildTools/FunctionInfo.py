from sympy import Expr, Matrix



class FunctionInfo:
    def __init__(self,
                 name: str,
                 expr: Expr|Matrix,
                 params: list,
                 return_name: str = "_ret_std",
                 use_cse: bool = True,
                 is_symmetric: bool = False,
                 description: str = "",
                 by_reference: dict = {}) -> None:
        self.name = name
        self.expr = expr
        self.params = params
        self.return_name = return_name
        self.use_cse = use_cse
        self.is_symmetric = is_symmetric
        self.description = description
        self.by_reference = by_reference

        # info added in build process
        self.body: str = ""
        self.params_list: list[str]
        self.params_unpack: list[str]
        self.proto: str
