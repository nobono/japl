from sympy.codegen.ast import Token
from sympy.codegen.ast import String
from sympy import Function
from sympy import Basic
from sympy import Symbol
from japl.CodeGen.JaplFunction import JaplFunction



class JaplClass(Token):

    """Ast Token for generating classes."""

    __slots__ = _fields = ("name", "parent", "members")
    defaults = {"name": String("JaplClass"),
                "parent": String(""), "members": {}}

    _construct_name = String
    _construct_parent = String
    _construct_member = dict

    # @staticmethod
    # def _construct_members(val):
    #     return tuple([*val])


    def _build(self, code_type: str) -> None:
        # -----------------------------------------------------------------
        # NOTE:
        # this code is only here to add:
        # `self.__some_method__` (for py-lang)
        # or
        # `this->__some_method__` (for c-lang).
        # -----------------------------------------------------------------
        for key in self.members.keys():
            member = self.members[key]
            if isinstance(member, JaplFunction):
                self.members[key].header_def = [
                        Symbol("aerotable = self.aerotable"),
                        Symbol("atmosphere = self.atmosphere"),
                        ]


# ----------------------------------------------------------------
# NOTE:
# below is attempt to rewrite JaplClass so it can be defined like
# JaplFunction.
# ----------------------------------------------------------------
# class ClassClass(type):

#     __members__: list
#     __methods__: list


#     @classmethod
#     def __prepare__(cls, name, bases, **kwds):  # type:ignore
#         return {"__members__": {}, "__methods__": {}}


#     def __new__(cls, name, bases, dct):
#         # Create the class as usual
#         obj = super().__new__(cls, name, bases, dct)

#         # Intercept during class creation
#         for key, val in dct.items():
#             if "__" in key:
#                 continue
#             if isinstance(val, Function):
#                 obj.__methods__[key] = val
#             elif isinstance(val, Basic):
#                 obj.__members__[key] = val
#             elif isinstance(val, (str, String)):
#                 obj.__members__[key] = val
#             else:
#                 obj.__members__[key] = val

#         return obj


#     def __setattr__(cls, key, value):
#         # Intercept after class creation
#         # print(f"Intercepted setattr: {key} = {value}")
#         super().__setattr__(key, value)


# class JaplClass(Token, metaclass=ClassClass):

#     __slots__ = _fields = ("name",
#                            "parent",
#                            "class_args",
#                            "class_kwargs",
#                            "class_call",
#                            "class_body")

#     defaults = {"name": String(""),
#                 "parent": String(""),
#                 "class_args": (),
#                 "class_kwargs": {},
#                 "class_call": None,
#                 "class_body": None}
