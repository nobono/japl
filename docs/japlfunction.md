# JaplFunction

Symbolic Functions allow one to define a function by specifying:

- function name
- function signature
- function body

Once instantiated, a Symbolic Function can be expressed in a variety of other target languages (c++, matlab, python... etc).

---

Let's create a JaplFunction, `foo`
with no defined function body.

```py
from japl.CodeGen import ccode

class foo(JaplFunction):
    pass
```

## Scalar Parameters
Let's have the function take parmeters `(1, 2)` and looks at its code-generation output
with `c` as the target language.
```py
from sympy import Symbols

a = Symbol("a")
b = Symbol("b", integer=True)
my_func = foo(1, a, b)
```
After using Code Generation targeting c-language:
>
```c title="function call:"
foo(1, a, b)
```
```c title="function prototype:"
void foo(double& _Dummy_var1, double& a, int& b);
```
```c title="function definition:"
void foo(double& _Dummy_var0, double& a, int& b){

}
```

---

## Array-Type Parameters

```py
from sympy import Matrix

A = Matrix([1, 2, 3])
my_func = foo(A)
```
>
```c title="function call:"
foo(_Dummy_var0)
```
```c title="function prototype:"
void foo(vector<double>& _Dummy_var0);
```
```c title="function definition:"
void foo(vector<double>& _Dummy_var0){

}
```

---

## Keyword Parameters
In python, it is common to pass arguments to functions as keyword-arguments (kwargs)
`#!python myFunc(a=a)`. In languages like `c`, this is supported by passing all
keyword arguments of a `JaplFunction` to a single `std::map<string, TYPE>` parameter.

```py
from sympy import symbols

a, b = symbols("a, b")
my_func = foo(a=a, b=b)
```
>
```c title="function call:"
foo({{"a", a}, {"b", b}})
```
Here, you can see a std::map dummy parameter is established to hold all kwargs.
```c title="function prototype:"
void foo(map<string, double>& _Dummy_var0);
```
In the definition, the arguments contained within the `std::map` are automatically
accessed by there keys.
```c title="function definition:"
void foo(map<string, double>& _Dummy_var0){
   const double a = _Dummy_var0["a"];
   const double b = _Dummy_var0["b"];
   /*  */;
}
```

---

## Defining a Function Body
To define the body of a JaplFunction you can simply set the `expr` attribute in the
JaplFunction class definition.

```py
class add(JaplFunction):
    expr = a + b


my_func = add(a, b)
```
>
```c title="function call:"
add(a, b)
```
```c title="function prototype:"
double add(double& a, double& b);
```
```c title="function definition:"
double add(double& a, double& b){
   double _Ret_arg;
   _Ret_arg = a + b;
   return _Ret_arg;
}
```

---

## Defining a Method

What if we wanted to make our `JaplFunction` a method to a class instead of
a standalone function?

We can set the attributes

- **parent**: name of the parent object which this function is a method of.
- **class_name**: name of the class reference which this function belongs to.

```py
class add(JaplFunction):
    parent = "my_object"
    class_name = "MyClass"
    expr = a + b


my_func = foo(a, b)
```
>
```c title="function call:"
my_object.add(a, b)
```
```c title="function prototype:"
double add(double& a, double& b);
```
```c title="function definition:"
double MyClass::add(double& a, double& b){
   double _Ret_arg;
   _Ret_arg = a + b;
   return _Ret_arg;
}
```
