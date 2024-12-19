# Examples

---

Create a Symbolic Function (`JaplFunction`)
---

Symbolic Functions allow one to define a function by specifying:

- function name
- function signature
- function body

Once instantiated, a Symbolic Function can be expressed in a variety of other target languages (c++, matlab, python... etc).

---

Using sympy, let's create a Symbolic Function for discrete linear motion:

$$
X =
\begin{bmatrix}
p \\
v
\end{bmatrix}
$$

$$
\begin{gather}
p_{k+1} =& p_k + v_k * \Delta t\\
v_{k+1} =& v_k + a_k * \Delta t
\end{gather}
$$

---

First, let's define the expression for what we want to function to represent.
```py
from sympy import Symbol

t = Symbol("t")
dt = Symbol("dt")
pos = Symbol("pos")
vel = Symbol("vel")
acc = Symbol("acc")

pos_new = pos + vel * dt
vel_new = vel + acc * dt

state = Matrix([pos, vel])
state_new = Matrix([pos_new, state_new])
```

Now, create a subclass of `JaplFunction` and call our function `linear_motion`.
The function body can be defined by setting the class member `expr`.
```py
from japl import JaplFunction

class linear_motion(JaplFunction):
    expr = state_new
```

Initialization of the subclass determines the function signature.
Whatever format is used to initialize will be reflected in how the function
parameters are passed and, in applicable target languages, the signature of the
function prototype.
```py
# instantiate the function to establish function signature
func = linear_motion(pos, vel, acc, dt)
```

We're all done!

Either `func` or `linear_motion(pos, vel, acc, dt)` can be used inline with other your other code
and represent equivalent forms of the same expression defined by `linear_motion`.
>
```py
result = (func + 1) / 2
```
```py
result = (linear_motion(pos, vel, acc, dt) + 1) / 2
```

---

Code Generation
---

code generation is dynamic for a specified target language so a symbolic function must be built.
> Typically, you will not have to build anything manually. This example is just to take a look under the hood.

```py
from japl.CodeGen import ccode

func._build("c")
code_str = ccode(func.function_def)
print(code_str)
```

> output:
```c

vector<double> linear_motion(double& pos, double& vel, double& acc, double& dt){
   vector<double> _Ret_arg = vector<double>(2);
   _Ret_arg[0] = dt*vel + pos;
   _Ret_arg[1] = acc*dt + vel;
   return _Ret_arg;
}
```

or we could target the matlab / octave language
```py
from japl.CodeGen import octave_code

func._build("octave")
code_str = octave_code(func.function_def)
print(code_str)
```

> output:
```matlab
function [_Ret_arg] = linear_motion(pos, vel, acc, dt)
  _Ret_arg = zeros(2, 1);
  _Ret_arg(1, 1) = dt.*vel + pos;
  _Ret_arg(2, 1) = acc.*dt + vel;
end
```

---

Full Example Script
---

```py
--8<-- "./examples/linear_motion_discrete.py"
```
