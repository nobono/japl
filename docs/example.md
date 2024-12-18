# Examples

---

Create a Symbolic Function
---

Using sympy, let's create a Symbolic Function for linear motion.
First setup the model variables
```py
from sympy import Symbol

t = Symbol("t")
dt = Symbol("dt")

pos = Symbol("pos")
vel = Symbol("vel")
acc = Symbol("acc")

state = Matrix([pos, vel])
```

We then define the model expression for how the state evolves.

```py
pos_new = pos + vel * dt
vel_new = vel + acc * dt
state_new = Matrix([pos_new, state_new])
```

Now, lets create a class called `linear_motion` as a subclass of `JaplFunction` to get the
```py
from japl import JaplFunction

# ----------------------------------------------
# setup a subclass of JaplFunction to establish:
#   - function name: linear_motion
#   - function body: state_new
# ----------------------------------------------
class linear_motion(JaplFunction):
    expr = state_new

# instantiate the function to establish function parameters
func = linear_motion(acc)
```

---

Code Generation
---

code generation is dynamic for a specified target language so a symbolic function must be built.
> Typically, you will not have to build anything manually. This example is just to take a look under the hood.

```py
from japl.CodeGen import ccode

target_lang = "c"
func._build(target_lang)
```

Taking a look at the function definition
```py
code_str = ccode(func.function_def)
print(code_str)
```

> output:
```c
vector<double> linear_motion(double& acc){
   vector<double> _Ret_arg = vector<double>(2);
   _Ret_arg[0] = dt*vel + pos;
   _Ret_arg[1] = acc*dt + vel;
   return _Ret_arg;
}
```
