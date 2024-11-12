from textwrap import dedent
from japl import Atmosphere


atmos = Atmosphere()
iters = [atmos._alts,
         atmos._atmos.pressure,
         atmos._atmos.density,
         atmos._atmos.temperature_in_celsius,
         atmos._atmos.speed_of_sound,
         atmos._atmos.grav_accel]
names = ["atmosphere_alts",
         "atmosphere_pressure",
         "atmosphere_density",
         "atmosphere_temperature",
         "atmosphere_speed_of_sound",
         "atmosphere_grav_accel"]
aliases = ["_alts",
           "_pressure",
           "_density",
           "_temperature",
           "_speed_of_sound",
           "_grav_accel"]


for name, iter, alias in zip(names, iters, aliases):
    header = f"""\
    #ifndef _{name.upper()}_H_
    #define _{name.upper()}_H_

    #include <vector>

    extern std::vector<double> {alias} ="""

    count = 0
    fstr = ""
    fstr += dedent(header)
    fstr += " {\n"  # }
    for i, val in enumerate(iter):
        fstr += f"{val}"
        if count > 500:
            count = 0
            fstr += '\n'
        if i < iter.size - 1:
            fstr += ", "
        else:
            fstr += "};\n"
        count += 1
    fstr += "\n#endif"


    with open(f"_{name}.hpp", 'a+') as f:
        f.write(fstr)
