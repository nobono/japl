import os
import numpy as np
from japl.Util.Matlab import MatFile
from japl import PyQtGraphPlotter
from japl.Util.Results import Results
DIR = os.path.dirname(__file__)
np.set_printoptions(suppress=True, precision=3)


plotter = PyQtGraphPlotter(frame_rate=30,
                           figsize=[10, 6],
                           aspect="auto",
                           background_color="white",
                           text_color="black")


kft2m = .3048 * 1000

fo = MatFile(DIR + "/../../../data/flyout.mat").flyout  # type:ignore
run1 = Results.load(DIR + "/run1_ballistic.pickle")
run = run1


# col = ("Thrust", "thrust")
# plotter.plot(getattr(fo, "Time"), getattr(fo, col[0]),
#              title="Thrust vs. Time",
#              ylabel="Thrust (N)",
#              xlabel="Time (s)")
# plotter.plot(getattr(run, "t"), getattr(run, col[1]))


# col = ("Ca", "CA")
# plotter.figure()
# plotter.plot(getattr(fo, "Time"), getattr(fo, col[0]),
#              title="CA vs. Time",
#              ylabel="CA",
#              xlabel="Time")
# plotter.plot(getattr(run, "t"), getattr(run, col[1]))

# col = ("Cn", "CN")
# plotter.figure()
# plotter.plot(getattr(fo, "Time"), getattr(fo, col[0]),
#              title="CN vs. Time",
#              ylabel="CN",
#              xlabel="Time")
# plotter.plot(getattr(run, "t"), getattr(run, col[1]))


# col = ("Angle_of_Attack", "alpha")
# plotter.figure()
# plotter.plot(getattr(fo, "Time"), getattr(fo, col[0]),
#              title="alpha vs. Time",
#              ylabel="alpha (deg)",
#              xlabel="Time (s)")
# plotter.plot(getattr(run, "t"), np.degrees(getattr(run, col[1])))

# col = ("Altitude", "r_u")
# plotter.figure()
# plotter.plot(getattr(fo, "Time"), getattr(fo, col[0]) * kft2m,
#              title="Altitude vs. Time",
#              ylabel="Altitude (m)",
#              xlabel="Time (s)")
# plotter.plot(getattr(run, "t"), getattr(run, col[1]))

# row = ("Range", "r_n")
# col = ("Altitude", "r_u")
# plotter.figure()
# plotter.plot(getattr(fo, "Range") * kft2m, getattr(fo, col[0]) * kft2m,
#              title="Altitude vs. Range",
#              ylabel="Altitude (m)",
#              xlabel="Range (m)")
# plotter.plot(getattr(run, "r_n"), getattr(run, col[1]))

# col = ("Mass", "wet_mass")
# plotter.figure()
# plotter.plot(getattr(fo, "Time"), getattr(fo, col[0]),
#              title="",
#              ylabel="",
#              xlabel="")
# plotter.plot(getattr(run, "t"), getattr(run, col[1]))

col = ("Mass", "wet_mass")
plotter.figure()
plotter.plot(getattr(fo, "Time"), getattr(fo, col[0]),
             title="",
             ylabel="",
             xlabel="")
plotter.plot(getattr(run, "t"), getattr(run, col[1]))

plotter.show()

# Acceleration
# Altitude
# Angle_of_Attack
# Angle_of_Attack_Rate
# Atmospheric_Density
# Atmospheric_Pressure
# Atmospheric_Temperature
# Axial_Acceleration
# Azimuth
# Ballistic_Coeff
# Bank_Angle
# Bank_Rate
# Ca
# Cn
# Crossrange
# Downrange
# Drag
# Dynamic_Pressure
# Eastward_Acceleration
# FPA
# FPA_TVC_Angle
# FPA_TVC_Angle_Rate
# Fin_Deflection
# Fin_Slew_Rate
# Geocentric_Latitude
# Heat_Flux_Rate
# Heat_Flux_Total
# Heating_Rate
# Icg
# Impact_Latitude
# Impact_Longitude
# Inertial_Azimuth
# Inertial_FPA
# Inertial_Pitch
# Inertial_Roll
# Inertial_Velocity
# Inertial_Yaw
# Kinetic_Energy
# Kn
# Lateral_Acceleration
# Latitude
# Lift
# Longitude
# Mach
# Mass
# Northward_Acceleration
# Nozzle_Separation
# Potential_Energy
# Ptot
# Range
# Re
# Side_Force
# Stagnation_Heating
# Static_Margin
# Thrust
# Time
# Total_Distance
# Total_Energy
# Upward_Acceleration
# Velocity
# XCP
# Xcg
# Yaw
# Yaw_Acceleration
