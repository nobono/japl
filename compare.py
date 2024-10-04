import os
import numpy as np
from japl.Util.Matlab import MatFile
from japl import PyQtGraphPlotter
from japl.Util.Results import Results
import pyqtgraph as pg
from pyqtgraph.exporters import ImageExporter
import argparse
from japl import JAPL_HOME_DIR

DIR = os.path.dirname(__file__)
np.set_printoptions(suppress=True, precision=3)



plotter = PyQtGraphPlotter(frame_rate=30,
                           figsize=[10, 6],
                           aspect="auto",
                           background_color="white",
                           text_color="black")

kft2m = .3048 * 1000
km2m = 1000.0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Example script using argparse")
    parser.add_argument('-n',
                        dest="filename",
                        default="run_ld_67",
                        help='filename in /data')
    args = parser.parse_args()

    filename = args.filename

    fo = MatFile(JAPL_HOME_DIR + "/data/flyout.mat").flyout  # type:ignore
    # run1 = Results.load(DIR + "/run1_ballistic.pickle")
    run1 = Results.load(JAPL_HOME_DIR + f"/data/{filename}.pickle")
    run = run1


    col = ("Thrust", "thrust")
    plotter.plot(getattr(fo, "Time"), getattr(fo, col[0]),
                 title="Thrust vs Time",
                 ylabel="Thrust (N)",
                 xlabel="Time (s)",
                 legend_name="GPOPS")
    plotter.plot(getattr(run, "t"), getattr(run, col[1]),
                 legend_name="CHAD")


    col = ("Ca", "CA")
    plotter.figure()
    plotter.plot(getattr(fo, "Time"), getattr(fo, col[0]),
                 title="CA vs Time",
                 ylabel="CA",
                 xlabel="Time",
                 legend_name="GPOPS")
    plotter.plot(getattr(run, "t"), getattr(run, col[1]),
                 legend_name="CHAD")

    col = ("Cn", "CN")
    plotter.figure()
    plotter.plot(getattr(fo, "Time"), getattr(fo, col[0]),
                 title="CN vs Time",
                 ylabel="CN",
                 xlabel="Time",
                 legend_name="GPOPS")
    plotter.plot(getattr(run, "t"), getattr(run, col[1]),
                 legend_name="ChAD")


    col = ("Angle_of_Attack", "alpha")
    plotter.figure()
    plotter.plot(getattr(fo, "Time"), getattr(fo, col[0]),
                 title="alpha vs Time",
                 ylabel="alpha (deg)",
                 xlabel="Time (s)",
                 legend_name="GPOPS")
    plotter.plot(getattr(run, "t"), np.degrees(getattr(run, col[1])),
                 legend_name="ChAD")

    # col = ("Altitude", "r_u")
    # plotter.figure()
    # plotter.plot(getattr(fo, "Time"), getattr(fo, col[0]) * km2m,
    #              title="Altitude vs Time",
    #              ylabel="Altitude (m)",
    #              xlabel="Time (s)",
    #              legend_name="GPOPS")
    # plotter.plot(getattr(run, "t"), getattr(run, col[1]),
    #              legend_name="ChAD")

    row = ("Range", "r_n")
    col = ("Altitude", "r_u")
    plotter.figure()
    plotter.plot(getattr(fo, "Range") * km2m, getattr(fo, col[0]) * km2m,
                 title="Altitude vs Range",
                 ylabel="Altitude (m)",
                 xlabel="Range (m)",
                 legend_name="GPOPS")
    plotter.plot(getattr(run, "r_n"), getattr(run, col[1]),
                 legend_name="ChAD")

    col = ("Mass", "wet_mass")
    plotter.figure()
    plotter.plot(getattr(fo, "Time"), getattr(fo, col[0]),
                 title="Mass vs Time",
                 ylabel="Mass",
                 xlabel="Time",
                 legend_name="GPOPS")
    plotter.plot(getattr(run, "t"), getattr(run, col[1]),
                 legend_name="ChAD")

    col = ("Mach", "mach")
    plotter.figure()
    plotter.plot(getattr(fo, "Time"), getattr(fo, col[0]),
                 title="Mach vs Time",
                 ylabel="Mach",
                 xlabel="Time (s)",
                 legend_name="GPOPS")
    plotter.plot(getattr(run, "t"), getattr(run, col[1]),
                 legend_name="ChAD")

    col = ("Drag", "drag")
    plotter.figure()
    plotter.plot(getattr(fo, "Time"), getattr(fo, col[0]),
                 title="Drag vs Time",
                 ylabel="Drag (N)",
                 xlabel="Time (s)",
                 legend_name="GPOPS")
    plotter.plot(getattr(run, "t"), getattr(run, col[1]),
                 legend_name="ChAD")

    # col = ("Lateral_Acceleration", "a_c_z")
    # plotter.figure()
    # plotter.plot(getattr(fo, "Time"), getattr(fo, col[0]) * 10,
    #              title="Lateral Acc vs Time",
    #              ylabel="Lateral Acc (m/s)",
    #              xlabel="Time (s)",
    #              legend_name="GPOPS")
    # plotter.plot(getattr(run, "t"), getattr(run, col[1]),
    #              legend_name="ChAD")

    # col = ("Axial_Acceleration", "a_b_x")
    # plotter.figure()
    # plotter.plot(getattr(fo, "Time"), getattr(fo, col[0]) * 10,
    #              title="Axial Acc. vs Time",
    #              ylabel="Axial Acc (m/s^2)",
    #              xlabel="Time (s)",
    #              legend_name="GPOPS")
    # plotter.plot(getattr(run, "t"), getattr(run, col[1]),
    #              legend_name="ChAD")

    col = ("Velocity", "vel_mag_e")
    plotter.figure()
    plotter.plot(getattr(fo, "Time"), getattr(fo, col[0]) * 1000,
                 title="Velocity vs Time",
                 ylabel="Velocity (m/s)",
                 xlabel="Time (s)",
                 legend_name="GPOPS")
    plotter.plot(getattr(run, "t"), getattr(run, col[1]),
                 legend_name="ChAD")


    plotter.show()

    items = plotter.wins[0].centralWidget.items  # type:ignore
    for pitem in items:
        export = ImageExporter(pitem)
        export.export('./data/test.png')
    pass

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
