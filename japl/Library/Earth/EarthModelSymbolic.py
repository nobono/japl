import numpy as np
from sympy import Symbol, symbols
from sympy import deg, sqrt
from sympy import sin, cos, atan2
from sympy import Float
# from japl import Model
# from japl.BuildTools.DirectUpdate import DirectUpdate



def cosd(ang):
    return deg(cos(ang))


def sind(ang):
    return deg(sin(ang))


def atan2d(y, x):
    return deg(atan2(y, x))


class EarthModelSymbolic:

    """This class is an Earth Ellipsoid Reference Model based on
    WGS84."""

    def __init__(self) -> None:
        self.mu = 3986004.418e8              # (m^3 / s^2), Earth gravitational parameter
        self.J2 = 0.10826299890519e-2        # dimensionless, J2
        self.omega = 7.2921159e-5            # (rad / s), Earth angular velocity
        self.semimajor_axis = 6_378_137.0    # meters
        self.inv_flattening = 298.257223563  # dimensionless
        self.flattening = 1 / 298.257223563  # dimensionless
        self.semiminor_axis = self.semimajor_axis - (self.flattening * self.semimajor_axis)
        self.eccentricity = np.sqrt(self.semimajor_axis**2 - self.semiminor_axis**2) / self.semimajor_axis
        self.second_flattening = (self.semimajor_axis - self.semiminor_axis) / self.semiminor_axis
        self.mean_radius = (2 * self.semimajor_axis + self.semiminor_axis) / 3


    def calc_radius(self, lla0):
        """
        -------------------------------------------------------------------
        Arguments:

        - state: array, list or nested list of initial state array
        -------------------------------------------------------------------
        Returns:

        - X_dot: state dynamics "Xdot = A*X + B*U"
        -------------------------------------------------------------------
        """


    def enu2ecef_cms(self, x_east, y_north, z_up, lat0, lon0, h0):
        """Transform position from local Cartesian (ENU) to geocentric (ECEF)"""
        a = Float(self.semimajor_axis)
        f = Float(self.flattening)
        e2 = Float(self.eccentricity**2)

        if f == 0:
            r = h0 + a
            rho = r * cosd(lat0)
            z0 = r * sind(lat0)
        else:
            sin_phi = sind(lat0)
            cos_phi = cosd(lat0)
            N = a / sqrt(1 - e2 * sin_phi**2)
            rho = (N + h0) * cos_phi
            z0 = (N * (1 - e2) + h0) * sin_phi  # type:ignore
        x0 = rho * cosd(lon0)
        y0 = rho * sind(lon0)

        t = (cosd(lat0) * z_up) - (sind(lat0) * y_north)
        dz = (sind(lat0) * z_up) + (cosd(lat0) * y_north)

        dx = (cosd(lon0) * t) - (sind(lon0) * x_east)
        dy = (sind(lon0) * t) + (cosd(lon0) * x_east)

        # Origin + offset from origin equals position in ECEF
        x = x0 + dx
        y = y0 + dy
        z = z0 + dz

        return (x, y, z)


    def ecef_to_geodetic(self,
                         ecef_x: Symbol,
                         ecef_y: Symbol,
                         ecef_z: Symbol,
                         flattening,
                         semimajor_axis):

        a = Float(self.semimajor_axis)
        b = Float(self.semiminor_axis)
        f = Float(self.flattening)
        e2 = Float(self.eccentricity**2)

        x = ecef_x
        y = ecef_y
        z = ecef_z
        rho = sqrt(x**2 + y**2)  # type:ignore
        lamb = atan2d(y, x)

        if flattening == 0:
            phi = atan2d(z, rho)
            h = sqrt(z**2 + rho**2) - a  # type:ignore
        else:
            # Spheroid properties
            ep2 = e2 / (1 - e2)  # Square of second eccentricity # type:ignore

            # Bowring's formula for initial parametric (beta) and geodetic
            # (phi) latitudes
            beta = atan2d(z, (1 - f) * rho)  # type:ignore
            sb = sind(beta)
            cb = cosd(beta)
            sb3 = sb**3
            cb3 = cb**3
            phi = atan2d((z + b * ep2 * sb3),
                         (rho - a * e2 * cb3))

            # Fixed-point iteration with Bowring's formula
            # (typically converges within two or three iterations)
            beta_new = atan2d((1 - f) * sind(phi), cosd(phi))  # type:ignore

            for _ in range(3):
                beta = beta_new
                sb = sind(beta)
                cb = cosd(beta)
                sb3 = sb**3
                cb3 = cb**3
                phi = atan2d((z + b * ep2 * sb3), (rho - a * e2 * cb3))
                beta_new = atan2d((1 - f) * sind(phi), cosd(phi))  # type:ignore

            # Ellipsoidal height from final value for latitude
            sin_phi = sind(phi)
            N = semimajor_axis / sqrt(1 - e2 * sin_phi**2)
            h = rho * cosd(phi) + (z + e2 * N * sin_phi) * sin_phi - N

        return (phi, lamb, h)


# earth = EarthModelSymbolic()
# print(earth.eccentricity**2)
# print(earth.flattening * (2 - earth.flattening))
# j2 = (2 / 3) * earth.flattening * (1 - (2 / 5) * ((earth.omega**2 * earth.semimajor_axis**3) / earth.mu))
# print(earth.J2)
# print(j2)

# print(earth.flattening)
# print(earth.inv_flattening)
# print(earth.eccentricity**2)
# print(earth.semimajor_axis)