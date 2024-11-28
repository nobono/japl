#include "atmosphere.hpp"


double Atmosphere::pressure(double alt) {
    return this->interp_pressure.interpolate(vector<dVec>({{alt}}))[0];
}

double Atmosphere::temperature(double alt) {
    return this->interp_temperature.interpolate(vector<dVec>({{alt}}))[0];
}

double Atmosphere::density(double alt) {
    return this->interp_density.interpolate(vector<dVec>({{alt}}))[0];
}

double Atmosphere::speed_of_sound(double alt) {
    return this->interp_speed_of_sound.interpolate(vector<dVec>({{alt}}))[0];
}

double Atmosphere::grav_accel(double alt) {
    return this->interp_grav_accel.interpolate(vector<dVec>({{alt}}))[0];
}

double Atmosphere::dynamics_pressure(vector<double> vel, double alt) {
    double sum = 0;
    for (int i = 0; i < vel.size(); ++i) {
        sum += pow(vel[i], 2.0);
    }
    double vel_norm = sqrt(sum);
    return pow(vel_norm, 2.0) * this->density(alt) / 2.0;
}


PYBIND11_MODULE(atmosphere, m) {
    pybind11::class_<Atmosphere>(m, "Atmosphere")
        .def(py::init<>())
        .def("pressure", &Atmosphere::pressure, "get pressure")
        .def("density", &Atmosphere::density, "get density")
        .def("temperature", &Atmosphere::temperature, "get temperature in celsius")
        .def("speed_of_sound", &Atmosphere::speed_of_sound, "get speed-of-sound")
        .def("grav_accel", &Atmosphere::grav_accel, "get gravitational acceleration")
        .def("dynamics_pressure", &Atmosphere::dynamics_pressure, "get dynamics pressure");
}
