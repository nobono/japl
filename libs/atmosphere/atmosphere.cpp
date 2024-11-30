#include "atmosphere.hpp"


double Atmosphere::pressure(double& alt) {
    return this->interp_pressure.interpolate(vector<dVec>({{alt}}))[0];
}

double Atmosphere::pressure(const map<string, double>& kwargs) {
    for (const auto& item : kwargs) {
        string key = static_cast<string>(item.first);
        if (key == "alt") {
            double val = static_cast<double>(item.second);
            return this->pressure(val);
        }
    }
    throw std::invalid_argument("invalid argument.");
}

double Atmosphere::temperature(double& alt) {
    return this->interp_temperature.interpolate(vector<dVec>({{alt}}))[0];
}

double Atmosphere::temperature(const map<string, double>& kwargs) {
    for (const auto& item : kwargs) {
        string key = static_cast<string>(item.first);
        if (key == "alt") {
            double val = static_cast<double>(item.second);
            return this->temperature(val);
        }
    }
    throw std::invalid_argument("invalid argument.");
}

double Atmosphere::density(double& alt) {
    return this->interp_density.interpolate(vector<dVec>({{alt}}))[0];
}

double Atmosphere::density(const map<string, double>& kwargs) {
    for (const auto& item : kwargs) {
        string key = static_cast<string>(item.first);
        if (key == "alt") {
            double val = static_cast<double>(item.second);
            return this->density(val);
        }
    }
    throw std::invalid_argument("invalid argument.");
}

double Atmosphere::speed_of_sound(double& alt) {
    return this->interp_speed_of_sound.interpolate(vector<dVec>({{alt}}))[0];
}

double Atmosphere::speed_of_sound(const map<string, double>& kwargs) {
    for (const auto& item : kwargs) {
        string key = static_cast<string>(item.first);
        if (key == "alt") {
            double val = static_cast<double>(item.second);
            return this->speed_of_sound(val);
        }
    }
    throw std::invalid_argument("invalid argument.");
}

double Atmosphere::grav_accel(double& alt) {
    return this->interp_grav_accel.interpolate(vector<dVec>({{alt}}))[0];
}

double Atmosphere::grav_accel(const map<string, double>& kwargs) {
    for (const auto& item : kwargs) {
        string key = static_cast<string>(item.first);
        if (key == "alt") {
            double val = static_cast<double>(item.second);
            return this->grav_accel(val);
        }
    }
    throw std::invalid_argument("invalid argument.");
}

double Atmosphere::dynamics_pressure(vector<double>& vel, double& alt) {
    double sum = 0;
    for (int i = 0; i < vel.size(); ++i) {
        sum += pow(vel[i], 2.0);
    }
    double vel_norm = sqrt(sum);
    return pow(vel_norm, 2.0) * this->density(alt) / 2.0;
}

double Atmosphere::dynamics_pressure(const map<string, double>& kwargs) {
    double alt;
    vector<double> vel;
    bool bvalid_alt = false;
    bool bvalid_vel = false;
    for (const auto& item : kwargs) {
        string key = static_cast<string>(item.first);
        if (key == "alt") {
            alt = static_cast<double>(item.second);
            bvalid_alt = true;
        }
        else if (key == "vel") {
            vel = static_cast<vector<double>>(item.second);
            bvalid_vel = true;
        }
    }
    if (bvalid_alt && bvalid_vel) {
        return this->dynamics_pressure(vel, alt);
    }
    throw std::invalid_argument("invalid arguments.");
}


PYBIND11_MODULE(atmosphere, m) {
    pybind11::class_<Atmosphere>(m, "Atmosphere")
        .def(py::init<>())
        .def("pressure", py::overload_cast<double&>(&Atmosphere::pressure), "get pressure")
        .def("pressure", py::overload_cast<const map<string, double>&>(&Atmosphere::pressure), "get pressure")
        // .def("pressure", [](Atmosphere& self, const map<string, double>& kwargs) -> double {
        //     for (const auto& item : kwargs) {
        //         string key = static_cast<string>(item.first);
        //         if (key == "alt") {
        //             double val = static_cast<double>(item.second);
        //             return self.pressure(val);
        //         }
        //     }
        //     throw std::invalid_argument("invalid argument.");
        // })

        .def("density", py::overload_cast<double&>(&Atmosphere::density), "get density")
        .def("density", py::overload_cast<const map<string, double>&>(&Atmosphere::density), "get density")
        // .def("density", [](Atmosphere& self, const map<string, double>& kwargs) -> double {
        //     for (const auto& item : kwargs) {
        //         string key = static_cast<string>(item.first);
        //         if (key == "alt") {
        //             double val = static_cast<double>(item.second);
        //             return self.density(val);
        //         }
        //     }
        //     throw std::invalid_argument("invalid argument.");
        // })

        .def("temperature", py::overload_cast<double&>(&Atmosphere::temperature), "get temperature in celsius")
        .def("temperature", py::overload_cast<const map<string, double>&>(&Atmosphere::temperature), "get temperature in celsius")
        // .def("temperature", [](Atmosphere& self, const map<string, double>& kwargs) -> double {
        //     for (const auto& item : kwargs) {
        //         string key = static_cast<string>(item.first);
        //         if (key == "alt") {
        //             double val = static_cast<double>(item.second);
        //             return self.temperature(val);
        //         }
        //     }
        //     throw std::invalid_argument("invalid argument.");
        // })

        .def("speed_of_sound", py::overload_cast<double&>(&Atmosphere::speed_of_sound), "get speed-of-sound")
        .def("speed_of_sound", py::overload_cast<const map<string, double>&>(&Atmosphere::speed_of_sound), "get speed-of-sound")
        // .def("speed-of-sound", [](Atmosphere& self, const map<string, double>& kwargs) -> double {
        //     for (const auto& item : kwargs) {
        //         string key = static_cast<string>(item.first);
        //         if (key == "alt") {
        //             double val = static_cast<double>(item.second);
        //             return self.speed_of_sound(val);
        //         }
        //     }
        //     throw std::invalid_argument("invalid argument.");
        // })

        .def("grav_accel", py::overload_cast<double&>(&Atmosphere::grav_accel), "get gravitational acceleration")
        .def("grav_accel", py::overload_cast<const map<string, double>&>(&Atmosphere::grav_accel), "get gravitational acceleration")
        // .def("grav_accel", [](Atmosphere& self, const map<string, double>& kwargs) -> double {
        //     for (const auto& item : kwargs) {
        //         string key = static_cast<string>(item.first);
        //         if (key == "alt") {
        //             double val = static_cast<double>(item.second);
        //             return self.grav_accel(val);
        //         }
        //     }
        //     throw std::invalid_argument("invalid argument.");
        // })

        .def("dynamics_pressure", py::overload_cast<vector<double>&, double&>(&Atmosphere::dynamics_pressure), "get dynamics pressure")
        .def("dynamics_pressure", py::overload_cast<const map<string, double>&>(&Atmosphere::dynamics_pressure), "get dynamics pressure")
        // .def("dynamics_pressure", [](Atmosphere& self, const map<string, double>& kwargs) -> double {
        //     double alt;
        //     vector<double> vel;
        //     bool bvalid_alt = false;
        //     bool bvalid_vel = false;
        //     for (const auto& item : kwargs) {
        //         string key = static_cast<string>(item.first);
        //         if (key == "alt") {
        //             alt = static_cast<double>(item.second);
        //             bvalid_alt = true;
        //         }
        //         else if (key == "vel") {
        //             vel = static_cast<vector<double>>(item.second);
        //             bvalid_vel = true;
        //         }
        //     }
        //     if (bvalid_alt && bvalid_vel) {
        //         return self.dynamics_pressure(vel, alt);
        //     }
        //     throw std::invalid_argument("invalid arguments.");
        // })
    ;
}
