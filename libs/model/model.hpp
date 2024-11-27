#ifndef _MODEL_H_
#define _MODEL_H_

#include <vector>

#include "../atmosphere/atmosphere.hpp"
#include "../aerotable/aerotable.hpp"
#include <pybind11/stl.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;


class Model {
public:
    Atmosphere atmosphere = Atmosphere();
    AeroTable aerotable = AeroTable();
    // py::object atmosphere = py::object();
    // py::object aerotable = py::object();

    Model() = default;

    py::array_t<double> dynamics(double t,
                                 std::vector<double> _X_arg,
                                 std::vector<double> _U_arg,
                                 std::vector<double> _S_arg,
                                 double dt);

    py::array_t<double> state_updates(double t,
                                      std::vector<double> _X_arg,
                                      std::vector<double> _U_arg,
                                      std::vector<double> _S_arg,
                                      double dt);

    py::array_t<double> input_updates(double t,
                                      std::vector<double> _X_arg,
                                      std::vector<double> _U_arg,
                                      std::vector<double> _S_arg,
                                      double dt);

    void set_aerotable(AeroTable& aerotable) {
        this->aerotable = aerotable;
    }

    void set_atmosphere(Atmosphere& atmosphere) {
        this->atmosphere = atmosphere;
    }
};

#endif
