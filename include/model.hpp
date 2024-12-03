#ifndef _MODEL_H_
#define _MODEL_H_

#include <vector>

#include "atmosphere.hpp"
#include "aerotable.hpp"
#include <pybind11/stl.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;


class Model {
public:
    Atmosphere atmosphere = Atmosphere();
    AeroTable aerotable = AeroTable();

    Model() = default;
    ~Model() = default;

    Model(const Model& other)
    :   atmosphere(other.atmosphere),
        aerotable(other.aerotable) {}

    Model& operator=(Model& other) {
        if (this == &other) {
            return *this; // Handle self-assignment
        }
        this->atmosphere = other.atmosphere;
        this->aerotable = other.aerotable;
        return *this;
    }

    vector<double> dynamics(double t,
                                 std::vector<double> _X_arg,
                                 std::vector<double> _U_arg,
                                 std::vector<double> _S_arg,
                                 double dt);

    vector<double> state_updates(double t,
                                      std::vector<double> _X_arg,
                                      std::vector<double> _U_arg,
                                      std::vector<double> _S_arg,
                                      double dt);

    vector<double> input_updates(double t,
                                      std::vector<double> _X_arg,
                                      std::vector<double> _U_arg,
                                      std::vector<double> _S_arg,
                                      double dt);

    void set_aerotable(const AeroTable& aerotable) {
        this->aerotable = aerotable;
    }

    void set_atmosphere(const Atmosphere& atmosphere) {
        this->atmosphere = atmosphere;
    }
};

#endif
