#ifndef _DATATABLE_H_
#define _DATATABLE_H_

#include <map>
#include <variant>
#include <string>

#include <pybind11/stl.h>
#include "linterp/linterp.h"
#include "boost/multi_array.hpp"



using std::map;
using std::string;
using std::variant;
using std::get_if;
namespace py = pybind11;

typedef variant<
    InterpMultilinear<1,double>,
    InterpMultilinear<2,double>,
    InterpMultilinear<3,double>,
    InterpMultilinear<4,double>,
    InterpMultilinear<5,double>
    >
interp_table_t;


class DataTable {

public:
    map<string, dVec> axes = {};
    interp_table_t interp;

    DataTable() {
        // Create empty Interp object
        vector<dVec> axes = {{}};
        py::array_t<double> data;
        this->interp = create_interp_N<1, double>(axes, data);
        this->axes = {{"null", vector<double>({})}};
    };

    DataTable(py::array_t<double>& data, py::dict& axes);

    DataTable(py::array_t<double>& data, vector<dVec>& axes);

    // copy constructor
    DataTable(const DataTable& other)
    :   axes(other.axes),
        interp(other.interp) {}

    // copy assignment operator
    DataTable& operator=(const DataTable& other) {
        if (this == &other) {
            return *this; // Handle self-assignment
        }
        axes = other.axes;
        interp = other.interp;
        return *this;
    }

    // Call interpolation table
    vector<double> operator()(const vector<dVec>& points);

    // Call interpolation table (python overload)
    py::array_t<double> operator()(const py::array_t<double>& points);

    // Call interpolation table (python keywords overload)
    vector<double> operator()(const map<string, double>& kwargs);

    // Handle passing kwargs -> args
    dVec _get_table_args(const map<string, double>& kwargs);

    // Creates NDInterpolator object from 2 vectors
    template <int N, class T>
    InterpMultilinear<N, T> create_interp_N(const vector<dVec> axes, const py::array_t<double> data);
};


#endif
