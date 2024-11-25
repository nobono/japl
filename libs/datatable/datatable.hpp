#ifndef _DATATABLE_H_
#define _DATATABLE_H_

#include <map>
#include <variant>
#include <string>

#include <pybind11/stl.h>
#include "../linterp/src/linterp.h"
#include "../boost/multi_array.hpp"



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
    py::array_t<double> _data;

    DataTable() = default;

    DataTable(py::array_t<double>& data, py::dict& axes);

    DataTable(py::array_t<double>& data, vector<dVec>& axes);

    // // copy constructor
    // DataTable(const DataTable& other) {
    //     _data = other._data;
    //     interp = std::move(other.interp);
    //     axes = other.axes;
    // }

    // DataTable& operator=(const DataTable& other) {
    //     if (this != &other) { // Prevent self-assignment
    //         // Copy all members from `other` to `this`
    //         this->_data = other._data;
    //         this->axes = other.axes;
    //         this->interp = other.interp;

    //     }
    //     return *this;
    // }

    // // Copy constructor
    // DataTable(const DataTable& other) {
    //     this->interp = std::visit(
    //         [](auto&& arg) -> interp_table_t {
    //             using T = std::decay_t<decltype(arg)>;
    //             return T(arg); // Use copy constructor of the active type
    //         },
    //         other.interp
    //     );
    // }

    // // Copy assignment operator
    // DataTable& operator=(const DataTable& other) {
    //     if (this != &other) {
    //         interp = std::visit(
    //             [](auto&& arg) -> interp_table_t {
    //                 using T = std::decay_t<decltype(arg)>;
    //                 return T(arg); // Use copy constructor of the active type
    //             },
    //             other.interp
    //         );
    //     }
    //     return *this;
    // }

    // Call interpolation table
    vector<double> operator()(const vector<dVec>& points) {
        interp_table_t* table_ptr = &this->interp;
        switch(this->interp.index()) {
            case 0: return get_if<InterpMultilinear<1, double>>(table_ptr)->interpolate(points);
            case 1: return get_if<InterpMultilinear<2, double>>(table_ptr)->interpolate(points);
            case 2: return get_if<InterpMultilinear<3, double>>(table_ptr)->interpolate(points);
            case 3: return get_if<InterpMultilinear<4, double>>(table_ptr)->interpolate(points);
            case 4: return get_if<InterpMultilinear<5, double>>(table_ptr)->interpolate(points);
            default: throw std::invalid_argument("unhandled case.");
        }
    }

    // Call interpolation table (python overload)
    py::array_t<double> operator()(const py::array_t<double>& points) {
        interp_table_t* table_ptr = &this->interp;
        switch(this->interp.index()) {
            case 0: return get_if<InterpMultilinear<1, double>>(table_ptr)->interpolate(points);
            case 1: return get_if<InterpMultilinear<2, double>>(table_ptr)->interpolate(points);
            case 2: return get_if<InterpMultilinear<3, double>>(table_ptr)->interpolate(points);
            case 3: return get_if<InterpMultilinear<4, double>>(table_ptr)->interpolate(points);
            case 4: return get_if<InterpMultilinear<5, double>>(table_ptr)->interpolate(points);
            default: throw std::invalid_argument("unhandled case.");
        }
    }

    // Call interpolation table (python keywords overload)
    vector<double> operator()(const map<string, double>& kwargs) {
        interp_table_t* table_ptr = &this->interp;
        vector<dVec> points = {this->_get_table_args(kwargs)};
        switch(this->interp.index()) {
            case 0: return get_if<InterpMultilinear<1, double>>(table_ptr)->interpolate(points);
            case 1: return get_if<InterpMultilinear<2, double>>(table_ptr)->interpolate(points);
            case 2: return get_if<InterpMultilinear<3, double>>(table_ptr)->interpolate(points);
            case 3: return get_if<InterpMultilinear<4, double>>(table_ptr)->interpolate(points);
            case 4: return get_if<InterpMultilinear<5, double>>(table_ptr)->interpolate(points);
            default: throw std::invalid_argument("unhandled case.");
        }
    }

    dVec _get_table_args(const map<string, double>& kwargs) {
        dVec args;

        for (const auto& item : kwargs) {
            string label = item.first;
            double val = static_cast<double>(item.second);
            if (this->axes.count(label) > 0) {
                args.push_back(val);
            }
        }

        if (args.size() != this->axes.size()) {
            throw std::invalid_argument("not enough arguments provided.");
        }

        return args;
    }

private:
    // Creates NDInterpolator object from 2 vectors
    template <int N, class T>
    InterpMultilinear<N, T> create_interp_N(const vector<dVec>& axes, const py::array_t<double>& data);

    template <int N, class T>
    void set_table(const vector<dVec>& axes, const py::array_t<double>& data) {
        InterpMultilinear<N, T> _interp = create_interp_N<N, T>(axes, data);
        interp = std::move(_interp);
    }
};


#endif
