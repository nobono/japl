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

    DataTable() = default;

    DataTable(py::array_t<double>& data, py::dict& axes) {

        vector<dVec> _axes;

        for (auto item : axes) {
            // Cast keys and values to specific types
            string key = py::cast<std::string>(item.first);
            py::array_t<double> val = py::cast<py::array_t<double>>(item.second);
            // Convert to dVec
            dVec axis_vec(val.size());
            for (int i = 0; i < val.size(); ++i) {
                axis_vec[i] = val.mutable_at(i);
            }
            // store axes
            this->axes[key] = axis_vec;

            _axes.push_back(axis_vec);
        }

        int ndim = static_cast<int>(axes.size());
        switch(ndim) {
            case 1:
                set_table<1, double>(_axes, data);
                break;
            case 2:
                set_table<2, double>(_axes, data);
                break;
            case 3:
                set_table<3, double>(_axes, data);
                break;
            case 4:
                set_table<4, double>(_axes, data);
                break;
            case 5:
                set_table<5, double>(_axes, data);
                break;
            default:
                throw std::invalid_argument("unhandled interp dimensions. table ndim:"
                                            + std::to_string(ndim));
        }
    };

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

    dVec _get_table_args(map<string, double>& kwargs) {
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
