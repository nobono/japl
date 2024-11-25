#ifndef _AEROTABLE_H_
#define _AEROTABLE_H_

#include <iostream>
#include <variant>
#include <map>
#include <memory>

#include <pybind11/stl.h>
#include "../linterp/src/linterp.h"
#include "../boost/multi_array.hpp"

using std::map;
using std::variant;
using std::string;
using std::get_if;
namespace py = pybind11;

typedef variant<
    InterpMultilinear<1,double>,
    InterpMultilinear<2,double>,
    InterpMultilinear<3,double>,
    InterpMultilinear<4,double>,
    InterpMultilinear<5,double>
    >
table_t;

struct AeroTableArgs {
    double alpha = 0;
    double phi = 0;
    double mach = 0;
    double alt = 0;
    double iota = 0;
};


class AeroTable {

public:

    map<string, int> table_info{
    {"CA", 1},
    {"CA_Boost", 2},
    {"CA_Coast", 3},
    {"CNB", 4},
    {"CYB", 5},
    {"CA_Boost_alpha", 6},
    {"CA_Coast_alpha", 7},
    {"CNB_alpha", 8},
    };

    // table_t CA;
    // table_t CA_boost;
    // table_t CA_coast;
    // table_t CNB;
    // table_t CYB;

    table_t CA;
    table_t CA_Boost;
    table_t CA_Coast;
    table_t CNB;
    table_t CYB;
    table_t CA_Boost_alpha;
    table_t CA_Coast_alpha;
    table_t CNB_alpha;
    double Sref;
    double Lref;
    double MRC;

    AeroTable() = default;

    AeroTable(const py::kwargs& kwargs);

    inline vector<string> get_keys() {
        vector<string> keys;
        for (const auto& pair : table_info) {
            keys.push_back(pair.first);
        }
        return keys;
    }

    // double get_Sref(void) {return Sref;};
    // double get_Lref(void) {return Lref;};
    // double get_CA(map<string, double> args);
    // double get_CA_Boost(double alpha, double phi, double mach, double alt, double iota);
    // double get_CA_Coast(double alpha, double phi, double mach, double alt, double iota);
    // double get_CNB(double alpha, double phi, double mach, double alt, double iota);
    // double get_CYB(double alpha, double phi, double mach, double alt, double iota);

    // Creates 1D NDInterpolator object from 2 vectors
    template <int N, class T>
    InterpMultilinear<N, T> create_interp_N(pybind11::tuple& axes, pybind11::array_t<double>& data);

    // template <int N>
    // boost::multi_array<double, N> _numpy_array_to_multi(py::array_t<double> table) {
    //     // request buffer
    //     py::buffer_info buf = table.request();

    //     // create boost multi array
    //     boost::multi_array<double, N> array;

    //     // get shape of py::array_t with same dims
    //     typename boost::multi_array<double, N>::extent_gen extents;
    //     for (size_t i =0; i < N; ++i) {
    //         extents.ranges_[i] = buf.shape[i];
    //     }
    //     array.resize(extents);

    //     // copy data
    //     const double* table_ptr = static_cast<double*>(buf.ptr);
    //     std::copy(table_ptr, table_ptr + buf.size, array.data());

    //     return array;
    // }

    // template<class T>
    // void _set_table_member(T AeroTable::*member, const T& val) {
    //     
    // }

    // void set_Sref(double val) {
    //     Sref = val;
    // }

    // void set_Lref(double val) {
    //     Lref = val;
    // }

    // void set_MRC(double val) {
    //     MRC = val;
    // }

    // void set_CA(py::object table) {
    //     
    // }

    // void set_CA_Boost(py::object table) {

    // }

    // void set_CA_Coast(py::object table) {

    // }

    // void set_CNB(py::object table) {

    // }

    // void set_CYB(py::object table) {

    // }

    // void set_CA_Boost_alpha(py::object table) {

    // }

    // void set_CA_Coast_alpha(py::object table) {

    // }

    // void set_CNB_alpha(py::object table) {

    // }
    
    // vector<vector<double>> args = {{1., 1.}};
    // py::print(args, args.size(), CA.index());
    // auto ret = get_if<InterpMultilinear<2, double>>(&CA);
    // return ret->interpolate(args);
    py::array_t<double> get_CA(py::kwargs kwargs) {
        vector<vector<double>> args = _kwargs_to_args(kwargs);
        table_t* table_ptr = &CA;
        switch(CA.index()) {
            case 0: return get_if<InterpMultilinear<1, double>>(table_ptr)->interpolate(args);
            case 1: return get_if<InterpMultilinear<2, double>>(table_ptr)->interpolate(args);
            case 2: return get_if<InterpMultilinear<3, double>>(table_ptr)->interpolate(args);
            case 3: return get_if<InterpMultilinear<4, double>>(table_ptr)->interpolate(args);
            case 4: return get_if<InterpMultilinear<5, double>>(table_ptr)->interpolate(args);
            default: throw std::invalid_argument("unhandled case.");
        }
    }

    // py::array_t<double> _call_table_interp() 

    vector<vector<double>> _kwargs_to_args(py::kwargs kwargs) {
        vector<vector<double>> args;
        vector<double> point;
        for (auto& item : kwargs) {
            point.push_back(item.second.cast<double>());
        }
        args.push_back(point);
        return args;
    }

    // vector<double> _get_table_args()
        // for (const auto& [key, value] : args) {
        //     std::cout << key << " " << value << "\n";
        // }

    // -----------------------------------------------------------
    // For c++ native DataTables
    // -----------------------------------------------------------
    template <int N, class T>
    void set_table_from_id(string name, pybind11::tuple& axes, pybind11::array_t<double>& data) {
        InterpMultilinear<N, T> table = create_interp_N<N, T>(axes, data);
        int id = table_info[name];
        set_table<N, T>(table, id);
    }

    template <int N, class T>
    void set_table(InterpMultilinear<N, T>& table, int& id) {
        switch (id) {
            case 1:
                CA = std::move(table);
                break;
            case 2:
                CA_Boost = std::move(table);
                break;
            case 3:
                CA_Coast = std::move(table);
                break;
            case 4:
                CNB = std::move(table);
                break;
            case 5:
                CYB = std::move(table);
                break;
            default:
                throw std::invalid_argument("unhandled case.");
        }
    }
    // -----------------------------------------------------------

    // void set_table_from_id(string name, py::object& table) {
    //     int id = table_info[name];
    //     set_table(table, id);
    // }

    // void set_table(py::object& table, int& id) {
    //     switch (id) {
    //         case 1:
    //             CA = std::move(table);
    //             break;
    //         case 2:
    //             CA_Boost = std::move(table);
    //             break;
    //         case 3:
    //             CA_Coast = std::move(table);
    //             break;
    //         case 4:
    //             CNB = std::move(table);
    //             break;
    //         case 5:
    //             CYB = std::move(table);
    //             break;
    //         case 6:
    //             CA_Boost_alpha = std::move(table);
    //             break;
    //         case 7:
    //             CA_Coast_alpha = std::move(table);
    //             break;
    //         case 8:
    //             CNB_alpha = std::move(table);
    //             break;
    //         default:
    //             throw std::invalid_argument("unhandled case.");
    //     }
    // }
};

#endif
