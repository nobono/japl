#ifndef _AEROTABLE_H_
#define _AEROTABLE_H_

#include <variant>
#include <map>

#include "../datatable/datatable.hpp"
#include <pybind11/stl.h>
#include "../linterp/src/linterp.h"
#include "../boost/multi_array.hpp"
#include "../datatable/datatable.hpp"



// using std::map;
// using std::variant;
// using std::string;
// using std::get_if;
// namespace py = pybind11;

// typedef variant<
//     InterpMultilinear<1,double>,
//     InterpMultilinear<2,double>,
//     InterpMultilinear<3,double>,
//     InterpMultilinear<4,double>,
//     InterpMultilinear<5,double>
//     >
// table_t;

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

    DataTable CA;
    DataTable CA_Boost;
    DataTable CA_Coast;
    DataTable CNB;
    DataTable CYB;
    DataTable CA_Boost_alpha;
    DataTable CA_Coast_alpha;
    DataTable CNB_alpha;
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


    // // vector<vector<double>> args = {{1., 1.}};
    // // py::print(args, args.size(), CA.index());
    // // auto ret = get_if<InterpMultilinear<2, double>>(&CA);
    // // return ret->interpolate(args);
    // py::array_t<double> get_CA(py::kwargs kwargs) {
    //     // interp_table_t* table_ptr = &CA;
    //     // vector<vector<double>> args = _kwargs_to_args(kwargs);
    //     map<string, double> map_kwargs = convert_dictlike_to_map<py::kwargs, string, double>(kwargs);
    //     // switch(CA.index()) {
    //     //     case 0: return get_if<InterpMultilinear<1, double>>(table_ptr)->interpolate(args);
    //     //     case 1: return get_if<InterpMultilinear<2, double>>(table_ptr)->interpolate(args);
    //     //     case 2: return get_if<InterpMultilinear<3, double>>(table_ptr)->interpolate(args);
    //     //     case 3: return get_if<InterpMultilinear<4, double>>(table_ptr)->interpolate(args);
    //     //     case 4: return get_if<InterpMultilinear<5, double>>(table_ptr)->interpolate(args);
    //     //     default: throw std::invalid_argument("unhandled case.");
    //     // }
    //     // dVec ret = get_if<InterpMultilinear<2, double>>(table_ptr)->interpolate(args);
    //     // vector<dVec> args = {self._get_table_args(kw_map)};
    //     dVec ret = this->CA(map_kwargs);
    //     return convert_vec_to_numpy<double>(ret);
    // }

    vector<vector<double>> _kwargs_to_args(py::kwargs kwargs) {
        vector<vector<double>> args;
        vector<double> point;
        for (auto& item : kwargs) {
            point.push_back(item.second.cast<double>());
        }
        args.push_back(point);
        return args;
    }

    void set_table_from_id(DataTable& table, int& id) {
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
};

#endif
