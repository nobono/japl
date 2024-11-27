#ifndef _AEROTABLE_H_
#define _AEROTABLE_H_

#include <variant>
#include <map>

#include "../datatable/datatable.hpp"
#include <pybind11/stl.h>
#include "../linterp/src/linterp.h"
#include "../boost/multi_array.hpp"
#include "../datatable/datatable.hpp"



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
                CA = table;
                break;
            case 2:
                CA_Boost = table;
                break;
            case 3:
                CA_Coast = table;
                break;
            case 4:
                CNB = table;
                break;
            case 5:
                CYB = table;
                break;
            case 6:
                CA_Boost_alpha = table;
                break;
            case 7:
                CA_Coast_alpha = table;
                break;
            case 8:
                CNB_alpha = table;
                break;
            default:
                throw std::invalid_argument("unhandled case.");
        }
    }
};

#endif
