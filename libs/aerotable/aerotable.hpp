#ifndef _AEROTABLE_H_
#define _AEROTABLE_H_

#include <variant>
#include <map>

#include <pybind11/stl.h>
#include "../linterp/src/linterp.h"
#include "../boost/multi_array.hpp"

using std::map;
using std::variant;
using std::string;

typedef variant<
    InterpMultilinear<1,double>,
    InterpMultilinear<2,double>,
    InterpMultilinear<3,double>,
    InterpMultilinear<4,double>,
    InterpMultilinear<5,double>,
    py::object>
table_t;


class AeroTable {

public:

    map<string, int> table_info{
    {"CA", 1},
    {"CNB", 2},
    };

    table_t CA;
    table_t CNB;

    AeroTable(const py::kwargs& kwargs);

    inline vector<string> get_keys() {
        vector<string> keys;
        for (const auto& pair : table_info) {
            keys.push_back(pair.first);
        }
        return keys;
    }

private:
    // Creates 1D NDInterpolator object from 2 vectors
    template <int N, class T>
    InterpMultilinear<N, T> create_interp_N(pybind11::tuple& axes, pybind11::array_t<double>& data);

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
                CNB = std::move(table);
                break;
            default:
                throw std::invalid_argument("unhandled case.");
        }
    }
};

#endif
