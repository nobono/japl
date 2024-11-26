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
    interp_table_t interp2;
    // py::array_t<double> _data;

    DataTable() = default;

    DataTable(py::array_t<double>& data, py::dict& axes);

    DataTable(py::array_t<double>& data, vector<dVec>& axes);

    // copy constructor
    // DataTable(const DataTable& other)
    // :   axes(other.axes),
    //     interp(other.interp),
    //     _data(other._data) {}

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
    vector<double> operator()(const vector<dVec>& points);

    // Call interpolation table (python overload)
    py::array_t<double> operator()(const py::array_t<double>& points);

    // Call interpolation table (python keywords overload)
    vector<double> operator()(const map<string, double>& kwargs);

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

    void cc_test(void) {
        InterpMultilinear<2, double>* table = std::get_if<InterpMultilinear<2, double>>(&this->interp);
        // InterpMultilinear<2, double>* thing = table;
        //
        py::print("in cc_test()");
        // py::print("_data", table->_data);
        // py::print("m_pF.size:", table->m_pF->size());
        // py::print("m_grid_list", table->m_grid_list.size());

        // for (int i=0;i<3;++i) {
        //     for (int j=0;j<3;++j) {
        //         array<int, 2> index({i, j});
        //         // py::print((table->m_F_copy[i + j]));
        //         py::print(((*table->m_pF)(index)));
        //     }
        //     py::print();
        // }

        // for (auto i : table->m_grid_list) {
        //     for (auto j : i) {
        //         std::cout << j << " ";
        //     }
        //     std::cout << "\n";
        // }

        vector<dVec> points = {{1., 1.}};
        py::print("call()", table->interpolate(points));
        py::print();
    }

private:
    // Creates NDInterpolator object from 2 vectors
    template <int N, class T>
    InterpMultilinear<N, T> create_interp_N(const vector<dVec> axes, const py::array_t<double> data);

    template <int N, class T>
    void set_table(const vector<dVec> axes, const py::array_t<double> data) {
        // InterpMultilinear<N, T> _interp = create_interp_N<N, T>(axes, data);
        // interp = std::move(_interp);
        InterpMultilinear<N, T> thing = create_interp_N<N, T>(axes, data);
        // InterpMultilinear<N, T> thing = create_interp_N<N, T>(axes, data);
        // InterpMultilinear<N, T>* table = std::get_if<InterpMultilinear<N, T>>(&this->interp);
        // vector<dVec> points = {{1., 1.}};
        // py::print("thing:", thing.interpolate(points));
        // InterpMultilinear<N, T>* table = std::get_if<InterpMultilinear<N, T>>(&thing);
        // py::print("table:", thing.interpolate(points));
        // this->interp = _interp;
        // py::print(_interp._data);
        // py::print(_interp._f_gridList);
        // py::print(_interp.m_grid_ref_list);
    }
};


#endif
