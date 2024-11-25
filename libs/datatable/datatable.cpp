#include "datatable.hpp"



template <int N, class T>
InterpMultilinear<N, T> DataTable::create_interp_N(const vector<dVec>& axes, const py::array_t<double>& data) {
    vector<dVec> f_gridList = axes;

    // create f_sizes
    vector<int> f_sizes(N);
    for (int i=0;i<f_gridList.size();i++) {
        f_sizes[i] = f_gridList[i].size();
    }

    py::array_t<double> flat = data.attr("flatten")().cast<py::array_t<double>>();

    auto begins_ends = get_begins_ends(f_gridList.begin(), f_gridList.end());
    InterpMultilinear<N, T> interp_multilinear(begins_ends.first.begin(), f_sizes,
                                           flat.data(), flat.data() + flat.size());

    // store values to make interp class pickeable
    interp_multilinear._f_gridList = f_gridList;
    interp_multilinear._f_sizes = f_sizes;
    interp_multilinear._data = data.attr("copy")();
    return interp_multilinear;
}


PYBIND11_MODULE(datatable, m) {
    pybind11::class_<DataTable>(m, "DataTable")
        .def(py::init<>())
        .def(py::init<py::array_t<double>&, py::dict&>())
        .def("__call__", [](DataTable& self, const py::kwargs& kwargs) {
            map<string, double> kw_map = convert_dict_to_map<py::kwargs, string, double>(kwargs);
            vector<dVec> args = {self._get_table_args(kw_map)};
            return self(args);
        }, "call operator.")
        .def_readonly("axes", &DataTable::axes, "")
        .def_readonly("interp", &DataTable::interp, "")
        ;
}
