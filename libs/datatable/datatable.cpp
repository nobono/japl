#include "datatable.hpp"




DataTable::DataTable(py::array_t<double>& data, py::dict& axes) {

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

    this->_data = data;

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

DataTable::DataTable(py::array_t<double>& data, vector<dVec>& axes) {
    this->_data = data;
    int ndim = static_cast<int>(axes.size());
    switch(ndim) {
        case 1:
            set_table<1, double>(axes, data);
            break;
        case 2:
            set_table<2, double>(axes, data);
            break;
        case 3:
            set_table<3, double>(axes, data);
            break;
        case 4:
            set_table<4, double>(axes, data);
            break;
        case 5:
            set_table<5, double>(axes, data);
            break;
        default:
            throw std::invalid_argument("unhandled interp dimensions. table ndim:"
                                        + std::to_string(ndim));
    }
};


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
        .def(py::init<py::array_t<double>&, vector<dVec>&>())
        .def("__call__", [](DataTable& self, const py::kwargs& kwargs) {
            map<string, double> kw_map = convert_dictlike_to_map<py::kwargs, string, double>(kwargs);
            vector<dVec> args = {self._get_table_args(kw_map)};
            return self(args);
        }, py::return_value_policy::reference, "call operator.")
        .def_readwrite("_data", &DataTable::_data, "")
        .def_readwrite("axes", &DataTable::axes, "")
        // .def_readonly("interp", &DataTable::interp, "")
        ;
}
