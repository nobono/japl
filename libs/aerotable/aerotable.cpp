#include "aerotable.hpp"

#include <string>

namespace py = pybind11;
using std::string;


template <int N, class T>
InterpMultilinear<N, T> AeroTable::create_interp_N(pybind11::tuple& axes, pybind11::array_t<double>& data) {
    // store axes in gridList
    vector<dVec> f_gridList = process_tuple_of_arrays(axes);

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

AeroTable::AeroTable(const py::kwargs& kwargs)
    // defaults
    // CA(py::none())
{
    for (auto key : get_keys()) {
        // check if key exists
        py::object item = kwargs.attr("get")(key, py::none());

        bool bkey_in_table = (table_info.find(key) != table_info.end());
        bool bitem_valid = !item.is_none();

        if (bitem_valid && bkey_in_table) {
            // -----------------------------------------------------------
            // For c++ native DataTables
            // -----------------------------------------------------------
            // if DataTable is passed, get LinearInterp member
            if (py::hasattr(item, "interp")) {
                item = item.attr("interp").attr("interp_obj");
            }
            py::tuple axes = item.attr("_f_gridList").cast<py::tuple>();
            int ndim = static_cast<int>(axes.size());
            py::array_t<double> data = item.attr("_data").cast<py::array_t<double>>();

            switch(ndim) {
                case 1:
                    set_table_from_id<1, double>(key, axes, data);
                    break;
                case 2:
                    set_table_from_id<2, double>(key, axes, data);
                    break;
                case 3:
                    set_table_from_id<3, double>(key, axes, data);
                    break;
                case 4:
                    set_table_from_id<4, double>(key, axes, data);
                    break;
                case 5:
                    set_table_from_id<5, double>(key, axes, data);
                    break;
                default:
                    throw std::invalid_argument("unhandled interp dimensions. table ndim:"
                                                + std::to_string(ndim));
            }
            // -----------------------------------------------------------

            // set_table_from_id(key, item);
        }
    }
}


PYBIND11_MODULE(aerotable, m) {
    pybind11::class_<AeroTable>(m, "AeroTable")
        .def(py::init<py::kwargs>())
        .def_readonly("Sref", &AeroTable::Sref, "")
        .def_readonly("Lref", &AeroTable::Lref, "")
        .def_readonly("CA", &AeroTable::CA, "")
        .def_readonly("CA_Boost", &AeroTable::CA_Boost, "")
        .def_readonly("CA_Coast", &AeroTable::CA_Coast, "")
        .def_readonly("CNB", &AeroTable::CNB, "")
        .def_readonly("CYB", &AeroTable::CYB, "")
        .def_readonly("CA_Boost_alpha", &AeroTable::CA_Boost_alpha, "")
        .def_readonly("CA_Coast_alpha", &AeroTable::CA_Coast_alpha, "")
        .def_readonly("CNB_alpha", &AeroTable::CA_Coast_alpha, "")
        .def_readwrite("table_info", &AeroTable::table_info, "tables and their axes dimensions")

        .def("get_CA", &AeroTable::get_CA, "")
        ;
}
