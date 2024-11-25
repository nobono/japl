#include "aerotable.hpp"

#include <string>

namespace py = pybind11;
using std::string;



AeroTable::AeroTable(const py::kwargs& kwargs) {
    for (auto& item : kwargs) {
        // check if key exists
        string key = py::cast<string>(item.first);
        bool bkey_in_table = (table_info.find(key) != table_info.end());

        if (bkey_in_table) {
            string key = py::cast<string>(item.first);
            int table_id = this->table_info[key];
            py::array_t<double> _data = py::cast<py::array_t<double>>(item.second.attr("_data"));
            py::dict axes = py::cast<py::dict>(item.second.attr("axes"));
            auto table = DataTable(_data, axes);
            this->set_table_from_id(table, table_id);
        }
    }
}


PYBIND11_MODULE(aerotable, m) {
    pybind11::class_<AeroTable>(m, "AeroTable")
        .def(py::init<py::kwargs&>())
        // .def(py::init([](py::kwargs& kwargs) {
        //     map<string, DataTable&> map_kwargs = {};
        //     for (auto& item : kwargs) {
        //         string key = py::cast<string>(item.first);
        //         DataTable& val = py::cast<DataTable&>(item.second);
        //         map_kwargs[key] = std::move(val);
        //     }
        //     return AeroTable(map_kwargs);
        // }))
        // .def(py::init<map<string, DataTable>&>())
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
        // .def("get_CA", &AeroTable::get_CA, "")
        ;
}
