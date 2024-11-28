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
            // if (std::is_same_v<>) {}
            string key = py::cast<string>(item.first);
            int id = this->table_info[key];

            if (py::isinstance<py::float_>(item.second)) {
                double attr = py::cast<double>(item.second);
                this->set_attr_from_id(attr, id);
            } else if (py::isinstance<py::int_>(item.second)) {
                double attr = py::cast<double>(item.second);
                this->set_attr_from_id(attr, id);
            } else if (py::isinstance<DataTable>(item.second)) {
                DataTable table = py::cast<DataTable>(item.second);
                this->set_table_from_id(table, id);
            } else {
                throw std::invalid_argument("argument type not accepted. "
                                            "Accepted types are [int, double, DataTable]");
            }
        }
    }
}


PYBIND11_MODULE(aerotable, m) {
    pybind11::class_<AeroTable>(m, "AeroTable")
        .def(py::init<py::kwargs&>())
        // .def_readonly("Sref", &AeroTable::Sref)
        // .def_readonly("Lref", &AeroTable::Lref)
        // .def_readonly("MRC", &AeroTable::MRC)
        // .def_readwrite("CA", &AeroTable::CA)
        // .def_readonly("CA_Boost", &AeroTable::CA_Boost)
        // .def_readonly("CA_Coast", &AeroTable::CA_Coast)
        // .def_readwrite("CNB", &AeroTable::CNB)
        // .def_readonly("CYB", &AeroTable::CYB)
        // .def_readonly("CA_Boost_alpha", &AeroTable::CA_Boost_alpha)
        // .def_readonly("CA_Coast_alpha", &AeroTable::CA_Coast_alpha)
        // .def_readonly("CNB_alpha", &AeroTable::CNB_alpha)

        .def_readonly("table_info", &AeroTable::table_info, "tables and their axes dimensions")

        .def_property("Sref",
                      [](const AeroTable& self) {return self.Sref;},  // getter
                      [](AeroTable& self, const decltype(AeroTable::Sref)& value) {self.Sref = value;})  // setter
        .def_property("Lref",
                      [](const AeroTable& self) {return self.Lref;},  // getter
                      [](AeroTable& self, const decltype(AeroTable::Lref)& value) {self.Lref = value;})  // setter
        .def_property("MRC",
                      [](const AeroTable& self) {return self.MRC;},  // getter
                      [](AeroTable& self, const decltype(AeroTable::MRC)& value) {self.MRC = value;})  // setter
        .def_property("CA",
                      [](const AeroTable& self) -> const DataTable& {return self.CA;},  // getter
                      [](AeroTable& self, const decltype(AeroTable::CA)& value) {self.CA = value;})  // setter
        .def_property("CA_Boost",
                      [](const AeroTable& self) -> const DataTable& {return self.CA_Boost;},  // getter
                      [](AeroTable& self, const decltype(AeroTable::CA_Boost)& value) {self.CA_Boost = value;})  // setter
        .def_property("CA_Coast",
                      [](const AeroTable& self) -> const DataTable& {return self.CA_Coast;},  // getter
                      [](AeroTable& self, const decltype(AeroTable::CA_Coast)& value) {self.CA_Coast = value;})  // setter
        .def_property("CNB",
                      [](const AeroTable& self) -> const DataTable& {return self.CNB;},  // getter
                      [](AeroTable& self, const decltype(AeroTable::CNB)& value) {self.CNB = value;})  // setter
        .def_property("CYB",
                      [](const AeroTable& self) -> const DataTable& {return self.CYB;},  // getter
                      [](AeroTable& self, const decltype(AeroTable::CYB)& value) {self.CYB = value;})  // setter
        .def_property("CA_Boost_alpha",
                      [](const AeroTable& self) -> const DataTable& {return self.CA_Boost_alpha;},  // getter
                      [](AeroTable& self, const decltype(AeroTable::CA_Boost_alpha)& value) {self.CA_Boost_alpha = value;})  // setter
        .def_property("CA_Coast_alpha",
                      [](const AeroTable& self) -> const DataTable& {return self.CA_Coast_alpha;},  // getter
                      [](AeroTable& self, const decltype(AeroTable::CA_Coast_alpha)& value) {self.CA_Coast_alpha = value;})  // setter
        .def_property("CNB_alpha",
                      [](const AeroTable& self) -> const DataTable& {return self.CNB_alpha;},  // getter
                      [](AeroTable& self, const decltype(AeroTable::CNB_alpha)& value) {self.CNB_alpha = value;})  // setter
        ;
}
