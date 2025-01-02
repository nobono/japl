#include "../include/masstable.hpp"

#include <string>
#include <algorithm>

namespace py = pybind11;
using std::string;



MassTable::MassTable(const py::kwargs& kwargs) {

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
            } else if (py::isinstance<py::array_t<double>>(item.second)) {
                vector<double> attr = py::cast<vector<double>>(item.second);
                this->set_array_from_id(attr, id);
            } else if (py::isinstance<py::array_t<int>>(item.second)) {
                vector<double> attr = py::cast<vector<double>>(item.second);
                this->set_array_from_id(attr, id);
            } else if (py::isinstance<py::list>(item.second)) {
                vector<double> attr = py::cast<vector<double>>(item.second);
                this->set_array_from_id(attr, id);
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


PYBIND11_MODULE(masstable, m) {
    pybind11::class_<MassTable>(m, "MassTable")
        .def(py::init<py::kwargs&>())

        // ------------------------------------------------------------------------------------------------
        // class methods
        // ------------------------------------------------------------------------------------------------
        .def("get_wet_mass", &MassTable::get_wet_mass)
        .def("get_dry_mass", &MassTable::get_dry_mass)
        .def("get_mass_dot", &MassTable::get_mass_dot)
        .def("get_cg", &MassTable::get_cg)
        .def("get_isp", &MassTable::get_isp)
        .def("get_raw_thrust", &MassTable::get_raw_thrust)
        .def("get_thrust", &MassTable::get_thrust)
        .def_readonly("table_info", &MassTable::table_info, "tables and their axes dimensions")

        // ------------------------------------------------------------------------------------------------
        // Staged class methods
        // ------------------------------------------------------------------------------------------------
        // .def("get_increments", &MassTable::get_increments, "returns increments of the current stage")
        .def("add_stage", &MassTable::add_stage)
        .def("set_stage", &MassTable::set_stage)
        .def("get_stage", &MassTable::get_stage)
        .def_readonly("stages", &MassTable::stages)
        .def_readonly("stage_id", &MassTable::stage_id, "id of current stage")
        .def_readonly("is_stage", &MassTable::is_stage, "true if class is stage instance; false if container for stages")

        .def_readonly("nozzle_area", &MassTable::nozzle_area)

        // ------------------------------------------------------------------------------------------------
        // Setters & Getters
        // ------------------------------------------------------------------------------------------------
        // .def_property("nozzle_area",
        //               [](const MassTable& self) -> const double& {return self.nozzle_area;},  // getter
        //               [](MassTable& self, const decltype(MassTable::nozzle_area)& value) {self.nozzle_area = value;})  // setter
        .def_property("mass_dot",
                      [](const MassTable& self) -> const DataTable& {return self.mass_dot;},  // getter
                      [](MassTable& self, const decltype(MassTable::mass_dot)& value) {self.mass_dot = value;})  // setter
        .def_property("cg",
                      [](const MassTable& self) -> const DataTable& {return self.cg;},  // getter
                      [](MassTable& self, const decltype(MassTable::cg)& value) {self.cg = value;})  // setter
        .def_property("thrust",
                      [](const MassTable& self) -> const DataTable& {return self.thrust;},  // getter
                      [](MassTable& self, const decltype(MassTable::thrust)& value) {self.thrust = value;})  // setter
        .def_property("dry_mass",
                      [](const MassTable& self) -> const double& {return self.dry_mass;},  // getter
                      [](MassTable& self, const decltype(MassTable::dry_mass)& value) {self.dry_mass = value;})  // setter
        .def_property("wet_mass",
                      [](const MassTable& self) -> const double& {return self.wet_mass;},  // getter
                      [](MassTable& self, const decltype(MassTable::wet_mass)& value) {self.wet_mass = value;})  // setter
        .def_property("vac_flag",
                      [](const MassTable& self) -> const double& {return self.vac_flag;},  // getter
                      [](MassTable& self, const decltype(MassTable::vac_flag)& value) {self.vac_flag = value;})  // setter
        .def_property("propellant_mass",
                      [](const MassTable& self) -> const vector<double>& {return self.propellant_mass;},  // getter
                      [](MassTable& self, const decltype(MassTable::propellant_mass)& value) {self.propellant_mass = value;})  // setter
        .def_property("burn_time",
                      [](const MassTable& self) -> const vector<double>& {return self.burn_time;},  // getter
                      [](MassTable& self, const decltype(MassTable::burn_time)& value) {self.burn_time = value;})  // setter
        .def_property("burn_time_max",
                      [](const MassTable& self) -> const double& {return self.burn_time_max;},  // getter
                      [](MassTable& self, const decltype(MassTable::burn_time_max)& value) {self.burn_time_max = value;})  // setter
        ;
}
