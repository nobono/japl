#include "../include/model.hpp"


PYBIND11_MODULE(model, m) {
    pybind11::class_<Model>(m, "Model")
        .def(py::init<>())
        .def("set_aerotable", &Model::set_aerotable)
        .def("set_atmosphere", &Model::set_atmosphere)
        .def_property("atmosphere",
                      [](const Model& self) -> const Atmosphere& {return self.atmosphere;},  // getter
                      [](Model& self, const decltype(Model::atmosphere)& value) {self.atmosphere = value;})  // setter
        .def_property("aerotable",
                      [](const Model& self) -> const AeroTable& {return self.aerotable;},  // getter
                      [](Model& self, const decltype(Model::aerotable)& value) {self.aerotable = value;})  // setter
        .def("set_input_function", &Model::set_input_function)
        .def("set_pre_update_functions", &Model::set_pre_update_functions)
        .def("set_post_update_functions", &Model::set_post_update_functions)
    ;
}
