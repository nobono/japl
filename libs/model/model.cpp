#include "model.hpp"


PYBIND11_MODULE(model, m) {
    pybind11::class_<Model>(m, "Model")
        .def(py::init<>())
        .def("set_aerotable", &Model::set_aerotable)
        .def("set_atmosphere", &Model::set_atmosphere)
        .def_readonly("aerotable", &Model::aerotable)
        .def_readonly("atmosphere", &Model::atmosphere)
    ;
}
