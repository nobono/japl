#include "../include/aerotable.hpp"

#include <string>
#include <algorithm>

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


double AeroTable::inv_aerodynamics(const map<string, double>& kwargs) {
    // unpack kwargs
    double thrust = kwargs.at("thrust");
    double acc_cmd = kwargs.at("acc_cmd");
    double dynamic_pressure = kwargs.at("dynamic_pressure");
    double mass = kwargs.at("mass");
    double alpha = kwargs.at("alpha");
    double beta = kwargs.at("beta");
    double phi = kwargs.at("phi");
    double mach = kwargs.at("mach");
    double alt = kwargs.at("alt");
    double iota = kwargs.at("iota");

    AeroTable stage = this->get_stage();
    double alpha_tol = 0.01;
    dVec increment_alpha = this->increments["alpha"];
    double alpha_max = *std::max_element(increment_alpha.begin(), increment_alpha.end());
    Sref = this->get_Sref({});

    double alpha_last = -1000.0;
    int count = 0;

    bool boosting = (thrust > 0.0);

    // gradient search
    while ((abs(alpha - alpha_last) > alpha_tol) and (count < 10)) {
        count += 1;
        alpha_last = alpha;

        double CA;
        double CN;
        double CA_alpha;
        double CN_alpha;

        // TODO switch between Boost / Coast
        // get coeffs from aerotable
        map<string, double> table_args({{"alpha", alpha}, {"beta", beta}, {"phi", phi}, {"mach", mach}, {"alt", alt}, {"iota", iota}});
        if (boosting) {
            CA = stage.get_CA_Boost(table_args);
        } else {
            CA = stage.get_CA_Coast(table_args);
        }
        CN = stage.get_CNB(table_args);

        if (boosting) {
            CA_alpha = stage.get_CA_Boost_alpha(table_args);
        } else {
            CA_alpha = stage.get_CA_Coast_alpha(table_args);
        }
        CN = stage.get_CNB_alpha(table_args);

        // get derivative of CL wrt alpha
        double cosa = cos(alpha);
        double sina = sin(alpha);
        double CL = (CN * cosa) - (CA * sina);
        double CL_alpha = ((CN_alpha - CA) * cosa) - ((CA_alpha + CN) * sina);
        // CD = (CN * sina) + (CA * cosa)
        // CD_alpha = ((CA_alpha + CN) * cosa) + ((CN_alpha - CA) * sina)

        // calculate current normal acceleration, acc0, and normal acceleration due to
        // the change in alpha, acc_alpha. Use the difference between the two to
        // iteratively update alpha.
        double acc_alpha = CL_alpha * dynamic_pressure * Sref / mass + thrust * cos(alpha) / mass;
        double acc0 = CL * dynamic_pressure * Sref / mass + thrust * sin(alpha) / mass;
        alpha = alpha + (acc_cmd - acc0) / acc_alpha;
        alpha = fmax(0.0, fmin(alpha, alpha_max));
    }

    double angle_of_attack = alpha;
    return angle_of_attack;
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

        .def_property("increments",
                    [](const AeroTable& self) -> const decltype(AeroTable::increments)& {
                        // py::dict ret;
                        // for (const auto& item : self.increments) {
                        //     const char* key = item.first.c_str();
                        //     py::array_t<double> np_arr(item.second.size());
                        //     std::copy(item.second.begin(), item.second.end(), np_arr.mutable_data());
                        //     ret[key] = np_arr;
                        // }
                        // return ret;
                        return self.increments;
                    },  // getter
                    [](AeroTable& self, const map<string, py::array_t<double>>& value) {
                        map<string, dVec> _increments;
                        for (const auto& item : value) {
                            string key = item.first;
                            py::array_t<double> np_val = py::cast<py::array_t<double>>(item.second);

                            py::buffer_info buf = np_val.request();
                            double* ptr = static_cast<double*>(buf.ptr);
                            dVec val(ptr, ptr + buf.shape[0]);
                            _increments[key] = val;
                        }
                        self.increments = _increments;
                    })  // setter

        .def_property("Sref",
                      [](const AeroTable& self) -> const double& {return self.Sref;},  // getter
                      [](AeroTable& self, const decltype(AeroTable::Sref)& value) {self.Sref = value;})  // setter
        .def_property("Lref",
                      [](const AeroTable& self) -> const double& {return self.Lref;},  // getter
                      [](AeroTable& self, const decltype(AeroTable::Lref)& value) {self.Lref = value;})  // setter
        .def_property("MRC",
                      [](const AeroTable& self) -> const double& {return self.MRC;},  // getter
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
