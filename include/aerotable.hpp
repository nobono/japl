#ifndef _AEROTABLE_H_
#define _AEROTABLE_H_

#include <variant>
#include <map>

#include "datatable.hpp"
#include <pybind11/stl.h>
#include "linterp/linterp.h"
#include "boost/multi_array.hpp"
#include "datatable.hpp"



class AeroTable {

public:

    map<string, int> table_info {
        {"Sref", 1},
        {"Lref", 2},
        {"MRC", 3},
        {"CA", 4},
        {"CA_Boost", 5},
        {"CA_Coast", 6},
        {"CNB", 7},
        {"CYB", 8},
        {"CA_Boost_alpha", 9},
        {"CA_Coast_alpha", 10},
        {"CNB_alpha", 11},
    };

    vector<AeroTable> stages;
    int stage_id = 0;
    bool is_stage = true;

    map<string, dVec> increments;

    DataTable CA;
    DataTable CA_Boost;
    DataTable CA_Coast;
    DataTable CNB;
    DataTable CYB;
    DataTable CA_Boost_alpha;
    DataTable CA_Coast_alpha;
    DataTable CNB_alpha;
    double Sref = 0.0;
    double Lref = 0.0;
    double MRC = 0.0;

    AeroTable() = default;
    ~AeroTable() = default;

    AeroTable(const py::kwargs& kwargs);

    // copy constructor
    AeroTable(const AeroTable& other)
    :
        CA(other.CA),
        CA_Boost(other.CA_Boost),
        CA_Coast(other.CA_Coast),
        CNB(other.CNB),
        CYB(other.CYB),
        CA_Boost_alpha(other.CA_Boost_alpha),
        CA_Coast_alpha(other.CA_Coast_alpha),
        CNB_alpha(other.CNB_alpha),
        Sref(other.Sref),
        Lref(other.Lref),
        MRC(other.MRC) {}

    AeroTable& operator=(const AeroTable& other) {
        if (this == &other) {
            return *this; // Handle self-assignment
        }
        this->CA = other.CA;
        this->CA.interp = other.CA.interp;
        this->CA.axes = other.CA.axes;
        this->CA_Boost = other.CA_Boost;
        this->CA_Coast = other.CA_Coast;
        this->CNB = other.CNB;
        this->CYB = other.CYB;
        this->CA_Boost_alpha = other.CA_Boost_alpha;
        this->CA_Coast_alpha = other.CA_Coast_alpha;
        this->CNB_alpha = other.CNB_alpha;
        this->Sref = other.Sref;
        this->Lref = other.Lref;
        this->MRC = other.MRC;
        return *this;
    }

    double get_Sref(const map<string, double>& kwargs) {
        return this->Sref;
    }

    double get_Lref(const map<string, double>& kwargs) {
        return this->Lref;
    }

    double get_MRC(const map<string, double>& kwargs) {
        return this->MRC;
    }

    double get_CA(const map<string, double>& kwargs) {
        return this->CA(kwargs)[0];
    }

    double get_CA_Boost(const map<string, double>& kwargs) {
        return this->CA_Boost(kwargs)[0];
    }

    double get_CA_Coast(const map<string, double>& kwargs) {
        return this->CA_Coast(kwargs)[0];
    }

    double get_CNB(const map<string, double>& kwargs) {
        return this->CNB(kwargs)[0];
    }

    double get_CYB(const map<string, double>& kwargs) {
        return this->CYB(kwargs)[0];
    }

    double get_CA_Boost_alpha(const map<string, double>& kwargs) {
        return this->CA_Boost_alpha(kwargs)[0];
    }

    double get_CA_Coast_alpha(const map<string, double>& kwargs) {
        return this->CA_Coast_alpha(kwargs)[0];
    }

    double get_CNB_alpha(const map<string, double>& kwargs) {
        return this->CNB_alpha(kwargs)[0];
    }

    double inv_aerodynamics(const map<string, double>& kwargs);

    void add_stage(AeroTable& aerotable) {
        this->is_stage = false;
        aerotable.is_stage = true;
        this->stages.push_back(aerotable);
    }

    void set_stage(int& stage) {
        /*
         * Set the current stage index for the aerotable. This is
         * so that \"get_stage()\" will return the corresponding aerotable.
        */
        if (stage >= this->stages.size()) {
            string err_msg = "cannot access stage " + std::to_string(stage) +
                " for container of size " + std::to_string(this->stages.size());
            throw std::invalid_argument(err_msg);
        }
        this->stage_id = stage;
    }

    AeroTable get_stage(void) {
        if (this->is_stage) {
            return *this;
        } else {
            return this->stages[this->stage_id];
        }
    }

    inline vector<string> get_keys() {
        vector<string> keys;
        for (const auto& pair : table_info) {
            keys.push_back(pair.first);
        }
        return keys;
    }

    vector<vector<double>> _kwargs_to_args(py::kwargs kwargs) {
        vector<vector<double>> args;
        vector<double> point;
        for (auto& item : kwargs) {
            point.push_back(item.second.cast<double>());
        }
        args.push_back(point);
        return args;
    }

    void set_attr_from_id(double& val, int& id) {
        switch (id) {
            case 1:
                Sref = val;
                break;
            case 2:
                Lref = val;
                break;
            case 3:
                MRC = val;
                break;
            default:
                throw std::invalid_argument("unhandled case.");
        }
    }

    void set_table_from_id(DataTable& table, int& id) {
        switch (id) {
            case 4:
                CA = table;
                break;
            case 5:
                CA_Boost = table;
                break;
            case 6:
                CA_Coast = table;
                break;
            case 7:
                CNB = table;
                break;
            case 8:
                CYB = table;
                break;
            case 9:
                CA_Boost_alpha = table;
                break;
            case 10:
                CA_Coast_alpha = table;
                break;
            case 11:
                CNB_alpha = table;
                break;
            default:
                throw std::invalid_argument("unhandled case.");
        }
    }
};

#endif
