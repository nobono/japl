#ifndef _MASSTABLE_H_
#define _MASSTABLE_H_

#include <variant>
#include <map>

#include "datatable.hpp"
#include "staged.hpp"
#include <pybind11/stl.h>
#include "linterp/linterp.h"
#include "boost/multi_array.hpp"
#include "datatable.hpp"
#include <algorithm>
#include <cmath>



class MassTable {

public:

    map<string, int> table_info {
        {"nozzle_area", 1},
        {"dry_mass", 2},
        {"wet_mass", 3},
        {"vac_flag", 4},
        {"propellant_mass", 5},
        {"burn_time", 6},
        {"mass_dot", 7},
        {"cg", 8},
        {"thrust", 9}
    };

    vector<MassTable> stages;
    int stage_id = 0;
    bool is_stage = true;

    DataTable mass_dot;
    DataTable cg;
    DataTable thrust;
    double nozzle_area;
    double dry_mass;
    double wet_mass;
    double vac_flag;
    vector<double> propellant_mass;
    vector<double> burn_time;
    double burn_time_max;

    MassTable() = default;
    ~MassTable() = default;

    MassTable(const py::kwargs& kwargs);

    // copy constructor
    MassTable(const MassTable& other)
    :
        mass_dot(other.mass_dot),
        cg(other.cg),
        thrust(other.thrust),
        nozzle_area(other.nozzle_area),
        dry_mass(other.dry_mass),
        wet_mass(other.wet_mass),
        vac_flag(other.vac_flag),
        propellant_mass(other.propellant_mass),
        burn_time(other.burn_time)
    {
        this->burn_time_max = *std::max_element(this->burn_time.begin(),
                                                this->burn_time.end());
    }

    // move constructor
    MassTable& operator=(const MassTable& other) {
        if (this == &other) {
            return *this; // Handle self-assignment
        }
        this->mass_dot = other.mass_dot;
        this->cg = other.cg;
        this->thrust = other.thrust;
        this->nozzle_area = other.nozzle_area;
        this->dry_mass = other.dry_mass;
        this->wet_mass = other.wet_mass;
        this->vac_flag = other.vac_flag;
        this->propellant_mass = other.propellant_mass;
        this->burn_time = other.burn_time;
        this->burn_time_max = *std::max_element(other.burn_time.begin(),
                                                other.burn_time.end());
        return *this;
    }


    void add_stage(MassTable& stage) {
        this->is_stage = false;
        stage.is_stage = true;
        this->stages.push_back(stage);
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

    MassTable& get_stage(void) {
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
                nozzle_area = val;
                break;
            case 2:
                dry_mass = val;
                break;
            case 3:
                wet_mass = val;
                break;
            case 4:
                vac_flag = val;
                break;
            default:
                throw std::invalid_argument("unhandled case.");
        }
    }

    void set_array_from_id(vector<double>& val, int&id) {
        switch (id) {
            case 5:
                propellant_mass = val;
                break;
            case 6:
                burn_time = val;
                burn_time_max = *std::max_element(val.begin(),
                                                  val.end());
                break;
            default:
                throw std::invalid_argument("unhandled case.");
        }

    }

    void set_table_from_id(DataTable& table, int& id) {
        switch (id) {
            case 7:
                mass_dot = table;
                break;
            case 8:
                cg = table;
                break;
            case 9:
                thrust = table;
                break;
            default:
                throw std::invalid_argument("unhandled case.");
        }
    }


    double get_wet_mass(void) {
        return this->get_stage().wet_mass;
    }


    double get_dry_mass(void) {
        return this->get_stage().dry_mass;
    }


    /*
     * Parameters:
     *      t:
    */
    double get_mass_dot(const map<string, double>& kwargs) {
        MassTable stage = this->get_stage();
        if (kwargs.at("t") >= stage.burn_time_max) {
            return 0.0;
        } else {
            return stage.mass_dot(kwargs)[0];
        }
    }


    /*
     * Parameters:
     *      t:
    */
    double get_cg(const map<string, double>& kwargs) {
        MassTable stage = this->get_stage();
        if (kwargs.at("t") >= stage.burn_time_max) {
            map<string, double> _args = {{"burn_time_max", this->burn_time_max}};
            return stage.cg(_args)[0];
        } else {
            return stage.cg(kwargs)[0];
        }
    }


    /*
     * Parameters:
     *      t:
     *      pressure:
    */
    double get_isp(const map<string, double>& kwargs) {
        MassTable stage = this->get_stage();
        double thrust = stage.get_thrust(kwargs);
        double g0 = 9.80665;
        double mass_dot = stage.get_mass_dot(kwargs);
        double isp = thrust / (mass_dot * g0);
        return isp;
    }


    /*
     * Parameters:
     *      t:
    */
    double get_raw_thrust(const map<string, double>& kwargs) {
        // (vac_thrust)
        MassTable stage = this->get_stage();
        if (kwargs.at("t") >= stage.burn_time_max) {
            return 0.0;
        } else {
            return stage.thrust(kwargs)[0];
        }
    }


    /*
     * Parameters:
     *      t:
     *      pressure:
    */
    double get_thrust(const map<string, double>& kwargs) {
        MassTable stage = this->get_stage();
        if (kwargs.at("t") <= stage.burn_time_max) {
            double raw_thrust = stage.get_raw_thrust(kwargs);
            double pressure = kwargs.at("pressure");
            double thrust;
            if (stage.vac_flag) {
                double vac_thrust = raw_thrust;
                thrust = std::max(vac_thrust - std::signbit(vac_thrust) * stage.nozzle_area * pressure, 0.0);
            } else {
                thrust = raw_thrust;
                double vac_thrust = thrust + stage.nozzle_area * pressure;
            }
            return thrust;
        } else {
            return 0.0;
        }
    }
};

#endif
