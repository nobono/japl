#ifndef _ATMOSPHERE_H_
#define _ATMOSPHERE_H_

#include <vector>

#include "../linterp/src/linterp.h"
#include "../boost/multi_array.hpp"

extern vector<double> _alts;
extern vector<double> _pressure;
extern vector<double> _density;
extern vector<double> _temperature;
extern vector<double> _speed_of_sound;
extern vector<double> _grav_accel;


class Atmosphere {

public:
    NDInterpolator_1_ML interp_pressure = _create_ndinterp(_alts, _pressure);
    NDInterpolator_1_ML interp_density = _create_ndinterp(_alts, _density);
    NDInterpolator_1_ML interp_temperature = _create_ndinterp(_alts, _temperature);
    NDInterpolator_1_ML interp_speed_of_sound = _create_ndinterp(_alts, _speed_of_sound);
    NDInterpolator_1_ML interp_grav_accel = _create_ndinterp(_alts, _grav_accel);

    Atmosphere() {};

    double pressure(double alt);
    double temperature(double alt);
    double density(double alt);
    double speed_of_sound(double alt);
    double grav_accel(double alt);
    double dynamics_pressure(vector<double> vel, double alt);

private:
    // Creates 1D NDInterpolator object from 2 vectors
    static NDInterpolator_1_ML _create_ndinterp(vector<double> axis, vector<double> data) {
        const int N = 1;

        int f_len = axis.size();

        // construct the grid in each dimension
        vector<dVec> f_gridList(N);
        for (size_t i = 0; i < N; ++i) {
            f_gridList[i] = axis;
        }

        // size of the grid in each dimension
        array<int, N> f_sizes;
        for (int i = 0; i < N; ++i) {
            f_sizes[i] = f_len;
        }

        // fill in the values of f(x) at the gridpoints
        boost::multi_array<double, N> f(f_sizes);
        vector<int> index(N);
        vector<double> arg(N);
        for (int i = 0; i < f_sizes[0]; ++i) {
            index[0] = i;
            f(index) = data[i];
        }

        auto begins_ends = get_begins_ends(f_gridList.begin(), f_gridList.end());
        NDInterpolator_1_ML interp_multilinear(begins_ends.first.begin(), f_sizes,
                                               f.data(), f.data() + f.size());
        return interp_multilinear;
    };
};


#endif
