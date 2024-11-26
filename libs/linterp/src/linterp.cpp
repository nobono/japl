
#include <stdio.h>
#include <ctime>
#include "linterp.h"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>  // Enables automatic conversion


dVec linspace(double first, double last, int len) {
    dVec result(len);
    double step = (last-first) / (len - 1);
    for (int i=0; i<len; i++) {
        result[i] = first + i*step;
    }
    return result;
}

vector<double> get_grid_point(vector<dVec> const &gridList, vector<int> const &index) {
    const int N = index.size();
    vector<double> result(N);
    for (int i=0; i<N; i++) {
        result[i] = gridList[i][index[i]];
    }
    return result;
}

vector<dVec> process_tuple_of_arrays(pybind11::tuple input_tuple) {
    int N = input_tuple.size();
    vector<dVec> f_gridList(N);

    // Iterate over each element in the tuple
    for (size_t i = 0; i < N; ++i) {
        // Extract the array from the tuple
        pybind11::array array = input_tuple[i];
  
        // Verify that the item is a NumPy array
        if (!pybind11::isinstance<pybind11::array>(array)) {
            throw std::runtime_error("All elements of the tuple must be NumPy arrays");
        }
  
        // Request buffer information
        pybind11::buffer_info buf_info = array.request();
  
        // Access the data pointer if needed
        // For example, process a double array
        if (pybind11::isinstance<pybind11::array_t<double>>(array)) {
            // auto typed_array = array.cast<pybind11::array_t<double>>();
            double* data_ptr = static_cast<double*>(buf_info.ptr);
  
            // store in f_gridList
            f_gridList[i].resize(buf_info.size);
            for (int j = 0; j < buf_info.size; j++) {
                f_gridList[i][j] = *(data_ptr + j);
            }
  
          // for (ssize_t dim : buf_info.shape) {
          // }
          // Process the data as needed
        }
  
        // Repeat similar processing for other data types if necessary
      }
    return f_gridList;
}


typedef double T;
typedef EmptyClass ArrayRefCountT;
typedef EmptyClass GridRefCountT;
typedef boost::numeric::ublas::array_adaptor<T> grid_type;
typedef boost::const_multi_array_ref<T, 1> array_type;
typedef std::unique_ptr<array_type> array_type_ptr;


// Binding the function to Python
PYBIND11_MODULE(linterp, m) {
    pybind11::class_<NDInterpolator_1_ML>(m, "Interp1d")
        .def(py::init<py::tuple&, py::array_t<double>&>())
        .def("__call__", py::overload_cast<const py::tuple&>(&NDInterpolator_1_ML::interpolate, py::const_), "interpolation method")
        .def("__call__", py::overload_cast<const vector<dVec>&>(&NDInterpolator_1_ML::interpolate, py::const_), "interpolation method")
        .def_readonly("_f_gridList", &NDInterpolator_1_ML::_f_gridList, "")
        .def_readonly("_data", &NDInterpolator_1_ML::_data, "")
        .def(py::pickle(
            [](const NDInterpolator_1_ML &p) { // __getstate__
                /* Return a tuple that fully encodes the state of the object */
                return py::make_tuple(p._f_gridList, p._f_sizes, p._data);
            },
            [](py::tuple t) { // __setstate__
                if (t.size() != 3)
                    throw std::runtime_error("Invalid state!");

                /* Reinitialize Interp */
                vector<dVec> f_gridList = t[0].cast<vector<dVec>>();
                vector<int> f_sizes = t[1].cast<vector<int>>();
                py::array_t<double> data = t[2].cast<py::array_t<double>>();
                const int N = f_gridList.size();

                // process table values
                pybind11::array_t<double> f(f_sizes);
                vector<int> index(N);
                vector<double> arg(N);
                for (int i=0; i<f_sizes[0]; i++) {
                    index[0] = i;
                    arg = get_grid_point(f_gridList, index);
                    f.mutable_at(i) = *data.data(i);
                }
                auto begins_ends = get_begins_ends(f_gridList.begin(), f_gridList.end());
                NDInterpolator_1_ML p(begins_ends.first.begin(), f_sizes,
                                                       f.data(), f.data() + f.size());
                return p;
            }
        ));
    pybind11::class_<NDInterpolator_2_ML>(m, "Interp2d")
        .def(pybind11::init<py::tuple&, py::array_t<double>&>())
        .def("__call__", py::overload_cast<const py::tuple&>(&NDInterpolator_2_ML::interpolate, py::const_), "interpolation method")
        .def("__call__", py::overload_cast<const vector<dVec>&>(&NDInterpolator_2_ML::interpolate, py::const_), "interpolation method")
        .def_readonly("_f_gridList", &NDInterpolator_2_ML::_f_gridList, "")
        .def_readonly("_data", &NDInterpolator_2_ML::_data, "");
    pybind11::class_<NDInterpolator_3_ML>(m, "Interp3d")
        .def(pybind11::init<py::tuple&, py::array_t<double>&>())
        .def("__call__", py::overload_cast<const py::tuple&>(&NDInterpolator_3_ML::interpolate, py::const_), "interpolation method")
        .def("__call__", py::overload_cast<const vector<dVec>&>(&NDInterpolator_3_ML::interpolate, py::const_), "interpolation method")
        .def_readonly("_f_gridList", &NDInterpolator_3_ML::_f_gridList, "")
        .def_readonly("_data", &NDInterpolator_3_ML::_data, "");
    pybind11::class_<NDInterpolator_4_ML>(m, "Interp4d")
        .def(pybind11::init<py::tuple&, py::array_t<double>&>())
        .def("__call__", py::overload_cast<const py::tuple&>(&NDInterpolator_4_ML::interpolate, py::const_), "interpolation method")
        .def("__call__", py::overload_cast<const vector<dVec>&>(&NDInterpolator_4_ML::interpolate, py::const_), "interpolation method")
        .def_readonly("_f_gridList", &NDInterpolator_4_ML::_f_gridList, "")
        .def_readonly("_data", &NDInterpolator_4_ML::_data, "");
    pybind11::class_<NDInterpolator_5_ML>(m, "Interp5d")
        .def(pybind11::init<py::tuple&, py::array_t<double>&>())
        .def("__call__", py::overload_cast<const py::tuple&>(&NDInterpolator_5_ML::interpolate, py::const_), "interpolation method")
        .def("__call__", py::overload_cast<const vector<dVec>&>(&NDInterpolator_5_ML::interpolate, py::const_), "interpolation method")
        .def_readonly("_f_gridList", &NDInterpolator_5_ML::_f_gridList, "")
        .def_readonly("_data", &NDInterpolator_5_ML::_data, "");
}
