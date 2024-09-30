#ifndef _PYUTIL_H_
#define _PYUTIL_H_

#include <stdio.h>
#include <float.h>
#include <vector>
#include <array>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

using std::vector;
using std::array;
namespace py = pybind11;



inline vector<double> array_t_to_vector(py::array arr) {
    // Verify that the item is a NumPy array
    if (!py::isinstance<py::array>(arr)) {
        throw std::runtime_error("element is not NumPy array");
    }
    // request buffer to numpy array
    py::buffer_info buf_info = arr.request();
    if (py::isinstance<py::array_t<double>>(arr)) {
        // get pointer to data
        double* ptr = static_cast<double*>(buf_info.ptr);
        // create vector from pointer and array size
        return vector<double>(ptr, ptr + buf_info.size);
    }
    else if (py::isinstance<py::array_t<float>>(arr)) {
        vector<double> ret(buf_info.size);
        float* ptr = static_cast<float*>(buf_info.ptr);
        for (int i=0; i<buf_info.size; i++) {
            ret[i] = (double)*(ptr + i);
        }
        return ret;
    }
    else if (py::isinstance<py::array_t<int64_t>>(arr)) {
        vector<double> ret(buf_info.size);
        int64_t* ptr = static_cast<int64_t*>(buf_info.ptr);
        for (int i=0; i<buf_info.size; i++) {
            ret[i] = (double)*(ptr + i);
        }
        return ret;
    }
    else if (py::isinstance<py::array_t<int>>(arr)) {
        vector<double> ret(buf_info.size);
        int* ptr = static_cast<int*>(buf_info.ptr);
        for (int i=0; i<buf_info.size; i++) {
            ret[i] = (double)*(ptr + i);
        }
        return ret;
    } else {
        throw std::runtime_error("in array_t_to_vector(): unhandled conversion"
                                 "case for array datatype.");
    }
}



#endif // _PYUTIL_H_
