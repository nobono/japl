#ifndef _PYUTIL_H_
#define _PYUTIL_H_

#include <stdio.h>
#include <float.h>
#include <vector>
#include <array>
#include <string>
#include <map>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

using std::vector;
using std::array;
using std::string;
using std::map;
namespace py = pybind11;

template<class T>
inline py::array_t<T> convert_vec_to_numpy(const vector<T>& vec) {
    /* convert 1D vector<T> to numpy py::array_t<T> */
    py::array_t<T> ret(vec.size());
    std::copy(vec.begin(), vec.end(), ret.mutable_data());
    return ret;
}

// function to convert python dict or python kwargs to std::map
template<class Tdict, class Tkey, class Tval>
map<Tkey, Tval> convert_dict_to_map(const Tdict& py_dict_like) {
    map<Tkey, Tval> ret = {};
    for (const auto& item : py_dict_like) {
        ret[py::cast<Tkey>(item.first)] = py::cast<Tval>(item.second);
    }
    return ret;
}

// void unpack_dict_with_cast(const py::dict& py_dict) {
//     for (auto item : py_dict) {
//         // Cast keys and values to specific types
//         std::string key = py::cast<std::string>(item.first);
//         double value = py::cast<double>(item.second);

//         // Print the key and value
//         std::cout << "Key: " << key << ", Value: " << value << std::endl;
//     }
// }

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
