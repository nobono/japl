#ifndef __ODEINT_H__
#define __ODEINT_H__


#include <iostream>
#include <stdio.h>

#include <Python.h>
#include <numpy/arrayobject.h>

#include <boost/numeric/odeint.hpp>
#include <boost/math/quaternion.hpp>

#include <boost/qvm/quat_operations.hpp>
#include <boost/numeric/ublas/io.hpp> 
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_sparse.hpp>



/*
 * --------------------------------
 * Method Prototypes
 * --------------------------------
 */

static PyObject *test(PyObject *self, PyObject *args);
static PyObject *add(PyObject *self, PyObject *args);
static PyObject *kwfunc(PyObject *self, PyObject *args, PyObject *kwargs);
static PyObject *solve_ivp(PyObject *self, PyObject *args, PyObject *kwargs);

double *PyList_ToDoubleArray(PyObject *pyList);
double *pyArrayLike_ToDoubleArray(PyObject *pyArrayLike);



#endif
