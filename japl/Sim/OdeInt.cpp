#include "OdeInt.hpp"
#include "numpy/ndarrayobject.h"
#include "numpy/ndarraytypes.h"



using namespace std;
using namespace boost::numeric::odeint;
using boost::math::quaternion;
using namespace boost::numeric::ublas;

typedef boost::numeric::ublas::vector <double, fixed_vector <double, 3> > vector3;
typedef boost::numeric::ublas::matrix <double, fixed_matrix <double, 3, 3> > matrix33;

typedef std::vector <double> state_type;
typedef runge_kutta_dopri5 <state_type> dopri5_stepper_type;
typedef runge_kutta_cash_karp54 <state_type> karp54_stepper_type;

/*
 * --------------------------------
 * Module Docs
 * --------------------------------
 */


static char module_docstring[] =
    "This module provides an interface for OdeInt C.";
static char odeint_docstring[] =
    "Calls boostlib numeric odeint method.";


/*
 *
 * --------------------------------
 * Init Ext. Module
 * --------------------------------
 *
 */

static PyMethodDef module_methods[] = {
    {"test", test, METH_VARARGS, odeint_docstring},
    {"add", add, METH_VARARGS, odeint_docstring},
    {"kwfunc", (PyCFunction)kwfunc, METH_VARARGS | METH_KEYWORDS, odeint_docstring},
    {"solve_ivp", (PyCFunction)solve_ivp, METH_VARARGS | METH_KEYWORDS, odeint_docstring},
    {NULL, NULL, 0, NULL}
};


PyMODINIT_FUNC PyInit_odeint(void)
{
    
    PyObject *module;
    static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "_odeint",
        module_docstring,
        -1,    // '1' allow multiple subinterpreters, '-1' does not allow
        module_methods,
        NULL,
        NULL,
        NULL,
        NULL    // m_free, a function to call during deallocation of the module
    };
    module = PyModule_Create(&moduledef);
    if (!module) return NULL;

    /* Load `numpy` functionality. */
    // import_array();
    import_array();

    return module;
}

 /*
  * --------------------------------
  * Methods
  * --------------------------------
  */

static PyObject *add(PyObject *self, PyObject *args)
{
  PyObject *arg1;
  PyObject *arg2;

  if (!PyArg_ParseTuple(args, "OO", &arg1, &arg2)) {
      return NULL; // if there was a problem parsing the arguments
  }

  // increase REF count if returning list
  if(PyList_Check(arg1)) {
    Py_INCREF(arg1);
  }

  double a = PyFloat_AsDouble(arg1);
  double b = PyFloat_AsDouble(arg2);

  // printf("%.5f", );
  // PyObject *ret = PyFloat_FromDouble(10.0);

  // Py_RETURN_NONE;
  return PyFloat_FromDouble(a + b);
}


static PyObject *kwfunc(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyObject *a;
  PyObject *b;

  static char *kwlist[] = {
    (char*)"a",
    (char*)"b",
    NULL}; // list of arg names
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|O!O!", kwlist, &PyFloat_Type, &a, &PyFloat_Type, &b)) {
      return NULL; // Return if the arguments are not as expected
  }

  double _a = PyFloat_AsDouble(a);
  double _b = PyFloat_AsDouble(b);

  return PyFloat_FromDouble(_a > _b);
}


// integrate_observer
struct push_back_state_and_time
{
    std::vector< state_type > &m_states;
    std::vector< double >& m_times;

    push_back_state_and_time( std::vector< state_type > &states , std::vector< double > &times )
    : m_states( states ) , m_times( times ) { }

    void operator()( const state_type &x , double t )
    {
        m_states.push_back( x );
        m_times.push_back( t );
    }
};


static PyObject *solve_ivp(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyObject *t_span;
  PyObject *dt;
  PyObject *y0;
  PyObject *method;
  PyObject *rtol;
  PyObject *atol;

  static char *kwlist[] = {
    (char*)"t_span",
    (char*)"dt",
    (char*)"y0",
    (char*)"method",
    (char*)"rtol",
    (char*)"atol",
    NULL}; // list of arg names
  //
  if (!PyArg_ParseTupleAndKeywords(
    args,
    kwargs,
    "|O!O!",
    kwlist,
    &PyList_Type, &t_span,
    &PyFloat_Type, &dt,
    &PyFloat_Type, &y0,
    &PyFloat_Type, &method,
    &PyFloat_Type, &rtol,
    &PyFloat_Type, &atol
    )) { return NULL; } // Return if the arguments are not as expected

  
  double *_t_span = PyList_ToDoubleArray(t_span);

  cout << _t_span[0] << endl; 
  cout << _t_span[1] << endl; 
  cout << _t_span[2] << endl; 

  // double dt = PyFloat_AsDouble(dt);
  // double y0 = PyFloat_AsDouble(y0);
  // double method = PyFloat_AsDouble(method);
  // double rtol = PyFloat_AsDouble(rtol);
  // double atol = PyFloat_AsDouble(atol);

  // return PyFloat_FromDouble(_a > _b);
  Py_RETURN_NONE;
}


double *PyList_ToDoubleArray(PyObject *pyList)
{
  if(!PyList_Check(pyList)){
    // handle error here
    return nullptr;
  }

  Py_ssize_t size = PyList_Size(pyList);
  double *c_array = new double[size];

  for(Py_ssize_t i = 0; i < size; i++){

    PyObject *item = PyList_GetItem(pyList, i);

    if(!PyFloat_Check(item)){
      // handle error here
      delete[] c_array;
      return nullptr;
    }

    c_array[i] = PyFloat_AsDouble(item);

  }

  return c_array;
}


double *PyArrayLike_ToDoubleArray(PyObject *pyArrayLike)
{

  // Py_ssize_t size = PyList_Size(pyArrayLike);
  // double *c_array = new double[size];

  // PyArray_FROMANY(pyobj, type, min, max, flags)
  PyObject *array = PyArray_FROMANY(pyArrayLike, NPY_FLOAT, 0, 0, NPY_ARRAY_C_CONTIGUOUS);

  if(array == NULL) {
    // handle error
  }

  // cast as PyArrayObject to get pointer to numpy ArrayObject
  // (not just pointer to PyObject). now you can use this with
  // numpy C API.
  PyArrayObject *np_array = reinterpret_cast<PyArrayObject*>(array);

  // You can get direct access to the data
  double* c_array = static_cast<double*>(PyArray_DATA(np_array));

  // Decrease the reference count of the Python object when you're done. 
  // Py_DECREF(np_array);

  return c_array;
}


static PyObject *test(PyObject *self, PyObject *args)
{
  PyObject *arg1;

  if (!PyArg_ParseTuple(args, "O", &arg1)) {
      return NULL; // if there was a problem parsing the arguments
  }

  PyObject *array = PyArray_FROMANY(arg1, NPY_FLOAT64, 0, 0, NPY_ARRAY_C_CONTIGUOUS);
  // PyArrayObject *np_array = reinterpret_cast<PyArrayObject*>(array);

  // // You can get direct access to the data
  // double* c_array = static_cast<double*>(PyArray_DATA(np_array));

  // cout << c_array[0] << endl;
  // cout << c_array[1] << endl;
  // cout << c_array[3] << endl;
  
  // Py_RETURN_NONE; 

  return array;
}


// double *_solve_ivp(
//     double* t_span,
//     double* dt,
//     double* y0,
//     char* method,
//     double rtol,
//     double atol
//     )
// {
//     // const int N = full_num_states / STATE_DIM;

//     // if (full_num_states % STATE_DIM != 0)
//     // {
//     //     std::cerr << "\033[1;31m Error: size of rigidbody state has changed. Modify\
//     //     FULL_STATE_DIM and recompile.\033[0m" << std::endl;
//     // }

//     auto stepper = make_controlled( atol , rtol , dopri5_stepper_type() );
//     
//     if (strcmp(method, "dopri") == 0)
//     {
//         auto stepper = make_controlled( atol , rtol , dopri5_stepper_type() );
//     } 
//     else if (strcmp(method, "karp") == 0)
//     {
//         auto stepper = make_controlled( atol , rtol , karp54_stepper_type() );
//     }

//     // solution storage
//     std::vector<state_type> state_log;
//     std::vector<double> time_log;
//     state_type x_0;

//     // set inital state
//     for (int i=0;i<N*STATE_DIM;i++)
//     {
//         x_0.push_back(states[i]);
//     }

//     {
//         // using bind
//         using std::placeholders::_1;
//         using std::placeholders::_2;
//         using std::placeholders::_3;

//         // Use std::bind to pass extra args
//         auto wrapper = bind(n_body_model(), _1, _2, _3, 
//                         N_masses,
//                         N_inertias,
//                         N_acc,
//                         N_gacc,
//                         N_forces,
//                         N_torques,
//                         N
//                         );

//         double t_start = t_span[0];
//         double t_end = t_span[1];
//         integrate_adaptive( stepper, wrapper, x_0 , t_start, t_end , dt, push_back_state_and_time(state_log, time_log));

//         // return state at final time
//         int end_idx = state_log.size() - 1;
//         double *ret = new double[N*STATE_DIM];
//         copy(state_log[end_idx].begin(), state_log[end_idx].end(), ret);

//         return ret;
//     }
// }


// using namespace std;
// using namespace boost::numeric::odeint;
// using boost::math::quaternion;
// using namespace boost::numeric::ublas;

// #define STATE_DIM           13

// const double softening = 3e-2; 
// const double G = 6.67430E-11;   // const of gravity

// typedef boost::numeric::ublas::vector <double, fixed_vector <double, 3> > vector3;
// typedef boost::numeric::ublas::matrix <double, fixed_matrix <double, 3, 3> > matrix33;

// typedef std::vector <double> state_type;
// typedef runge_kutta_dopri5 <state_type> dopri5_stepper_type;
// typedef runge_kutta_cash_karp54 <state_type> karp54_stepper_type;

// struct n_body_model
// {
//     template< class State , class Deriv >
//     void operator()(
//         const State &states, 
//         Deriv &dstate_dt, 
//         double t, 
//         double N_masses[], 
//         double N_inertias[][9],
//         double N_acc[][3],
//         double N_gacc[][3],
//         double N_forces[][3],
//         double N_torques[][3],
//         const int N
//     ) const
//     {
//         
//         double x[N];
//         double y[N];
//         double z[N];
//         double vx[N];
//         double vy[N];
//         double vz[N];
//         for (int i=0;i<N;i++) 
//         {
//             x[i] = states[0 + i*STATE_DIM];
//             y[i] = states[1 + i*STATE_DIM];
//             z[i] = states[2 + i*STATE_DIM];
//             vx[i] = states[3 + i*STATE_DIM];
//             vy[i] = states[4 + i*STATE_DIM];
//             vz[i] = states[5 + i*STATE_DIM];
//         }

//         // stores all pairwise particle separations: r_j - r_i
//         double dx[N][N];
//         double dy[N][N];
//         double dz[N][N];
//         for (int i=0;i<N;i++)
//         {
//             for (int j=0;j<N;j++)
//             {
//                 dx[i][j] = x[j] - x[i];
//                 dy[i][j] = y[j] - y[i];
//                 dz[i][j] = z[j] - z[i];
//             }
//         }

//         // calc inv_r3 
//         double inv_r3[N][N];
//         double soft_2 = pow(softening, 2.0);
//         for (int i=0;i<N;i++)
//         {
//             for (int j=0;j<N;j++)
//             {
//                 inv_r3[i][j] = pow(dx[i][j], 2.0) + pow(dy[i][j], 2.0) + pow(dz[i][j], 2.0) + soft_2;
//                 if (inv_r3[i][j] > 0.0)
//                 {
//                     inv_r3[i][j] = pow(inv_r3[i][j], -1.5);
//                 }
//             }
//         }

//         // calc acclerations 
//         double dx_inv_r3[N][N];
//         double dy_inv_r3[N][N];
//         double dz_inv_r3[N][N];
//         double ax[N];
//         double ay[N];
//         double az[N];
//         for (int i=0;i<N;i++)
//         {
//             for (int j=0;j<N;j++)
//             {
//                 dx_inv_r3[i][j] = G * (dx[i][j] * inv_r3[i][j]);
//                 dy_inv_r3[i][j] = G * (dy[i][j] * inv_r3[i][j]);
//                 dz_inv_r3[i][j] = G * (dz[i][j] * inv_r3[i][j]);
//             }
//         }
//         for (int i=0;i<N;i++)
//         {
//             ax[i] = 0.0;
//             ay[i] = 0.0;
//             az[i] = 0.0;
//             for (int j=0;j<N;j++)
//             {
//                 ax[i] += dx_inv_r3[i][j] * N_masses[j];
//                 ay[i] += dy_inv_r3[i][j] * N_masses[j];
//                 az[i] += dz_inv_r3[i][j] * N_masses[j];
//             }
//             // store accelerartion due to gravity (out of state)
//             N_gacc[i][0] = ax[i];
//             N_gacc[i][1] = ay[i];
//             N_gacc[i][2] = az[i];
//         }

//         // external forces
//         for (int i=0;i<N;i++)
//         {
//             ax[i] += N_forces[i][0] / N_masses[i];
//             ay[i] += N_forces[i][1] / N_masses[i];
//             az[i] += N_forces[i][2] / N_masses[i];
//             N_acc[i][0] = ax[i];
//             N_acc[i][1] = ay[i];
//             N_acc[i][2] = az[i];
//         }

//         // get quaternion and angular velocity from states
//         double quats[N][4];
//         double ang_vels[N][3];
//         for (int i=0;i<N;i++)
//         {
//             // get quats
//             for (int j=0;j<4;j++)
//             {
//                 quats[i][j] = states[i*STATE_DIM + j + 6]; // quat index starts at 6
//             }
//             // get ang_vels
//             for (int j=0;j<3;j++)
//             {
//                 ang_vels[i][j] = states[i*STATE_DIM + j + 10];
//             }
//         }

//         // initialize rotational vars
//         double dot_quat[N][4];
//         // double ddot_quat[N][4];
//         double dot_ang_vel[N][3];
//         for (int i=0;i<N;i++)
//         {
//             for (int j=0;j<4;j++)
//             {
//                 dot_quat[i][j] = 0.0;
//                 // ddot_quat[i][j] = 0.0;
//             }
//             for (int j=0;j<3;j++)
//             {
//                 dot_ang_vel[i][j] = 0.0;
//             }
//         }

//         // rotational dynamics
//         for (int i=0;i<N;i++)
//         {
//             quaternion <double> ang_vel(0, ang_vels[i][0], ang_vels[i][1], ang_vels[i][2]);

//             // calculate angular accelaeration (inv(inertia) * torque)
//             double _ang_acc[3];
//             double torque[3] = {N_torques[i][0], N_torques[i][1], N_torques[i][2]};
//             double inertia[9];
//             inertia[0] = N_inertias[i][0]; 
//             inertia[1] = N_inertias[i][1]; 
//             inertia[2] = N_inertias[i][2];
//             inertia[3] = N_inertias[i][3]; 
//             inertia[4] = N_inertias[i][4];
//             inertia[5] = N_inertias[i][5];
//             inertia[6] = N_inertias[i][6]; 
//             inertia[7] = N_inertias[i][7];
//             inertia[8] = N_inertias[i][8];
//             _ang_acc[0] = (inertia[1]*inertia[5] - inertia[2]*inertia[4])*torque[2]/(inertia[0]*inertia[4]*inertia[8] - inertia[0]*inertia[5]*inertia[7] - inertia[1]*inertia[3]*inertia[8] + inertia[1]*inertia[5]*inertia[6] + inertia[2]*inertia[3]*inertia[7] - inertia[2]*inertia[4]*inertia[6]) + (-inertia[1]*inertia[8] + inertia[2]*inertia[7])*torque[1]/(inertia[0]*inertia[4]*inertia[8] - inertia[0]*inertia[5]*inertia[7] - inertia[1]*inertia[3]*inertia[8] + inertia[1]*inertia[5]*inertia[6] + inertia[2]*inertia[3]*inertia[7] - inertia[2]*inertia[4]*inertia[6]) + (inertia[4]*inertia[8] - inertia[5]*inertia[7])*torque[0]/(inertia[0]*inertia[4]*inertia[8] - inertia[0]*inertia[5]*inertia[7] - inertia[1]*inertia[3]*inertia[8] + inertia[1]*inertia[5]*inertia[6] + inertia[2]*inertia[3]*inertia[7] - inertia[2]*inertia[4]*inertia[6]);
//             _ang_acc[1] = (-inertia[0]*inertia[5] + inertia[2]*inertia[3])*torque[2]/(inertia[0]*inertia[4]*inertia[8] - inertia[0]*inertia[5]*inertia[7] - inertia[1]*inertia[3]*inertia[8] + inertia[1]*inertia[5]*inertia[6] + inertia[2]*inertia[3]*inertia[7] - inertia[2]*inertia[4]*inertia[6]) + (inertia[0]*inertia[8] - inertia[2]*inertia[6])*torque[1]/(inertia[0]*inertia[4]*inertia[8] - inertia[0]*inertia[5]*inertia[7] - inertia[1]*inertia[3]*inertia[8] + inertia[1]*inertia[5]*inertia[6] + inertia[2]*inertia[3]*inertia[7] - inertia[2]*inertia[4]*inertia[6]) + (-inertia[3]*inertia[8] + inertia[5]*inertia[6])*torque[0]/(inertia[0]*inertia[4]*inertia[8] - inertia[0]*inertia[5]*inertia[7] - inertia[1]*inertia[3]*inertia[8] + inertia[1]*inertia[5]*inertia[6] + inertia[2]*inertia[3]*inertia[7] - inertia[2]*inertia[4]*inertia[6]);
//             _ang_acc[2] = (inertia[0]*inertia[4] - inertia[1]*inertia[3])*torque[2]/(inertia[0]*inertia[4]*inertia[8] - inertia[0]*inertia[5]*inertia[7] - inertia[1]*inertia[3]*inertia[8] + inertia[1]*inertia[5]*inertia[6] + inertia[2]*inertia[3]*inertia[7] - inertia[2]*inertia[4]*inertia[6]) + (-inertia[0]*inertia[7] + inertia[1]*inertia[6])*torque[1]/(inertia[0]*inertia[4]*inertia[8] - inertia[0]*inertia[5]*inertia[7] - inertia[1]*inertia[3]*inertia[8] + inertia[1]*inertia[5]*inertia[6] + inertia[2]*inertia[3]*inertia[7] - inertia[2]*inertia[4]*inertia[6]) + (inertia[3]*inertia[7] - inertia[4]*inertia[6])*torque[0]/(inertia[0]*inertia[4]*inertia[8] - inertia[0]*inertia[5]*inertia[7] - inertia[1]*inertia[3]*inertia[8] + inertia[1]*inertia[5]*inertia[6] + inertia[2]*inertia[3]*inertia[7] - inertia[2]*inertia[4]*inertia[6]);
//             quaternion <double> ang_acc(0, _ang_acc[0], _ang_acc[1], _ang_acc[2]);

//             // calc dot_quat
//             quaternion <double> quat(quats[i][0], quats[i][1], quats[i][2], quats[i][3]);
//             quaternion <double> dquat_dt = 0.5 * ang_vel * quat;
//             quaternion <double> d2quat_dt2 = 0.5 * (ang_acc * quat + ang_vel * dquat_dt);
//             quaternion <double> quat_conj = boost::math::conj(quat);
//             quaternion <double> ang_acc2 = 2.0 * (d2quat_dt2 * quat_conj) - 2.0 * boost::math::pow(dquat_dt * quat_conj, 2.0);
//             
//             dot_ang_vel[i][0] = ang_acc2.R_component_2();
//             dot_ang_vel[i][1] = ang_acc2.R_component_3();
//             dot_ang_vel[i][2] = ang_acc2.R_component_4();

//             dot_quat[i][0] = dquat_dt.R_component_1();
//             dot_quat[i][1] = dquat_dt.R_component_2();
//             dot_quat[i][2] = dquat_dt.R_component_3();
//             dot_quat[i][3] = dquat_dt.R_component_4();
//         }
//             //-----------------------------------------------------------------

//         // store change in state
//         for (int i=0;i<N;i++)
//         {
//             dstate_dt[0 + i*STATE_DIM] = vx[i];
//             dstate_dt[1 + i*STATE_DIM] = vy[i];
//             dstate_dt[2 + i*STATE_DIM] = vz[i];
//             dstate_dt[3 + i*STATE_DIM] = ax[i];
//             dstate_dt[4 + i*STATE_DIM] = ay[i];
//             dstate_dt[5 + i*STATE_DIM] = az[i];
//             dstate_dt[6 + i*STATE_DIM] = dot_quat[i][0];
//             dstate_dt[7 + i*STATE_DIM] = dot_quat[i][1];
//             dstate_dt[8 + i*STATE_DIM] = dot_quat[i][2];
//             dstate_dt[9 + i*STATE_DIM] = dot_quat[i][3];
//             dstate_dt[10 + i*STATE_DIM] = dot_ang_vel[i][0];
//             dstate_dt[11 + i*STATE_DIM] = dot_ang_vel[i][1];
//             dstate_dt[12 + i*STATE_DIM] = dot_ang_vel[i][2];
//         }
//     }
// };


// #ifdef __cplusplus
// 	extern "C" {
// #endif


// void free_double_array_p(double *p)
// {
// 	free(p);
// }

// void matmul(void){
//     // Declare two 3x3 matrices 
//     using namespace boost::numeric::ublas; 
// 	matrix<double> A (3,3); 
// 	matrix<double> B (3,3); 
// 	
// 	// Initialize the elements of A 
// 	for (unsigned i = 0; i < A.size1 (); ++ i) 
// 		for (unsigned j = 0; j < A.size2 (); ++ j) 
// 			A (i, j) = 3 * i + j; 
// 	
// 	// Initialize the elements of B 
// 	for (unsigned i = 0; i < B.size1 (); ++ i) 
// 		for (unsigned j = 0; j < B.size2 (); ++ j) 
// 			B (i, j) = 3 * i + j; 
// 	
// 	// Declare the output matrix 
// 	matrix<double> C (3,3); 
// 	
// 	// Perform the matrix multiplication 
// 	C = prod (A, B); 
// 	
// 	// Print the output matrix 
// 	std::cout << C << std::endl; 
// }


// void test(){
//     printf("test\n");
// }


// // integrate_observer
// struct push_back_state_and_time
// {
//     std::vector< state_type > &m_states;
//     std::vector< double >& m_times;

//     push_back_state_and_time( std::vector< state_type > &states , std::vector< double > &times )
//     : m_states( states ) , m_times( times ) { }

//     void operator()( const state_type &x , double t )
//     {
//         m_states.push_back( x );
//         m_times.push_back( t );
//     }
// };


// double *solve_ivp(
//     double start_time, 
//     double end_time, 
//     double dt,
//     double *states,
//     double *N_masses,
//     double N_inertias[][9],
//     double N_acc[][3],
//     double N_gacc[][3],
//     double N_forces[][3],
//     double N_torques[][3],
//     int full_num_states,
//     char* method,
//     double rtol,
//     double atol
//     )
// {
//     const int N = full_num_states / STATE_DIM;

//     if (full_num_states % STATE_DIM != 0)
//     {
//         std::cerr << "\033[1;31m Error: size of rigidbody state has changed. Modify\
//         FULL_STATE_DIM and recompile.\033[0m" << std::endl;
//     }

//     auto stepper = make_controlled( atol , rtol , dopri5_stepper_type() );
//     
//     if (strcmp(method, "dopri") == 0)
//     {
//         auto stepper = make_controlled( atol , rtol , dopri5_stepper_type() );
//     } 
//     else if (strcmp(method, "karp") == 0)
//     {
//         auto stepper = make_controlled( atol , rtol , karp54_stepper_type() );
//     }

//     // solution storage
//     std::vector<state_type> state_log;
//     std::vector<double> time_log;
//     state_type x_0;

//     // set inital state
//     for (int i=0;i<N*STATE_DIM;i++)
//     {
//         x_0.push_back(states[i]);
//     }

//     {
//         // using bind
//         using std::placeholders::_1;
//         using std::placeholders::_2;
//         using std::placeholders::_3;

//         // Use std::bind to pass extra args
//         auto wrapper = bind(n_body_model(), _1, _2, _3, 
//                         N_masses,
//                         N_inertias,
//                         N_acc,
//                         N_gacc,
//                         N_forces,
//                         N_torques,
//                         N
//                         );

//         integrate_adaptive( stepper, wrapper, x_0 , start_time, end_time , dt, push_back_state_and_time(state_log, time_log));

//         // return state at final time
//         int end_idx = state_log.size() - 1;
//         double *ret = new double[N*STATE_DIM];
//         copy(state_log[end_idx].begin(), state_log[end_idx].end(), ret);

//         return ret;
//     }
// }


// #ifdef __cplusplus
// 	}
// #endif
