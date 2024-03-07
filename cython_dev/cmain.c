#include <stdio.h>
#include "Python.h"


int main(int argc, char *argv[]) {
    Py_Initialize();

    // Import the .pyd module
    PyObject *pName = PyUnicode_DecodeFSDefault("example");
    PyObject *pModule = PyImport_Import(pName);
    Py_DECREF(pName);

    if (pModule != NULL) {
        // Call a function from the module
        PyObject *pFunc = PyObject_GetAttrString(pModule, "myfunc");
        if (pFunc && PyCallable_Check(pFunc)) {
            PyObject *pArgs = PyTuple_New(2); // New reference
            PyObject *pValue1 = PyFloat_FromDouble(1.123); // New reference
            PyObject *pValue2 = PyFloat_FromDouble(2.456); // New reference
            PyTuple_SetItem(pArgs, 0, pValue1);
            PyTuple_SetItem(pArgs, 1, pValue2);

            PyObject *pResult = PyObject_CallObject(pFunc, pArgs);  // Assuming your function takes no arguments
            if (pResult != NULL) {
                printf("Call was successful\n");
                Py_DECREF(pResult);
            }
        }
    }

    printf("hello\n");
}