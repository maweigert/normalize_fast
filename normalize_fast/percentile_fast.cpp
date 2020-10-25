#include <Python.h>
#include "numpy/arrayobject.h"
#include "numpy/npy_math.h"
#include <cstdio>
#include <limits>

template <typename T> static PyObject* T_percentile_fast(PyObject *self, PyObject *args) {

  PyArrayObject *src_arr = NULL;

  float pmin, pmax;
  
  if (!PyArg_ParseTuple(args, "O!ff",
                        &PyArray_Type, &src_arr,
                        &pmin, &pmax))
    return NULL;

  npy_intp size = PyArray_SIZE(src_arr);
  
  T * src = (T*) PyArray_DATA(src_arr);

  const int32_t dtype_min = std::numeric_limits<T>::min();
  const int32_t dtype_max = std::numeric_limits<T>::max();

  const int32_t bins = dtype_max-dtype_min+1;

  int64_t i;
  
  int64_t counts[bins] = {0};
#pragma omp parallel for reduction(+:counts)
  for(i=0; i<size; i++) {
    counts[src[i]]++;
  }
 
  int64_t sum = 0;
  int64_t mi=0, ma=0;

  pmin = size*pmin/100;
  pmax = size*pmax/100;
  
  for (i=0; i<bins; i++) {
    sum += counts[i];
    if (sum>=pmin){
      mi = i;
      break;
    }
  }
  sum = 0;
  for (i=0; i<bins; i++) {
    sum += counts[i];
    if (sum>=pmax){
      ma = i;
      break;
    }
  }
  return Py_BuildValue("ii",mi,ma);
}


//------------------------------------------------------------------------

static struct PyMethodDef methods[] = {
                                       {"c_percentile_fast_uint8",
                                        T_percentile_fast<uint8_t>,
                                        METH_VARARGS,
                                        "c_percentile_fast_uint8"},
                                       
                                       {"c_percentile_fast_uint16",
                                        T_percentile_fast<uint16_t>,
                                        METH_VARARGS,
                                        "c_percentile_fast_uint16"},
                                       
                                       {"c_percentile_fast_int8",
                                        T_percentile_fast<int8_t>,
                                        METH_VARARGS,
                                        "c_percentile_fast_int8"},
                                       
                                       {"c_percentile_fast_int16",
                                        T_percentile_fast<int16_t>,
                                        METH_VARARGS,
                                        "c_percentile_fast_int16"},


                                       
                                       {NULL, NULL, 0, NULL}                                       
};

static struct PyModuleDef moduledef = {
                                       PyModuleDef_HEAD_INIT,
                                       "percentile_fast", 
                                       NULL,         
                                       -1,           
                                       methods,
                                       NULL,NULL,NULL,NULL
};

PyMODINIT_FUNC PyInit_percentile_fast(void) {
  import_array();
  return PyModule_Create(&moduledef);
}
