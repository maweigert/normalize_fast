#include <Python.h>
#include "numpy/arrayobject.h"
#include "numpy/npy_math.h"
#include <cstdio>
#include <limits>


#ifdef _OPENMP
#include <omp.h>
#endif

template <typename T> static PyObject* T_normalize_fast(PyObject *self, PyObject *args) {

  PyArrayObject *src_arr = NULL;
  PyArrayObject *dst_arr = NULL;

  float mi, ma, mi2, ma2;
  int verbose;
  
  if (!PyArg_ParseTuple(args, "O!O!ffffi",
                        &PyArray_Type, &src_arr,
                        &PyArray_Type, &dst_arr,
                        &mi, &ma,&mi2, &ma2, &verbose))
    return NULL;

  npy_intp size = PyArray_SIZE(src_arr);
  
  T * src = (T*) PyArray_DATA(src_arr);
  T * dst = (T*) PyArray_DATA(dst_arr);

  const int32_t dtype_min = std::numeric_limits<T>::min();
  const int32_t dtype_max = std::numeric_limits<T>::max();

  int64_t i;  

  if (verbose){
    printf("Using %d thread(s)\n", omp_get_max_threads());
    fflush(stdout);
  }

#pragma omp parallel for schedule(static)
  for(i=0; i<size; i++) {

    T in = src[i];
    T out = (in-mi)/(ma-mi)*(ma2-mi2)+mi2;
    // // clip  to dtype min/max
    // out = out < dtype_min ? dtype_min : out > dtype_max ? dtype_max : out;
    // dst[i] = (T)out;
    dst[i] = (T)out;

  }
   return Py_None;
}


//------------------------------------------------------------------------

static struct PyMethodDef methods[] = {
                                       {"c_normalize_fast_uint8",
                                        T_normalize_fast<uint8_t>,
                                        METH_VARARGS,
                                        "c_normalize_fast_uint8"},
                                       
                                       {"c_normalize_fast_uint16",
                                        T_normalize_fast<uint16_t>,
                                        METH_VARARGS,
                                        "c_normalize_fast_uint16"},
                                       
                                       {"c_normalize_fast_int8",
                                        T_normalize_fast<int8_t>,
                                        METH_VARARGS,
                                        "c_normalize_fast_int8"},
                                       
                                       {"c_normalize_fast_int16",
                                        T_normalize_fast<int16_t>,
                                        METH_VARARGS,
                                        "c_normalize_fast_int16"},


                                       
                                       {NULL, NULL, 0, NULL}                                       
};

static struct PyModuleDef moduledef = {
                                       PyModuleDef_HEAD_INIT,
                                       "normalize_fast", 
                                       NULL,         
                                       -1,           
                                       methods,
                                       NULL,NULL,NULL,NULL
};

PyMODINIT_FUNC PyInit_normalize_fast2(void) {
  import_array();
  return PyModule_Create(&moduledef);
}
