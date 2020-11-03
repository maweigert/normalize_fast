#include <Python.h>
#include "numpy/arrayobject.h"
#include "numpy/npy_math.h"
#include <cstdio>
#include <limits>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

template <typename T> static PyObject* T_percentile_fast(PyObject *self, PyObject *args) {

  PyArrayObject *src_arr = NULL;

  float pmin, pmax;
  int verbose;
  
  if (!PyArg_ParseTuple(args, "O!ffi",
                        &PyArray_Type, &src_arr,
                        &pmin, &pmax, &verbose))
    return NULL;

  npy_intp size = PyArray_SIZE(src_arr);
  
  T * src = (T*) PyArray_DATA(src_arr);

  const int32_t dtype_min = std::numeric_limits<T>::min();
  const int32_t dtype_max = std::numeric_limits<T>::max();

  const int32_t bins = dtype_max-dtype_min+1;

  int64_t i;  

  if (verbose){
    printf("Using %d thread(s)\n", omp_get_max_threads());
    fflush(stdout);
  }
  
  int64_t counts[bins] = {0};
#pragma omp parallel for reduction(+:counts)
  for(i=0; i<size; i++) {
    counts[src[i]-dtype_min]++;
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


template <class T> static PyObject* T_percentile_fast2(PyObject *self, PyObject *args) {

  
  PyArrayObject *src = NULL;
  PyArrayObject *quants = NULL;
  int64_t i;  

  int verbose;
  
  if (!PyArg_ParseTuple(args, "O!O!i",
                        &PyArray_Type, &src,
                        &PyArray_Type, &quants,
                        &verbose))
    return NULL;

  npy_intp *dims_src = PyArray_DIMS(src);
  npy_intp n_quants = PyArray_SIZE(quants);

  const npy_intp n_slow = dims_src[0];
  const npy_intp n_fast = dims_src[1];
  npy_intp dims_dst[2] = {n_quants, n_slow};

  PyArrayObject * dst = (PyArrayObject*)PyArray_ZEROS(2,dims_dst,NPY_INT32,0);
  
  if (verbose){
    printf("Using %d thread(s)\n", omp_get_max_threads());
    fflush(stdout);
  }
  
  const int32_t dtype_min = std::numeric_limits<T>::min();
  const int32_t dtype_max = std::numeric_limits<T>::max();
  const int32_t n_bins = dtype_max-dtype_min+1;
  

  int64_t counts[n_bins];
  
  for (int64_t i = 0; i < n_slow; ++i) {


    std::fill(counts, counts+n_bins, 0);
    
#pragma omp parallel for reduction(+:counts)
    for(int64_t j=0; j<n_fast; j++) {
      counts[*(T *)PyArray_GETPTR2(src,i,j)-dtype_min]++;
    }


    for(int64_t j=0; j<n_quants; j++) {
      float val = floor(0.01*n_fast*(*(float *)PyArray_GETPTR1(quants,j)));
    
      int64_t sum = 0;
      
      for (int64_t k =0; k<n_bins; k++) {
        sum += counts[k];
        if ((sum>val) || ((val>0) && (sum>=val))){
          *(T *)PyArray_GETPTR2(dst,j,i) = (T)(k+dtype_min);
          
          break;
        }
      }
      
    }  
  }
  
  return PyArray_Return(dst);
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

  //-----------------

  {"c_percentile_fast2_uint8",
   T_percentile_fast2<uint8_t>,
   METH_VARARGS,
   "c_percentile_fast2_uint8"},
                                       
  {"c_percentile_fast2_uint16",
   T_percentile_fast2<uint16_t>,
   METH_VARARGS,
   "c_percentile_fast2_uint16"},
                                                                              
  {"c_percentile_fast2_int8",
   T_percentile_fast2<int8_t>,
   METH_VARARGS,
   "c_percentile_fast2_int8"},
                                       
  {"c_percentile_fast2_int16",
   T_percentile_fast2<int16_t>,
   METH_VARARGS,
   "c_percentile_fast2_int16"},
  
  
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
