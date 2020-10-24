#include "Python.h"
#include "math.h"
#include "numpy/ndarraytypes.h"
#include "numpy/ufuncobject.h"
#include "numpy/halffloat.h"
#include <typeinfo>
#include <iostream>
#include "stdio.h"

#ifdef _OPENMP
#include <omp.h>
#endif


template <typename T> static void T_protoype_normalize_mi_ma_fast(char **args,
                                                         npy_intp const *dimensions,
                                                         npy_intp const *steps, void* data){

  npy_intp i;
  npy_intp n = dimensions[0];

  char *in_arr   = args[0];
  char *mi_arr   = args[1];
  char *ma_arr   = args[2];
  char *mi2_arr  = args[3];
  char *ma2_arr  = args[4];
  char *out_arr  = args[5];

  npy_intp in_step   = steps[0];
  npy_intp mi_step   = steps[1];
  npy_intp ma_step   = steps[2];
  npy_intp mi2_step  = steps[3];
  npy_intp ma2_step  = steps[4];
  npy_intp out_step  = steps[5];   

  if (mi_step == 0 && ma_step == 0 && mi2_step == 0 && ma2_step == 0) {

    T mi, ma, mi2, ma2;

    mi  = *(T *)mi_arr;
    ma  = *(T *)ma_arr;
    mi2 = *(T *)mi2_arr;
    ma2 = *(T *)ma2_arr;
           
# pragma omp parallel for 
    for (i = 0; i < n; i++) {

      T in;
      float out;
      in  = *(T *)(&in_arr[i*in_step]);
      
      out = ((float)in-mi)/(ma-mi)*(ma2-mi2)+mi2;
      out = out <= mi2 ? mi2 : out >= ma2 ? ma2 : out;
      
#pragma omp atomic write
      *((T *)(&out_arr[i*out_step])) = (T) out;

    }
  }
  else{
# pragma omp parallel for 
    for (i = 0; i < n; i++) {

      T in, mi, ma, mi2, ma2;
      float out;
      
      in  = *(T *)(&in_arr[i*in_step]);
      mi  = *(T *)(&mi_arr[i*mi_step]);
      ma  = *(T *)(&ma_arr[i*ma_step]);
      mi2 = *(T *)(&mi2_arr[i*mi2_step]);
      ma2 = *(T *)(&ma2_arr[i*ma2_step]);
        
      out = ((float)in-mi)/(ma-mi)*(ma2-mi2)+mi2;
      out = out <= mi2 ? mi2 : out >= ma2 ? ma2 : out;

#pragma omp atomic write      
      *((T *)out_arr) = (T) out;
              
    }
  }    
}

// we need a non-const and a cont version as the signature was changed for recent numpy version
template <typename T> static void T_normalize_mi_ma_fast(char **args,
                                                         npy_intp  *dimensions,
                                                         npy_intp  *steps, void* data){

  npy_intp const *_dimensions = dimensions;
  npy_intp const *_steps = steps;
  return T_protoype_normalize_mi_ma_fast<T>(args, _dimensions, _steps, data);
}

template <typename T> static void T_normalize_mi_ma_fast(char **args,
                                                         npy_intp const  *dimensions,
                                                         npy_intp const *steps, void* data){
  return T_protoype_normalize_mi_ma_fast<T>(args, dimensions, steps, data);
}


#define NFUNCS 8

static char types[6*NFUNCS] = {
  NPY_UINT8, NPY_UINT8, NPY_UINT8, NPY_UINT8,NPY_UINT8, NPY_UINT8,
  NPY_INT8, NPY_INT8, NPY_INT8, NPY_INT8,NPY_INT8, NPY_INT8,

  NPY_UINT16, NPY_UINT16, NPY_UINT16, NPY_UINT16,NPY_UINT16, NPY_UINT16,
  NPY_INT16, NPY_INT16, NPY_INT16, NPY_INT16,NPY_INT16, NPY_INT16,

  NPY_UINT32, NPY_UINT32, NPY_UINT32, NPY_UINT32,NPY_UINT32, NPY_UINT32,
  NPY_INT32, NPY_INT32, NPY_INT32, NPY_INT32,NPY_INT32, NPY_INT32,

  NPY_FLOAT, NPY_FLOAT, NPY_FLOAT, NPY_FLOAT,NPY_FLOAT, NPY_FLOAT,
  NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE,NPY_DOUBLE, NPY_DOUBLE,

};

/*This gives pointers to the above functions*/
PyUFuncGenericFunction funcs[NFUNCS] = {
  &(T_normalize_mi_ma_fast<uint8_t>),
  &(T_normalize_mi_ma_fast<int8_t>),

  &(T_normalize_mi_ma_fast<uint16_t>),
  &(T_normalize_mi_ma_fast<int16_t>),

  &(T_normalize_mi_ma_fast<uint32_t>),
  &(T_normalize_mi_ma_fast<int32_t>),
  
  &(T_normalize_mi_ma_fast<float>),
  &(T_normalize_mi_ma_fast<double>)
};



static void *data[NFUNCS] = {NULL, NULL};

static PyMethodDef Methods[] = {
        {NULL, NULL, 0, NULL}
};


static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "normalize_fast",
    NULL,
    -1,
    Methods,
    NULL,
    NULL,
    NULL,
    NULL
};

PyMODINIT_FUNC PyInit_normalize_fast(void)
{
  PyObject *m, *func, *d;
  m = PyModule_Create(&moduledef);
  if (!m) {
    return NULL;
  }

  import_array();
  import_umath();

  
  func = PyUFunc_FromFuncAndData(funcs, data, types, NFUNCS, 5, 1,
                                  PyUFunc_None,
                                 "normalize_fast",
                                 "normalize_fast_mi_ma(x, mi, ma, mi2, ma2): affinely scales x such that the range (mi,ma) mappes to (mi2, ma2)", 0);

  d = PyModule_GetDict(m);

  PyDict_SetItemString(d, "normalize_mi_ma_fast", func);
  Py_DECREF(func);

  return m;
}
