import numpy as np
import ctypes

def array_to_ctypes_1d(A):
    # initialize properly sized array (to 0)
    arr=(ctypes.c_double*len(A))()
    for i in range(len(A)):
        arr[i]=A[i]
    return arr

def ctypes_to_array_1d(A,N):
    arr=np.zeros((N))
    for i in range(N):
        arr[i]=A[i]
    return arr

class c_spline(ctypes.Structure):
    _fields_=[("n",ctypes.c_size_t),
              ("x",ctypes.POINTER(ctypes.c_double)),
              ("y",ctypes.POINTER(ctypes.c_double)),
              ("c",ctypes.POINTER(ctypes.c_double))]

def spline_eval(psinspline,mhatspline,num_cer_points,NFIT=121):
    clib=ctypes.cdll.LoadLibrary('./libspline.so')
    clib.spline_eval.restype=ctypes.c_int
    clib.spline_eval.argtypes=[ctypes.c_size_t, #N
                               (ctypes.c_double*NFIT), #values[N]
                               (ctypes.c_double*NFIT), #eval_pts[N]
                               c_spline, #spl
                               (ctypes.c_double*(3*num_cer_points))] #scratch[3*spl.n]
    work=(ctypes.c_double*num_cer_points)()
    dummy=(ctypes.c_double*(3*num_cer_points))()
    v=(ctypes.c_double*NFIT)()

    psinspline=array_to_ctypes_1d(psinspline)
    mhatspline=array_to_ctypes_1d(mhatspline)

    eval_arr=np.linspace(0,1.2,NFIT) #NOTE: bug in PCS, 0.01 * np.arange(NFIT) dumb
    eval_arr=array_to_ctypes_1d(eval_arr)

    s=c_spline(n=num_cer_points, #ftsceri
               x=psinspline, #ftsspsin[1-76]
               y=mhatspline, #ftssmhat[1-76]
               c=work
               )
    clib.spline_eval(NFIT, v, eval_arr, s, dummy)
    return ctypes_to_array_1d(v, NFIT)
