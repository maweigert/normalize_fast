import numpy as np
import pytest
import numba
from timeit import timeit 
import numexpr
from csbdeep.utils import normalize_mi_ma
from normalize_fast import normalize_mi_ma_fast

@numba.vectorize([numba.uint8(numba.uint8, numba.uint8, numba.uint8, numba.uint8, numba.uint8)])
def normalize_mi_ma_numba(x,mi,ma, mi2, ma2):
    return mi2+(ma2-mi2)*(x-mi)/(ma-mi)

def normalize_mi_ma_inplace(x, mi, ma, mi_dst=0, ma_dst=255, mode = "numexpr"):
    """
    """    
    dtype = x.dtype.type
    mi, ma, mi_dst, ma_dst = map(dtype,(mi, ma, mi_dst, ma_dst))
    factor = dtype((ma_dst-mi_dst + 1.e-20) / ( ma - mi + 1e-20))

    dtype_min = np.iinfo(x.dtype).min
    dtype_max = np.iinfo(x.dtype).max

    if mode == "numpy":
        x -= mi
        x *= factor
        x += mi_dst
    elif mode == "numexpr":
        numexpr.evaluate('where((x-mi)<dtype_min, dtype_min, x-mi)' , out=x, casting='unsafe')
        numexpr.evaluate('x*factor', out=x, casting='unsafe')
        numexpr.evaluate('where((x+mi_dst)>dtype_max, dtype_max, x+mi_dst)', out=x, casting='unsafe')
    else:
        raise ValueError(f"Unknown mode {mode}")
    
    return x



def create_example(dtype, mi ,ma, mi2, ma2):
    np.random.seed(42)
    x = np.random.randint(0,100,(53,97)).astype(dtype)
    y1 = (mi2+(ma2-mi2)*normalize_mi_ma(x,mi,ma)).astype(dtype)
    y2 = normalize_mi_ma_fast(x,mi,ma, mi2, ma2)
    return x, y1, y2

def test_accuracy():
    def _check(dtype, mi ,ma, mi2, ma2):
        x, y1, y2 = create_example(dtype, mi ,ma, mi2, ma2)
        status = np.allclose(y1, y2)
        print(f"{dtype}, {mi}, {ma}, {mi2}, {ma2} -> {status}")
        
        assert status
        return x, y1, y2 

    _check(np.uint8, 0, 100, 0, 255)
    _check(np.uint8, 10, 90, 0, 255)


def test_speed():

    setup = """
from __main__ import np, normalize_mi_ma, normalize_mi_ma_fast, normalize_mi_ma_numba, normalize_mi_ma_inplace
x = np.tile(np.random.randint(0,100,(100,100)),(50,50)).astype(np.uint8);
mi, ma, mi2, ma2 = 0, 100, 0,255
print(f'normalize_mi_ma of array with shape {x.shape}')
    """

    niter = 20
    
    t1 = timeit("y = mi2+(ma2-mi2)*normalize_mi_ma(x,mi,ma)", setup = setup, number=niter)
    t2 = timeit("y = normalize_mi_ma_inplace(x,mi,ma, mi2, ma2, mode= 'numpy')", setup = setup, number=niter)
    t3 = timeit("y = normalize_mi_ma_inplace(x,mi,ma, mi2, ma2, mode= 'numexpr')", setup = setup, number=niter)
    t4 = timeit("y = normalize_mi_ma_numba(x,mi,ma, mi2, ma2)", setup = setup, number=niter)
    t5 = timeit("y = normalize_mi_ma_fast(x,mi,ma, mi2, ma2)", setup = setup, number=niter)
    t6 = timeit("normalize_mi_ma_fast(x,mi,ma, mi2, ma2, out = x)", setup = setup, number=niter)

    print("\n\n")
    print(f"csdbeep:            {1000*t1/niter:.2f} ms")
    print(f"numpy (inplace):    {1000*t2/niter:.2f} ms")
    print(f"numexpr (inplace):  {1000*t3/niter:.2f} ms")
    print(f"numba:              {1000*t4/niter:.2f} ms")
    print(f"openmp:             {1000*t5/niter:.2f} ms")
    print(f"openmp (inplace):   {1000*t6/niter:.2f} ms")
    
if __name__ == '__main__':

    test_speed()

    # test_accuracy()


    # x, y1, y2 = create_example(np.uint8, 10, 90, 0, 255)
