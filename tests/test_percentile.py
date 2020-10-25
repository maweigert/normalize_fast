import numpy as np
np.random.seed(42)
from timeit import timeit 
from normalize_fast import percentile_fast 

def test_speed():

    setup = """
from __main__ import np, percentile_fast
x = np.tile(np.random.randint(0,100,(100,100)),(100,100)).astype(np.uint16);
pmin, pmax = np.float32(10), np.float32(90)
print(f'percentile of array with shape {x.shape}')
    """

    niter = 6
    
    t1 = timeit("y = np.percentile(x, (pmin, pmax))", setup = setup, number=niter)
    t2 = timeit("y = percentile_fast(x, pmin, pmax)", setup = setup, number=niter)

    print("\n\n")
    print(f"numpy:   {1000*t1/niter:.2f} ms")
    print(f"openmp:  {1000*t2/niter:.2f} ms")

    
def test_accuracy():
    x = np.random.randint(0,101,100)

    for dtype in (np.uint8, np.uint16, np.int8, np.int16):
        print(dtype)
        pmin, pmax = np.random.uniform(0,100,2).astype(np.float32)
        p1 = tuple(np.percentile(x.astype(dtype),(pmin, pmax)).astype(int))
        p2 = tuple(percentile_fast(x.astype(dtype),pmin, pmax))
        print(p1)
        print(p2)
    

    
if __name__ == '__main__':

    test_speed()
    # test_accuracy()
