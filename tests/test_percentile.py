import numpy as np
np.random.seed(42)
from timeit import timeit 
from normalize_fast import percentile_fast, percentile_fast2 
from types import SimpleNamespace
from itertools import chain, combinations

def test_speed(n_tiles=30):
    base_size = 100
    setup = f"""
from __main__ import np, percentile_fast
x = np.tile(np.random.randint(0,100,({base_size},{base_size})),({n_tiles},{n_tiles})).astype(np.uint16);
pmin, pmax = np.float32(10), np.float32(90)
print('percentile of array with shape', x.shape)
    """

    niter = 6
    
    t1 = timeit("y = np.percentile(x, (pmin, pmax))", setup = setup, number=niter)
    t2 = timeit("y = percentile_fast(x, pmin, pmax, verbose=1)", setup = setup, number=niter)

    print("\n\n")
    print(f"numpy:   {1000*t1/niter:.2f} ms")
    print(f"openmp:  {1000*t2/niter:.2f} ms")

    
    return SimpleNamespace(t1=t1/niter, t2=t2/niter, size=base_size**2*n_tiles**2)
    
    
def test_accuracy():
    x = np.random.randint(0,101,100)

    for dtype in (np.uint8, np.uint16, np.int8, np.int16):
        print(dtype)
        pmin, pmax = np.random.uniform(0,100,2).astype(np.float32)
        p1 = tuple(np.percentile(x.astype(dtype),(pmin, pmax)).astype(int))
        p2 = tuple(percentile_fast(x.astype(dtype),pmin, pmax))
        print(p1)
        print(p2)
    

def benchmark(n_core_max = 16, n_tiles_max=100):
    import subprocess
    import re
    import matplotlib.pyplot as plt 

    def _get(n_cores, n_tiles):
        print(n_cores, n_tiles)
        output = subprocess.check_output([
            "python","-c",
            f"""import os 
os.environ['OMP_NUM_THREADS']='{n_cores}'
from test_percentile import *;
print(test_speed({n_tiles}));
        """])
        r = re.findall("namespace\((.*?)\)",str(output))[0]
        d = eval(f"dict({r})")
        return d["size"], d["t1"], d["t2"]

    n_cores = np.linspace(1,n_core_max,4).astype(int)
    n_tiles =  np.linspace(1,n_tiles_max,10).astype(int)

    fig = plt.figure()
    fig.clf()
    ax = fig.subplots(1,1)
    for i,c in enumerate(n_cores):
        size, t1, t2 = tuple(zip(*tuple(_get(c,n) for n in n_tiles)))
        col = next(ax._get_lines.prop_cycler)['color']
        if i==0: 
            ax.loglog(size,t1, label=f"numpy  ({c} cores)",color = col, ls = "--")
        ax.loglog(size,t2, label=f"openmp ({c} cores)", color = col, ls= "-")

    ax.set_xlabel("size (pixels)")
    ax.set_ylabel("time s)")
    fig.suptitle("Percentile")
    ax.legend()
    plt.show()

    
def test_accuracy2():

    def _powerset(dims):
        return chain.from_iterable(combinations(dims, r) for r in range(1,len(dims)+1))

    for dtype in (np.uint8, np.uint16, np.int8, np.int16):
        for ndim in range(1,5):
            shape = np.random.randint(31,51,ndim)
            x = np.random.randint(0,126,shape).astype(dtype)
            q = np.random.randint(0,100,7)
            all_axis = tuple(range(ndim))+ tuple(range(-ndim,0))
            for axis in _powerset(all_axis):
                print(x.shape, q.shape, axis)
                out1 = np.percentile(x,q,axis = axis)
                out2 = percentile_fast2(x,q,axis = axis)
                return x,q,axis
                print(x)
                print(out1)
                print(out2)
                assert np.allclose(out1, out2)

    
if __name__ == '__main__':

    np.random.seed(42)
    x = np.random.randint(0,100,(5,2,3)).astype(np.uint16)
    q = (0,100)
    axis = (0,1)
    # print(np.percentile(x, q, axis = axis).astype(int))
    # # print(percentile_fast(x.ravel(), 0,100))
    # print(percentile_fast2(x, q, axis = axis))
    
    # test_speed()
    # test_accuracy()

    # benchmark(16,100)

    test_accuracy2()
