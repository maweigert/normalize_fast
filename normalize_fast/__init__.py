from .normalize_fast import normalize_mi_ma_fast
from .percentile_fast import c_percentile_fast_uint8, c_percentile_fast_uint16, c_percentile_fast_int8, c_percentile_fast_int16


def percentile_fast(x, pmin, pmax):
    import numpy as np
    pmin, pmax = np.float32(pmin),np.float32(pmax)
    if x.dtype.type==np.uint8:
        return c_percentile_fast_uint8(x,pmin, pmax)
    elif x.dtype.type==np.uint16:
        return c_percentile_fast_uint16(x,pmin, pmax)
    elif x.dtype.type==np.int8:
        return c_percentile_fast_int8(x,pmin, pmax)
    elif x.dtype.type==np.int16:
        return c_percentile_fast_int16(x,pmin, pmax)
    else:
        raise ValueError("percentile_fast only implemented for {u}int{8|16} type")



