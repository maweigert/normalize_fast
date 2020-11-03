import numpy as np

from .normalize_fast import normalize_mi_ma_fast
from .normalize_fast2 import c_normalize_fast_uint8, c_normalize_fast_uint16, c_normalize_fast_int8, c_normalize_fast_int16


from .percentile_fast import c_percentile_fast_uint8, c_percentile_fast_uint16, c_percentile_fast_int8, c_percentile_fast_int16

from .percentile_fast import c_percentile_fast2_uint8, c_percentile_fast2_uint16, c_percentile_fast2_int8, c_percentile_fast2_int16


def normalize_fast(x, mi, ma, mi2, ma2, verbose=False, out = None):
    mi, ma = np.float32(mi),np.float32(ma)
    mi2, ma2 = np.float32(mi2),np.float32(ma2)
    if out is None:
        out = np.empty_like(x)
    else:
        assert x.dtype.type == out.dtype.type
    if x.dtype.type==np.uint8:
        c_normalize_fast_uint8( x, out, mi, ma, mi2, ma2, np.int32(verbose))
    elif x.dtype.type==np.uint16:
        c_normalize_fast_uint16(x, out, mi, ma, mi2, ma2, np.int32(verbose))
    elif x.dtype.type==np.int8:
        c_normalize_fast_int8(  x, out, mi, ma, mi2, ma2, np.int32(verbose))
    elif x.dtype.type==np.int16:
        c_normalize_fast_int16( x, out, mi, ma, mi2, ma2, np.int32(verbose))
    else:
        raise ValueError("percentile_fast only implemented for {u}int{8|16} type")
    return out
    


def percentile_fast(x, pmin, pmax, verbose=False):
    import numpy as np
    pmin, pmax = np.float32(pmin),np.float32(pmax)
    if x.dtype.type==np.uint8:
        return c_percentile_fast_uint8(x,pmin, pmax, np.int32(verbose))
    elif x.dtype.type==np.uint16:
        return c_percentile_fast_uint16(x,pmin, pmax, np.int32(verbose))
    elif x.dtype.type==np.int8:
        return c_percentile_fast_int8(x,pmin, pmax, np.int32(verbose))
    elif x.dtype.type==np.int16:
        return c_percentile_fast_int16(x,pmin, pmax, np.int32(verbose))
    else:
        raise ValueError("percentile_fast only implemented for {u}int{8|16} type")


def percentile_fast2(x, q, axis=None, out=None, verbose=True):
    x = np.asarray(x)
    q = np.asarray(q, dtype = np.float32)

    supported_dtypes = {
        np.uint8:c_percentile_fast2_uint8,
        np.uint16:c_percentile_fast2_uint16,
        np.int8:c_percentile_fast2_int8,
        np.int16:c_percentile_fast2_int16,
    }

    if not x.dtype.type in supported_dtypes:
        raise NotImplementedError(f"unsupported dtype {x.dtype.type}")
    else:
        _percentile_func = supported_dtypes[x.dtype.type]

    if q.ndim>1:
        raise ValueError("quantiles should be 1-dimensional")
                                            
    # normalize axis e.g. correctly mapping axis = -1
    full_axis = tuple(range(x.ndim))
    if axis is None:
        axis = full_axis
    else:
        if np.isscalar(axis):
            axis = (axis,)
        axis = tuple(set(axis))
        axis = tuple((axis,)) if np.isscalar(axis) else tuple(axis)
        axis = tuple(full_axis[a] for a in axis)
        axis = tuple(set(axis))

    remaining_axis = tuple(a for a in full_axis if not a in axis)
    
    shape = np.array(x.shape)
    n_slow = int(np.prod(tuple(x.shape[a] for a in remaining_axis)))
    n_fast = int(np.prod(tuple(x.shape[a] for a in axis)))
    x = x.transpose(remaining_axis+axis)
    x = x.reshape((n_slow, n_fast))

    res = _percentile_func(x, q, np.int32(verbose))
    res = res.reshape((len(q),)+(tuple(shape[np.array(remaining_axis)]) if len(remaining_axis)>0 else ()))
    return res

