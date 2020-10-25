# normalize_fast

## Install

``` 
git clone git@github.com:maweigert/normalize_fast.git
CC=gcc-7 CXX=g++-7 pip install normalize_fast/
```


## Examples 


### Normalize

```python 

import numpy as np
from csbdeep.utils import normalize_mi_ma
from normalize_fast import normalize_mi_ma_fast


x0 = np.tile((np.random.randn(2000,2000)).astype(np.uint16), (4,4))    

x = x0.copy() 
%timeit 255*normalize_mi_ma(x,0,100)


x = x0.copy() 
%timeit normalize_mi_ma_fast(x,0,100,0,255, out=x)


### Percentile

import numpy as np
from normalize_fast import percentile_fast


x0 = np.tile((np.random.randn(2000,2000)).astype(np.uint16), (4,4))    

x = x0.copy() 
%timeit np.percentile(x,(10,90))


x = x0.copy() 
%timeit percentile_fast(x,10,90)



```
