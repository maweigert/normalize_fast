# normalize_fast

## Examples 

```python 

import numpy as np
from csbdeep.utils import normalize_mi_ma
from normalize_fast import normalize_mi_ma_fast


x0 = np.tile((np.random.randn(2000,2000)).astype(np.uint16), (4,4))    

x = x0.copy() 
%timeit a = 255*normalize_mi_ma(x,0,100)


x = x0.copy() 
%timeit normalize_mi_ma_fast(x,0,100,0,255, out=x)





```
