# emeans

Fast [e-means](https://github.com/facebookresearch/faiss/tree/main/contrib/e_means) clustering, written in Rust. e-means is a k-means replacement from Meta FAIR that reaches a lower error in about a tenth of the passes.

```
pip install emeans
```

Same class as faiss `NumpyEmeans`, so most code only needs a new import:

```python
import numpy as np
from emeans import Emeans

x = np.random.rand(200_000, 64).astype("float32")
em = Emeans(64, 500)
centroids, labels = em.train(x)
distances, nearest = em.assign(x)
```

Needs only numpy. 1.7x to 6x faster than faiss `NumpyEmeans` in our tests, same error. Numbers and source: https://github.com/savannahar68/emeans
