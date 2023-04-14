#  Copyright (c) 2022 Robert Lieck.

"""
Usage Examples
===========================

This shows some common use cases of the :class:`~triangularmap.tmap.TMap` class.
"""

# %%
# Typical Use Cases
# -----------------
# We use a Numpy array as underlying storage because it allows for slicing and advanced indexing, which is used to
# provide some of the functionality.

import numpy as np
from triangularmap import TMap

n = 4
arr = np.array([" "] * TMap.size_from_n(n))
tmap = TMap(arr)

# %%
# Of course, we can set single elements:

tmap[1, 4] = 'o'
print(tmap.pretty(haxis=True))

# %%
# Slicing
# -------
# Horizontal Slices
# .................
# We can also set entire rows, specified by depth

tmap.dslice[2] = 'o'
print(tmap.pretty(daxis=True))

# %%
# or level

tmap.lslice[3] = 'x'
print(tmap.pretty(laxis=True))

# %%
# This syntax is required because the :meth:`~triangularmap.tmap.TMap.dslice` and
# :meth:`~triangularmap.tmap.TMap.lslice` methods return a (sliced) `view` of the underlying numpy array, which allows
# for directly assigning values. We can also slice the returned array again

tmap.lslice[1][1:3] = 'x'
print(tmap.pretty(laxis=True))

# %%
# Let's reset the underlying array

tmap.arr[:] = " "
print(tmap.pretty())

# %%
# Vertical Slices
# .................
# We can also use vertical slices defined by a start index

tmap.sslice[1] = 'o'
print(tmap.pretty(haxis=True))

# %%
# or an end index

tmap.eslice[3] = 'x'
print(tmap.pretty(haxis=True))

# %%
# In contrast to the :meth:`~triangularmap.tmap.TMap.dslice` and :meth:`~triangularmap.tmap.TMap.lslice` method, we
# cannot directly slice the underlying array, because it is not aligned with slicing direction. Instead, behind the
# scenes, advanced indexing is used to get and set the elements. For simple getting or setting, this does not make
# a big difference. But it can lead to subtle bugs when first getting a slice and then trying to set elements, because
# the :attr:`~triangularmap.tmap.TMap.sslice` and :attr:`~triangularmap.tmap.TMap.eslice` attributes effectively
# correspond to `copies` not `views` of the underlying array. To demonstrate this, let's first fill the map with numbers

tmap.arr[:] = list(range(len(tmap.arr)))  # implicitly converted to strings by Numpy
print(tmap.pretty(haxis=True, laxis=True))

# %%
# When we now do the following
lslice = tmap.lslice[2]
sslice = tmap.sslice[1]
print(lslice)
print(sslice)

# %%
# we get the corresponding slices, as expected. However, when we try to set elements via these objects

lslice[:] = "X"
sslice[:] = "O"
print(lslice)
print(sslice)
print(tmap.pretty(haxis=True, laxis=True))

# %%
# we see that the original map is only affected for the `lslice` object, because it is a view, while the `sslice` object
# is a copy of the underlying storage (likewise for `dslice` and `eslice`).