#!/usr/bin/env python
# coding: utf-8

# %%bash
# conda install -c conda-forge vaex


import vaex
df = vaex.example()
df  # Since this is the last statement in a cell, it will print the DataFrame in a nice HTML format.


df.x


import numpy as np
import matplotlib.pylab as plt

counts_x = df.count(binby=df.x, limits=[-10, 10], shape=64)
plt.plot(np.linspace(-10, 10, 64), counts_x)
plt.show()


import vaex
import numpy as np
x = np.arange(5)
y = x**2
df = vaex.from_arrays(x=x, y=y)
df




