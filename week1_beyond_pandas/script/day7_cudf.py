#!/usr/bin/env python
# coding: utf-8

# ## References
# 
# - https://rapids.ai/start.html
# - https://github.com/rapidsai/cudf
# - https://hub.docker.com/r/rapidsai/rapidsai/
# - https://docs.rapids.ai/api/cudf/stable/10min.html
# - https://github.com/beckernick/nersc-rapids-workshop
# - https://towardsdatascience.com/heres-how-you-can-speedup-pandas-with-cudf-and-gpus-9ddc1716d5f2
# 
# ### Videos
# 
# - [cuDF: RAPIDS GPU-Accelerated Dataframe Library" - Mark Harris (PyCon AU 2019)](https://www.youtube.com/watch?reload=9&v=lV7rtDW94do)
# - [Introduction to cuDF - NERSC NVIDIA RAPIDS Workshop on April 14, 2020](https://www.youtube.com/watch?v=pXnEniQRAdQ)
# 
# Notes:
# 
# - It requires NVidia GPU
# 
# Prerequisites
# 
#     - NVIDIA Pascal™ GPU architecture or better
#     - CUDA 10.1/10.2/11.0 with a compatible NVIDIA driver
#     - Ubuntu 16.04/18.04 or CentOS 7
#     - Docker CE v18+
#     - nvidia-docker v2+
#     
# Installation
# 
# `
# conda install -c nvidia -c rapidsai -c numba -c conda-forge -c defaults cudf
# `

# ## Misc
# 
# check computer GPU
# `
# sudo lshw -numeric -C display
# `
# 
# `
# sudo lspci -v | less
# `
# 
# Source:
# https://www.howtogeek.com/508993/how-to-check-which-gpu-is-installed-on-linux/
# 
# Check Cuda Version
# https://varhowto.com/check-cuda-version/
# https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html
# 
# Cuda Guide:
# https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1804&target_type=deblocal
# 
# Troubleshooting:
# https://stackoverflow.com/a/64593288/2670476

# ## What is cuDF?
# 
# cuDF is a Python GPU DataFrame library (built on the Apache Arrow columnar memory format) for loading, joining, aggregating, filtering, and otherwise manipulating tabular data using a DataFrame style API.

input_file = 'data/data_1gb.csv'


import pandas as pd
import numpy as np
import cudf

pandas_df = pd.DataFrame({'a': np.random.randint(0, 100000000, size=100000000),
                          'b': np.random.randint(0, 100000000, size=100000000)})
                          
cudf_df = cudf.DataFrame.from_pandas(pandas_df)


pandas_df.head()


cudf_df.head()


get_ipython().run_cell_magic('time', '', 'pandas_df.a.mean()')


get_ipython().run_cell_magic('time', '', 'cudf_df.a.mean()')


get_ipython().run_cell_magic('time', '', "pandas_df.merge(pandas_df, on='b')")


get_ipython().run_cell_magic('time', '', "cudf_df.merge(cudf_df, on='b')")


import cudf
gdf = cudf.read_csv(input_file)
for column in gdf.columns:
    print(gdf[column].mean())

# gdf


import cudf, io, requests
from io import StringIO

url = "https://github.com/plotly/datasets/raw/master/tips.csv"
content = requests.get(url).content.decode('utf-8')

tips_df = cudf.read_csv(StringIO(content))
tips_df['tip_percentage'] = tips_df['tip'] / tips_df['total_bill'] * 100

# display average tip by dining party size
print(tips_df.groupby('size').tip_percentage.mean())


# ## Dask-CUDA
# 
# https://rapids.ai/start.html#get-rapids
# 
# Updated using:
# 
# `conda create -n rapids-0.17 -c rapidsai -c nvidia -c conda-forge \
#     -c defaults rapids-blazing=0.17 python=3.7 cudatoolkit=11.0
# `
# 
# `
# conda activate rapids-0.17
# `

# source: https://github.com/rapidsai/cudf/issues/2288

import os
import gc
import timeit
import cudf as cu
import dask_cudf as dkcu
# x = cu.read_csv("data/data_1gb.csv", flush=True)
x = dkcu.read_csv("data/data_1gb.csv", flush=True)
x


import time

from dask.distributed import Client, wait
from dask_cuda import LocalCUDACluster

cluster = LocalCUDACluster()
client = Client(cluster)
client




