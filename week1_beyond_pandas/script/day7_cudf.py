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
# `conda install -c nvidia -c rapidsai -c numba -c conda-forge -c defaults cudf`

# ## What is cuDF?
# 
# cuDF is a Python GPU DataFrame library (built on the Apache Arrow columnar memory format) for loading, joining, aggregating, filtering, and otherwise manipulating tabular data using a DataFrame style API.

# input_file = '/data/kaggle/ISTAT_census_variables_2011.csv'


import pandas as pd
import numpy as np
import cudf

pandas_df = pd.DataFrame({'a': np.random.randint(0, 100000000, size=100000000),
                          'b': np.random.randint(0, 100000000, size=100000000)})
                          
cudf_df = cudf.DataFrame.from_pandas(pandas_df)


get_ipython().run_cell_magic('time', '', 'pandas_df.a.mean()')


get_ipython().run_cell_magic('time', '', 'cudf_df.a.mean()')


get_ipython().run_cell_magic('time', '', "pandas_df.merge(pandas_df, on='b')")


get_ipython().run_cell_magic('time', '', "cudf_df.merge(cudf_df, on='b')")








import cudf
gdf = cudf.read_csv(input_file)
for column in gdf.columns:
    print(gdf[column].mean())

# gdf




