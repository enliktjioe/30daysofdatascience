#!/usr/bin/env python
# coding: utf-8

# ## References
# 
# - https://towardsdatascience.com/an-overview-of-pythons-datatable-package-5d3a97394ee9
# - https://datatable.readthedocs.io/en/latest/start/quick-start.html

# Importing necessary Librariesimport numpy as np
import pandas as pd
import datatable as dt
print(dt.__version__)


get_ipython().run_cell_magic('time', '', "datatable_df = dt.fread('/data/kaggle_mobile_phone_activity_in_city/sms-call-internet-mi-2013-11-01.csv') #80mb data")


get_ipython().run_cell_magic('time', '', "pandas_df = pd.read_csv('/data/kaggle_mobile_phone_activity_in_city/sms-call-internet-mi-2013-11-01.csv') #80mb data")


# ## Datable to Numpy / Pandas

numpy_df = datatable_df.to_numpy()
pandas_df = datatable_df.to_pandas()


print(datatable_df)


pandas_df


# ## Calculate mean

get_ipython().run_cell_magic('time', '', 'datatable_df.mean() #80mb dataset')


# %%time
# pandas_df.mean() #80mb dataset
# # never ending


# ## Sorting the Frame

get_ipython().run_cell_magic('time', '', "datatable_df.sort('CellID')")


get_ipython().run_cell_magic('time', '', "pandas_df.sort_values(by='CellID')")


# ## Deleting Rows/Columns

del datatable_df[:, 'internet']


datatable_df


# ## GroupBy

get_ipython().run_cell_magic('time', '', 'for i in range(5):\n    datatable_df[:, dt.sum(dt.f.countrycode), dt.by(dt.f.CellID)]')


get_ipython().run_cell_magic('time', '', 'for i in range(5):\n    pandas_df.groupby("CellID")["countrycode"].sum()')


# **What does .f stand for?**
# 
# f stands for frame proxy, and provides a simple way to refer to the Frame that we are currently operating upon. In the case of our example, dt.f simply stands for dt_df.

# ## Filtering Rows

datatable_df[dt.f.CellID>dt.f.countrycode,"CellID"]


# ## Saving to CSV

datatable_df.to_csv('/data/output/datatable_output.csv')


# ## Summary
# 
# - datatable works really well for big data size, compared to pandas
# - in terms of functionality, pandas still won
