#!/usr/bin/env python
# coding: utf-8

# %%bash
# conda install -c conda-forge vaex


# ## References
# 
# - https://vaex.io/docs/tutorial.html

import vaex
df = vaex.example()
# df  # Since this is the last statement in a cell, it will print the DataFrame in a nice HTML format.


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


# Read in the NYC Taxi dataset straight from S3
nyctaxi = vaex.open('s3://vaex/taxi/yellow_taxi_2009_2015_f32.hdf5?anon=true')
nyctaxi.head(5)


# the data from S3 above were cached on disk 

# ## Data Reading

input_file = "/data/raw_data/spectx_cell_mapping_test/cell_mapping_1gb.csv"
df_tmp = vaex.from_csv(input_file, convert=True, copy_index=False)
df = vaex.open(input_file + '.hdf5')
df


# ## Parallel computations
# 
# - Vaex can do computations in parallel similar as in [joblib](https://joblib.readthedocs.io/en/latest/index.html) and `dask`

import vaex
df = vaex.example()
limits = [-10, 10]
delayed_count = df.count(df.E, binby=df.x, limits=limits,
                         shape=4, delay=True)
delayed_count


delayed_sum = df.sum(df.E, binby=df.x, limits=limits,
                         shape=4, delay=True)

@vaex.delayed
def calculate_mean(sums, counts):
    print('calculating mean')
    return sums/counts

print('before calling mean')
# since calculate_mean is decorated with vaex.delayed
# this now also returns a 'delayed' object (a promise)
delayed_mean = calculate_mean(delayed_sum, delayed_count)

# if we'd like to perform operations on that, we can again
# use the same decorator
@vaex.delayed
def print_mean(means):
    print('means', means)
print_mean(delayed_mean)

print('before calling execute')
df.execute()

# Using the .get on the promise will also return the result
# However, this will only work after execute, and may be
# subject to change
means = delayed_mean.get()
print('same means', means)


# ## Machine Learning with vaex.ml

import vaex
vaex.multithreading.thread_count_default = 8
import vaex.ml

import numpy as np
import pylab as plt


df = vaex.ml.datasets.load_iris()
dfdf.scatter(df.petal_length, df.petal_width, c_expr=df.class_);


df.scatter(df.petal_length, df.petal_width, c_expr=df.class_);


features = ['petal_length', 'petal_width', 'sepal_length', 'sepal_width']
scaler = vaex.ml.StandardScaler(features=features, prefix='scaled_')
scaler.fit(df)
df_trans = scaler.transform(df)
df_trans


# ### Encoding of Categorical Features

df =  vaex.ml.datasets.load_titanic()
df.head(5)


label_encoder = vaex.ml.LabelEncoder(features=['embarked'])
one_hot_encoder = vaex.ml.OneHotEncoder(features=['embarked'])
freq_encoder = vaex.ml.FrequencyEncoder(features=['embarked'])
bayes_encoder = vaex.ml.BayesianTargetEncoder(features=['embarked'], target='survived')
woe_encoder = vaex.ml.WeightOfEvidenceEncoder(features=['embarked'], target='survived')

df = label_encoder.fit_transform(df)
df = one_hot_encoder.fit_transform(df)
df = freq_encoder.fit_transform(df)
df = bayes_encoder.fit_transform(df)
df = woe_encoder.fit_transform(df)

df.head(5)


# ### Scikit-Learn example

from vaex.ml.sklearn import Predictor
from sklearn.ensemble import GradientBoostingClassifier

df = vaex.ml.datasets.load_iris()

features = ['petal_length', 'petal_width', 'sepal_length', 'sepal_width']
target = 'class_'

model = GradientBoostingClassifier(random_state=42)
vaex_model = Predictor(features=features, target=target, model=model, prediction_name='prediction')

vaex_model.fit(df=df)

df = vaex_model.transform(df)
df


# ## I/O Kung-Fu
# 
# - Binary file:
#  - HDF5
#  - Arrow
#  - Parquet
#  - FITS
# 
# - Text file:
#  - CSV
#  - ASCII
#  - JSON
#  
# - Cloud support:
#  - Google Cloud Storage
#  - AWS S3
# 
# - Extras:
#  - Aliases

# Convert from pandas dataframe to vaex dataframe
import pandas as pd

pandas_df = pd.read_csv('/data/raw_data/spectx_cell_mapping_test/cell_mapping_1gb.csv')
pandas_df


df = vaex.from_pandas(df=pandas_df, copy_index=True)
df


# Info from above process:
# 
# - pandas still powerful data reader, as it can read data from almost all kind of data sources including databases
# - by reading it first using pandas and then convert it into vaex when it's needed for multi-processing

import vaex
import pandas as pd
import sqlalchemy

connection_string = 'postgresql://enlik:' + 'Zg7vcQ9E1BF394Kg' + '@localhost:5432/spectx_test_ee'
engine = sqlalchemy.create_engine(connection_string)

pandas_df = pd.read_sql_query('select * from ee_telia_cells.cell where geom is not null', con=engine)
df = vaex.from_pandas(pandas_df, copy_index=False)
df


# ### Vaex Extras: using Aliases

vaex.aliases['input_1gb'] = '/data/raw_data/spectx_cell_mapping_test/cell_mapping_1gb.csv'
vaex.aliases['nyc_taxi_aws'] = 's3://vaex/taxi/nyc_taxi_2015_mini.hdf5?anon=true'


df = vaex.open('nyc_taxi_aws')
df.head(5)


df.export_hdf5('/data/raw_data/spectx_cell_mapping_test/test.hdf5')


# ## Test 12GB and 107GB Input Dataset
# 
# - https://vaex.io/docs/datasets.html

import vaex
df = vaex.open('s3://vaex/taxi/yellow_taxi_2009_2015_f32.hdf5?anon=true')


df




