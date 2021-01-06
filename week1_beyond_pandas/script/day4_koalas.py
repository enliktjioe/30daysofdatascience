#!/usr/bin/env python
# coding: utf-8

# ## References:
# 
# - https://koalas.readthedocs.io/en/latest/
# - https://koalas.readthedocs.io/en/latest/getting_started/videos_blogs.html#data-ai-summit-2020-europe-nov-18-19-2020
# - https://databricks.com/blog/2020/03/31/10-minutes-from-pandas-to-koalas-on-apache-spark.html
# - https://databricks.com/blog/2020/08/11/interoperability-between-koalas-and-apache-spark.html

# ## From Pandas to Koalas

import pandas as pd
import numpy as np
import databricks.koalas as ks
from pyspark.sql import SparkSession


dates = pd.date_range('20130101', periods=6)
dates


pdf = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=list('ABCD'))
pdf


kdf = ks.from_pandas(pdf)
type(kdf)


# ## From Spark DataFrame to Koalas DataFrame

spark = SparkSession.builder.getOrCreate()
sdf = spark.createDataFrame(pdf)
sdf.show()


kdf = sdf.to_koalas()
kdf


kdf.dtypes


# ## Missing Data

pdf1 = pdf.reindex(index=dates[0:4], columns=list(pdf.columns) + ['E'])
pdf1.loc[dates[0]:dates[1], 'E'] = 1
kdf1 = ks.from_pandas(pdf1)
kdf1


kdf1.dropna(how='any')


kdf1.fillna(value=5)


# ## Spark Configurations
# 
# PySpark config can be applied in Koalas. One of the example is to enable Arrow Optimization for huge speed up internal pandas conversion. 

prev = spark.conf.get("spark.sql.execution.arrow.enabled")  # Keep its default value.
ks.set_option("compute.default_index_type", "distributed")  # Use default index prevent overhead.
import warnings
warnings.filterwarnings("ignore")  # Ignore warnings coming from Arrow optimizations.


spark.conf.set("spark.sql.execution.arrow.enabled", True)
get_ipython().run_line_magic('timeit', 'ks.range(300000).to_pandas()')


spark.conf.set("spark.sql.execution.arrow.enabled", False)
get_ipython().run_line_magic('timeit', 'ks.range(300000).to_pandas()')


ks.reset_option("compute.default_index_type")
spark.conf.set("spark.sql.execution.arrow.enabled", prev)  # Set its default value back.


# ## Grouping

kdf = ks.DataFrame({'A': ['foo', 'bar', 'foo', 'bar',
                          'foo', 'bar', 'foo', 'foo'],
                    'B': ['one', 'one', 'two', 'three',
                          'two', 'two', 'one', 'three'],
                    'C': np.random.randn(8),
                    'D': np.random.randn(8)})
kdf


kdf.groupby('A').sum()


kdf.groupby(['A', 'B']).sum()


# ## Plotting

get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot as plt


pser = pd.Series(np.random.randn(1000),
                 index=pd.date_range('1/1/2000', periods=1000))


kser = ks.Series(pser)
kser = kser.cummax()
kser.plot()


pdf = pd.DataFrame(np.random.randn(1000,4), index=pser.index,
                      columns=['A','B','C','D'])
kdf = ks.from_pandas(pdf)
kdf = kdf.cummax()
kdf.plot()


speed = [0.1, 17.5, 40, 48, 52, 69, 88]
lifespan = [2, 8, 70, 1.5, 25, 12, 28]
index = ['snail', 'pig', 'elephant',
          'rabbit', 'giraffe', 'coyote', 'horse']
kdf = ks.DataFrame({'speed': speed,
                     'lifespan': lifespan}, index=index)
kdf.plot.bar()


kdf.plot.barh()


kdf = ks.DataFrame({'mass': [0.330, 4.87, 5.97],
                     'radius': [2439.7, 6051.8, 6378.1]},
                    index=['Mercury', 'Venus', 'Earth'])
 kdf.plot.pie(y='mass')


kdf = ks.DataFrame({
     'sales': [3, 2, 3, 9, 10, 6, 3],
     'signups': [5, 5, 6, 12, 14, 13, 9],
     'visits': [20, 42, 28, 62, 81, 50, 90],
 }, index=pd.date_range(start='2019/08/15', end='2020/03/09',
                        freq='M'))
kdf.plot.area()


kdf = ks.DataFrame({'pig': [20, 18, 489, 675, 1776],
                     'horse': [4, 25, 281, 600, 1900]},
                    index=[1990, 1997, 2003, 2009, 2014])
kdf.plot.line()


kdf = pd.DataFrame(
     np.random.randint(1, 7, 6000),
     columns=['one'])
kdf['two'] = kdf['one'] + np.random.randint(1, 7, 6000)
kdf = ks.from_pandas(kdf)
kdf.plot.hist(bins=12, alpha=0.5)


kdf = ks.DataFrame([[5.1, 3.5, 0], [4.9, 3.0, 0], [7.0, 3.2, 1],
                     [6.4, 3.2, 1], [5.9, 3.0, 2]],
                    columns=['length', 'width', 'species'])
kdf.plot.scatter(x='length', y='width', c='species', colormap='viridis')


# ## Getting data in/out

# csv
kdf.to_csv('output/foo.csv')
ks.read_csv('output/foo.csv').head(10)


# parquet
kdf.to_parquet('output/bar.parquet')
ks.read_parquet('output/bar.parquet').head(10)


# Spark IO
kdf.to_spark_io('output/zoo.orc', format="orc")
ks.read_spark_io('output/zoo.orc', format="orc").head(10)


# ## Using SQL in Koalas

kdf = ks.DataFrame({'year': [1990, 1997, 2003, 2009, 2014],
                     'pig': [20, 18, 489, 675, 1776],
                     'horse': [4, 25, 281, 600, 1900]})
kdf


ks.sql("SELECT year, pig, horse FROM {kdf} WHERE horse > 200")


pdf = pd.DataFrame({'year': [1990, 1997, 2003, 2009, 2014],
                     'sheep': [22, 50, 121, 445, 791],
                     'chicken': [250, 326, 589, 1241, 2118]})
pdf


ks.sql('''
    SELECT ks.pig, pd.chicken
    FROM {kdf} ks INNER JOIN {pdf} pd
    ON ks.year = pd.year
    ORDER BY ks.pig, pd.chicken
''')


# ## Spark Schema

import numpy as np
import pandas as pd
kdf = ks.DataFrame({'a': list('abc'),
                     'b': list(range(1, 4)),
                     'c': np.arange(3, 6).astype('i1'),
                     'd': np.arange(4.0, 7.0, dtype='float64'),
                     'e': [True, False, True],
                     'f': pd.date_range('20130101', periods=3)},
                    columns=['a', 'b', 'c', 'd', 'e', 'f'])


# Print the schema out in Spark’s DDL formatted string
kdf.spark.schema().simpleString()


kdf.spark.schema(index_col='index').simpleString()


# Print out the schema as same as Spark’s DataFrame.printSchema()
kdf.spark.print_schema()


# ## Explain Spark Plan

kdf.spark.explain()


kdf.spark.explain(True)


kdf.spark.explain(mode="extended")

