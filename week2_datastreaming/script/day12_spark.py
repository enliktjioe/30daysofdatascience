#!/usr/bin/env python
# coding: utf-8

# ## References
# 
# - https://towardsdatascience.com/a-neanderthals-guide-to-apache-spark-in-python-9ef1f156d427

get_ipython().run_line_magic('config', 'IPCompleter.greedy=True')


import findspark
findspark.init()

import pyspark
import random

sc = pyspark.SparkContext(appName="Pi")
# num_samples = 100000000
num_samples = 10000

def inside(p):     
  x, y = random.random(), random.random()
  return x*x + y*y < 1

count = sc.parallelize(range(0, num_samples)).filter(inside).count()

pi = 4 * count / num_samples
print(pi)

sc.stop()


from pyspark.sql import SparkSession
spark = SparkSession.builder     .master("local[*]")     .appName("Learning_Spark")     .getOrCreate()


data = spark.read.csv('data/vgsales.csv',inferSchema=True, header=True)


data


data.count(), len(data.columns)


data.show(5)


data.printSchema()


data.select("Name","Platform","NA_Sales","EU_Sales").show(15, truncate=False)


data.describe(["NA_Sales","EU_Sales"]).show()


data.groupBy("Platform") .count() .orderBy("count", ascending=False) .show(10)


condition1 = (data.NA_Sales.isNotNull()) | (data.EU_Sales.isNotNull())
condition2 = data.JP_Sales.isNotNull()
data = data.filter(condition1).filter(condition2)
data

