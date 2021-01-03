#!/usr/bin/env python
# coding: utf-8

# ## References
# 
# - https://www.tutorialspoint.com/pyspark/pyspark_introduction.htm
# - https://sparkbyexamples.com/pyspark/pyspark-read-csv-file-into-dataframe/

from pyspark import SparkContext
sc = SparkContext("local", "Second App")


sc


import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType,StructField, StringType, IntegerType 
from pyspark.sql.types import ArrayType, DoubleType, BooleanType
from pyspark.sql.functions import col,array_contains

spark = SparkSession.builder.appName('SparkByExamples.com').getOrCreate()

df = spark.read.csv("data/data_1gb.csv")

df.printSchema()


spark.version


# df2 = spark.read.option("header",True) \
#      .csv("data/data_1gb.csv")
# df2.printSchema()


# df3 = spark.read.options(header='True', delimiter=',') \
#   .csv("data/data_1gb.csv")
# df3.printSchema()


# ##with schema
# schema = StructType() \
#       .add("Test1",IntegerType(),True) \
#       .add("Test2",StringType(),True) \
#       .add("Test3",IntegerType(),True)
      
# df_with_schema = spark.read.format("csv") \
#       .option("header", True) \
#       .schema(schema) \
#       .load("data/data_1gb.csv")
# df_with_schema.printSchema()


# # Multiple csv output
df.write.csv('output/test.csv')


# Single csv ouput
df.repartition(1).write.csv("output/test.csv", sep=',')


get_ipython().system('ls -lh output/test.csv/')


get_ipython().system('head output/test.csv/part-00000-68bc54d9-853a-4277-bb57-65369954ea67-c000.csv')


get_ipython().system('tail output/test.csv/part-00000-68bc54d9-853a-4277-bb57-65369954ea67-c000.csv')




