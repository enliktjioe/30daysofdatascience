#!/usr/bin/env python
# coding: utf-8

# ## References
# 
# - [1] https://reflectivedata.com/getting-started-apache-superset-enterprise-ready-business-intelligence-platform/
# - [2] https://superset.apache.org/docs/installation/installing-superset-using-docker-compose
# - [3] https://medium.com/datadriveninvestor/create-your-first-sales-dashboard-in-apache-superset-c6a7f3d628d6

# ## Installation
# 
# - Using `docker-compose` is recommended [2]
# 
# ### Issues
# - Docker installation error due to existing local Postgres instance on Port 5432 https://github.com/apache/superset/issues/7800

# Running superset using docker-compose
from IPython.display import Image
Image(filename='img/superset_2.png')


# ## Dashboard Access
# 
# - via [http://localhost:8088](http://localhost:8088)

# Sample Dashboard view of Apache Superset (accessible from localhost:8088)
from IPython.display import Image
Image(filename='img/superset_1.png') 




