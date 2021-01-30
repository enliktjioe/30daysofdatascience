#!/usr/bin/env python
# coding: utf-8

# ## References
# 
# - [1] https://towardsdatascience.com/beyond-pandas-spark-dask-vaex-and-other-big-data-technologies-battling-head-to-head-a453a1f8cc13
# - [2] https://github.com/DataSystemsGroupUT/dataeng

# ## Week 1 - Beyond Pandas
# 
# - I've got the inspiration about trying pandas alternative for data processing from this article [[1]](https://towardsdatascience.com/beyond-pandas-spark-dask-vaex-and-other-big-data-technologies-battling-head-to-head-a453a1f8cc13)
# - In terms of implementation, `koalas` became my personal favourite over the other libraries, as its syntax work in very similar way like `pandas`, so there is nothing much changes need in the code when we used `pandas` before.
# - My second favourite is `vaex`, as it has active github repositories and very good documentation
# - `Dask` is long-time state-of-the-art for multi-processing library in Python, but lack of developer support and missing a lot of important pandas features that made me hard to migrate from pandas 
# - Overall, through this first week, I learned more about how multi-processing in Python can work better with open-source libraries, as an alternative to built-in `multiprocessing` library in Python.

# ## Week 2 - Data Streaming
# 
# - This topic inspired by [Data Engineering](https://github.com/DataSystemsGroupUT/dataeng) course from the University of Tartu, Estonia (Autumn 2020)
# - Even I didn't dig up thoroughly for every new frameworks that I learned, this challenge helped me to getting started about the advantage and how it works from top layer
# - I learned Kafka and Airflow mostly from Data Engineering course, but still left me many task that I need to explore again in my free time
# - I learned Spark through my internship time in [Positium](positium.com) to improve their big data framework performance.
# - Flink and Storm are two new things that I started to learn through this 2nd week.
# - Meanwhile, Docker also become the most important part to setup many different tools in different OS

# ## Week 3 - Data Visualization
# 
# - DataViz came from my inner voice to learn something that not too much technical
# - It turned out that I really enjoying this 3rd week, as I getting to know to many Python and R dataviz libraries, in addition to state-of-the-art commercial dataviz such as Tableau
# - Apache Superset are the new dataviz library that works really different compare to others, it works through web server with enterprise-ready web app.
# - I've got Tableau student license which helped me to spend time learning through the official tutorial that later I've found out, it needs more time to explore as it's a stand-alone enterprise software, not only single library
# - Plotly become my personal favourite for open-source interactive dataviz, as it able to be expanded into dashboard throught its Dash dashboard library from the same developer.

# ## Week 4 - Dashboard
# 
# - There are total 4 dashboard libraries that I learned through this 4th week: dash, django, bokeh, and R shiny.
# - Surprisingly I like how Django works as web app framework through Python, it's an anti-mainstream when you want try something new, beside using only Javascript.
# - I'm getting know to Shiny dashboard when I took Data Science 3-months course in Algoritma Data Science, Jakarta, Indonesia. On that time, Jan - Mar 2019, I focusedly learning Data Science using R, before moving on to Python.
# - Bokeh also interesting library, but I found out that Plotly Dash has more features compare to Bokeh, with very similar dataviz quality
# - Out there, probably open-source frameworks like Grafana and Kibana also worthful to learn in future

# ## What's Next?
# 
# - Currently,'m really interested to learn more about DevOps technology, especially in orchestration tool such as Kubernetes, Apache Zeppelin, or Apache Airflow
# - Due to my current study priorities to finish my Master's thesis in the field on NLP (Natural Language Processing) and Requirement Engineering, I decided to make next personal challenge in Feb 2021 as `#30DaysOfNLP`
