#!/usr/bin/env python
# coding: utf-8

# ## References
# 
# - [1] https://www.journaldev.com/19692/python-plotly-tutorial
# - [2] https://images.plot.ly/plotly-documentation/images/plotly_js_cheat_sheet.pdf (Cheat Sheet)
# - [3] https://www.kaggle.com/kanncaa1/plotly-tutorial-for-beginners

# ### Why Plotly [1]
# 
# Plotly has got some amazing features that make it better than other graphing libraries:
# 
# - It is interactive by default
# - Charts are not saved as images but serialized as JSON, making them open to be read with R, MATLAB, Julia and others easily
# - Exports vector for print/publication
# - Easy to manipulate/embed on web

import ssl 
ssl._create_default_https_context = ssl._create_unverified_context


# ### Simple Graph

import plotly
from plotly.graph_objs import Scatter, Layout

# # Open in new window as new HTML page
# plotly.offline.iplot({
#     "data": [Scatter(x=[1, 2, 3, 4], y=[4, 3, 2, 1])],
#     "layout": Layout(title="hello world")
# })

# Plot and embed in ipython notebook!
plotly.offline.iplot({
    "data": [Scatter(x=[1, 2, 3, 4], y=[4, 3, 2, 1])],
    "layout": Layout(title="hello world")
})


# ### Scatter Plot

import plotly
import plotly.graph_objs as go

# Create random data with numpy
import numpy as np

N = 1000
random_x = np.random.randn(N)
random_y = np.random.randn(N)

# Create a trace
trace = go.Scatter(
    x = random_x,
    y = random_y,
    mode = 'markers'
)
data = [trace]

plotly.offline.iplot(data, filename='basic-scatter')


# ### Line and Scatter Plot


import plotly
import plotly.graph_objs as go

# Create random data with numpy
import numpy as np

N = 100
random_x = np.linspace(0, 1, N)
random_y0 = np.random.randn(N)+5
random_y1 = np.random.randn(N)
random_y2 = np.random.randn(N)-5

# Create traces
trace0 = go.Scatter(
    x = random_x,
    y = random_y0,
    mode = 'markers',
    name = 'markers'
)

trace1 = go.Scatter(
    x = random_x,
    y = random_y1,
    mode = 'lines+markers',
    name = 'lines+markers'
)

trace2 = go.Scatter(
    x = random_x,
    y = random_y2,
    mode = 'lines',
    name = 'lines'
)

data = [trace0, trace1, trace2]
plotly.offline.iplot(data, filename='scatter-mode')


# ### Contour Plot

from plotly import tools
import plotly
import plotly.graph_objs as go

trace0 = go.Contour(
    z=[[2, 4, 7, 12, 13, 14, 15, 16],
       [3, 1, 6, 11, 12, 13, 16, 17],
       [4, 2, 7, 7, 11, 14, 17, 18],
       [5, 3, 8, 8, 13, 15, 18, 19],
       [7, 4, 10, 9, 16, 18, 20, 19],
       [9, 10, 5, 27, 23, 21, 21, 21],
       [11, 14, 17, 26, 25, 24, 23, 22]],
    line=dict(smoothing=0),
)

trace1 = go.Contour(
    z=[[2, 4, 7, 12, 13, 14, 15, 16],
       [3, 1, 6, 11, 12, 13, 16, 17],
       [4, 2, 7, 7, 11, 14, 17, 18],
       [5, 3, 8, 8, 13, 15, 18, 19],
       [7, 4, 10, 9, 16, 18, 20, 19],
       [9, 10, 5, 27, 23, 21, 21, 21],
       [11, 14, 17, 26, 25, 24, 23, 22]],
    line=dict(smoothing=0.85),
)

data = tools.make_subplots(rows=1, cols=2,
                          subplot_titles=('Without Smoothing', 'With Smoothing'))

data.append_trace(trace0, 1, 1)
data.append_trace(trace1, 1, 2)

plotly.offline.iplot(data)


# ### Time Series Plot

import plotly
import plotly.graph_objs as go
import pandas as pd

df = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/finance-charts-apple.csv")

data = [go.Scatter(
          x=df.Date,
          y=df['AAPL.Close'])]

plotly.offline.iplot(data)


# ### Kaggle Tutorial
# https://www.kaggle.com/kanncaa1/plotly-tutorial-for-beginners

timesData = pd.read_csv("data/kaggle_world_university_rankings/timesData.csv")
timesData.head()


# ### Line Charts

# prepare data frame
df = timesData.iloc[:100,:]

import plotly.graph_objs as go
import plotly.offline as pyoff

# Creating trace1
trace1 = go.Scatter(
                    x = df.world_rank,
                    y = df.citations,
                    mode = "lines",
                    name = "citations",
                    marker = dict(color = 'rgba(16, 112, 2, 0.8)'),
                    text= df.university_name)
# Creating trace2
trace2 = go.Scatter(
                    x = df.world_rank,
                    y = df.teaching,
                    mode = "lines+markers",
                    name = "teaching",
                    marker = dict(color = 'rgba(80, 26, 80, 0.8)'),
                    text= df.university_name)
data = [trace1, trace2]
layout = dict(title = 'Citation and Teaching vs World Rank of Top 100 Universities',
              xaxis= dict(title= 'World Rank',ticklen= 5,zeroline= False)
             )
fig = dict(data = data, layout = layout)
pyoff.iplot(fig)


# ### Pie Chart

# data preparation
df2016 = timesData[timesData.year == 2016].iloc[:7,:]
pie1 = df2016.num_students
pie1_list = [float(each.replace(',', '.')) for each in df2016.num_students]  # str(2,4) => str(2.4) = > float(2.4) = 2.4
labels = df2016.university_name
# figure
fig = {
  "data": [
    {
      "values": pie1_list,
      "labels": labels,
      "domain": {"x": [0, .5]},
      "name": "Number Of Students Rates",
      "hoverinfo":"label+percent+name",
      "hole": .3,
      "type": "pie"
    },],
  "layout": {
        "title":"Universities Number of Students rates",
        "annotations": [
            { "font": { "size": 20},
              "showarrow": False,
              "text": "Number of Students",
                "x": 0.20,
                "y": 1
            },
        ]
    }
}
pyoff.iplot(fig)


# ### Bubble Chart

# data preparation
df2016 = timesData[timesData.year == 2016].iloc[:20,:]
num_students_size  = [float(each.replace(',', '.')) for each in df2016.num_students]
international_color = [float(each) for each in df2016.international]
data = [
    {
        'y': df2016.teaching,
        'x': df2016.world_rank,
        'mode': 'markers',
        'marker': {
            'color': international_color,
            'size': num_students_size,
            'showscale': True
        },
        "text" :  df2016.university_name    
    }
]
pyoff.iplot(data)


# ### Histogram

# prepare data
x2011 = timesData.student_staff_ratio[timesData.year == 2011]
x2012 = timesData.student_staff_ratio[timesData.year == 2012]

trace1 = go.Histogram(
    x=x2011,
    opacity=0.75,
    name = "2011",
    marker=dict(color='rgba(171, 50, 96, 0.6)'))
trace2 = go.Histogram(
    x=x2012,
    opacity=0.75,
    name = "2012",
    marker=dict(color='rgba(12, 50, 196, 0.6)'))

data = [trace1, trace2]
layout = go.Layout(barmode='overlay',
                   title=' students-staff ratio in 2011 and 2012',
                   xaxis=dict(title='students-staff ratio'),
                   yaxis=dict( title='Count'),
)
fig = go.Figure(data=data, layout=layout)
pyoff.iplot(fig)


# ### Word Cloud (not in Plotly, using Matplotlib instead)

import matplotlib.pyplot as plt
from wordcloud import *

# data prepararion
x2011 = timesData.country[timesData.year == 2011]
plt.subplots(figsize=(8,8))
wordcloud = WordCloud(
                          background_color='white',
                          width=512,
                          height=384
                         ).generate(" ".join(x2011))
plt.imshow(wordcloud)
plt.axis('off')
plt.savefig('graph.png')

plt.show()


# ### 3D Scatter Plot with Colorscaling¶

dataframe = timesData[timesData.year == 2015]

# create trace 1 that is 3d scatter
trace1 = go.Scatter3d(
    x=dataframe.world_rank,
    y=dataframe.research,
    z=dataframe.citations,
    mode='markers',
    marker=dict(
        size=10,
        color='rgb(255,0,0)',                # set color to an array/list of desired values      
    )
)

data = [trace1]
layout = go.Layout(
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=0  
    )
    
)
fig = go.Figure(data=data, layout=layout)
pyoff.iplot(fig)




