#!/usr/bin/env python
# coding: utf-8

# ## References
# 
# - [1] https://realpython.com/pandas-plot-python/
# - [2] https://www.tutorialspoint.com/python_pandas/python_pandas_visualization.htm
# - [3] https://towardsdatascience.com/9-pandas-visualizations-techniques-for-effective-data-analysis-fc17feb651db
# - [4] https://pandas.pydata.org/pandas-docs/stable/user_guide/visualization.html
# - [5] https://stackabuse.com/introduction-to-data-visualization-in-python-with-pandas/
# - [6] https://programminghistorian.org/en/lessons/visualizing-with-bokeh
# - [7] https://programminghistorian.org/en/lessons/mapping-with-python-leaflet
# - [8] https://towardsdatascience.com/introduction-to-data-visualization-in-python-89a54c97fbed

# ### Notes
# 
# - Pandas Visualization, easy to use interface, built on Matplotlib [4]
# - The `%matplotlib` magic command sets up your Jupyter Notebook for displaying plots with Matplotlib [1]
#     - The standard Matplotlib graphics backend is used by default, and your plots will be displayed in a separate window. [1]
#     - the inline backend is popular for Jupyter Notebooks because it displays the plot in the notebook itself [1]

## solve issue on macOS 10.12.6 - URLError: <urlopen error [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed (_ssl.c:847)>
import ssl 
ssl._create_default_https_context = ssl._create_unverified_context


import pandas as pd

download_url = (
    "https://raw.githubusercontent.com/fivethirtyeight/"
    "data/master/college-majors/recent-grads.csv"
)

df = pd.read_csv(download_url)

type(df)


# #### showing plot outside of the notebook

get_ipython().run_line_magic('matplotlib', '')


df.plot(x="Rank", y=["P25th", "Median", "P75th"])


# #### showing plot inside of the notebook

get_ipython().run_line_magic('matplotlib', 'inline')


df.plot(x="Rank", y=["P25th", "Median", "P75th"])


# ## Basic Pandas Plot

import pandas as pd
import numpy as np

df = pd.DataFrame(np.random.randn(10,4),index=pd.date_range('1/1/2000',
   periods=10), columns=list('ABCD'))

df.plot()


import pandas as pd
import numpy as np

df = pd.DataFrame(np.random.rand(10,4),columns=['a','b','c','d'])

df.plot.barh(stacked=True)


import pandas as pd
import numpy as np

df = pd.DataFrame(3 * np.random.rand(4), index=['a', 'b', 'c', 'd'], columns=['x'])
df.plot.pie(subplots=True)


from sklearn.datasets import load_iris
import pandas as pd

data = load_iris()
df = pd.DataFrame(data['data'], columns=data['feature_names'])
df['species'] = data['target']
df.head()
df.plot.scatter(x='sepal length (cm)', y='sepal width (cm)')


df.plot.kde(subplots=True, figsize=(5,9))


# ### Scatter Matrix Plot

from pandas.plotting import scatter_matrix
scatter_matrix(df, figsize=(10, 10))


# ## Using sample data

import pandas as pd
menu = pd.read_csv('data/indian_food.csv')
menu


name_and_time = menu[['name','cook_time']].head(10)
name_and_time.plot.bar(x='name',y='cook_time', rot=30)


name_and_time = menu[['name','prep_time','cook_time']].head(10)
name_and_time.plot.bar(x='name', rot=30)


name_and_time.plot.bar(x='name', y=['prep_time','cook_time'], rot=30)


# Plotting Stacked Bar Graphs with Pandas
name_and_time.plot.bar(x='name', stacked=True)


# ### Customized Bar Plot in Pandas

import pandas as pd
import matplotlib.pyplot as plt

menu = pd.read_csv('data/indian_food.csv')
name_and_time = menu[['name','cook_time','prep_time']].head()

name_and_time.plot.barh(x='name',color =['orange','green'], title = "Dishes", grid = True, figsize=(5,6), legend = True)
plt.show()


# ### Change Tick Frequency for Pandas Histogram

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Clean data and extract what we're looking for
menu = pd.read_csv('data/indian_food.csv')
menu = menu[menu.cook_time != -1] # Filtering
cook_time = menu['cook_time']

# Construct histogram plot with 50 bins
cook_time.plot.hist(bins=50)

# Modify X-Axis ticks
plt.xticks(np.arange(0, cook_time.max(), 20))
plt.xticks(rotation = 45) 

plt.legend()
plt.show()


# multiple histograms
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Filtering and cleaning
menu = pd.read_csv('data/indian_food.csv')
menu = menu[(menu.cook_time!=-1) & (menu.prep_time!=-1)] 

# Extracting relevant data
cook_time = menu['cook_time']
prep_time = menu['prep_time']

# Alpha indicates the opacity from 0..1
prep_time.plot.hist(alpha = 0.6 , bins = 50) 
cook_time.plot.hist(alpha = 0.6, bins = 50)

plt.xticks(np.arange(0, cook_time.max(), 20))
plt.xticks(rotation = 45) 
plt.legend()
plt.show()


# Customizing Histograms Plot
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

menu = pd.read_csv('data/indian_food.csv')
menu = menu[(menu.cook_time!=-1) & (menu.prep_time!=-1)] #filltering

cook_time = menu['cook_time']
prep_time = menu['prep_time']

prep_time.plot.hist(alpha = 0.6 , color = 'green', title = 'Cooking time', grid = True, bins = 50)
cook_time.plot.hist(alpha = 0.6, color = 'red', figsize = (7,7), grid = True, bins = 50)

plt.xticks(np.arange(0, cook_time.max(), 20))
plt.xticks(rotation = 45) 

plt.legend()
plt.show()


# ### Plotting Area Plots with Pandas

import pandas as pd
import matplotlib.pyplot as plt

menu = pd.read_csv('data/indian_food.csv')
menu = menu[(menu.cook_time!=-1) & (menu.prep_time!=-1)]

# Simplifying the graph
time = menu.groupby('prep_time').mean() 
time.plot.area()

plt.legend()
plt.show()


# ### Plotting Stacked Area Plots

import pandas as pd
import matplotlib.pyplot as plt

menu = pd.read_csv('data/indian_food.csv')
menu = menu[(menu.cook_time!=-1) & (menu.prep_time!=-1)]

menu.plot.area()

plt.legend()
plt.show()


# ### Plotting Pie Charts with Pandas

import pandas as pd
import matplotlib.pyplot as plt

menu = pd.read_csv('data/indian_food.csv')

flavors = menu[menu.flavor_profile != '-1']
flavors['flavor_profile'].value_counts().plot.pie()

plt.legend()
plt.show()


# ### Plotting a Bootstrap Plot in Pandas

import pandas as pd
import matplotlib.pyplot as plt
import scipy
from pandas.plotting import bootstrap_plot

menu = pd.read_csv('data/indian_food.csv')

bootstrap_plot(menu['cook_time'])
plt.show()


# ### Stacked Bar Charts and Sub-sampling Data: Types of Munitions Dropped by Country

# !pip install bokeh


#munitions_by_country_stacked.py
import pandas as pd
from bokeh.plotting import figure, output_file, show
from bokeh.models import ColumnDataSource
from bokeh.palettes import Spectral3
output_file('types_of_munitions.html')

df = pd.read_csv('https://raw.githubusercontent.com/programminghistorian/ph-submissions/gh-pages/assets/visualizing-with-bokeh/thor_wwii.csv')
df.head()


filter = df['COUNTRY_FLYING_MISSION'].isin(('USA','GREAT BRITAIN'))
df = df[filter]


grouped = df.groupby('COUNTRY_FLYING_MISSION')['TONS_IC', 'TONS_FRAG', 'TONS_HE'].sum()

#convert tons to kilotons again
grouped = grouped / 1000


source = ColumnDataSource(grouped)
countries = source.data['COUNTRY_FLYING_MISSION'].tolist()
p = figure(x_range=countries)


p.vbar_stack(stackers=['TONS_HE', 'TONS_FRAG', 'TONS_IC'],
             x='COUNTRY_FLYING_MISSION', source=source,
             legend = ['High Explosive', 'Fragmentation', 'Incendiary'],
             width=0.5, color=Spectral3)


p.title.text ='Types of Munitions Dropped by Allied Country'
p.legend.location = 'top_left'

p.xaxis.axis_label = 'Country'
p.xgrid.grid_line_color = None	#remove the x grid lines

p.yaxis.axis_label = 'Kilotons of Munitions'

show(p)


from IPython.display import Image
Image(filename='img/bokeh_plot.png') 


# ### Time-Series and Annotations: Bombing Operations over Time

#my_first_timeseries.py
import pandas as pd
from bokeh.plotting import figure, output_file, show
from bokeh.models import ColumnDataSource
from bokeh.palettes import Spectral3
output_file('simple_timeseries_plot.html')

df = pd.read_csv('https://raw.githubusercontent.com/programminghistorian/ph-submissions/gh-pages/assets/visualizing-with-bokeh/thor_wwii.csv')

#make sure MSNDATE is a datetime format
df['MSNDATE'] = pd.to_datetime(df['MSNDATE'], format='%m/%d/%Y')

grouped = df.groupby('MSNDATE')['TOTAL_TONS', 'TONS_IC', 'TONS_FRAG'].sum()
grouped = grouped/1000

source = ColumnDataSource(grouped)

p = figure(x_axis_type='datetime')

p.line(x='MSNDATE', y='TOTAL_TONS', line_width=2, source=source, legend='All Munitions')
p.line(x='MSNDATE', y='TONS_FRAG', line_width=2, source=source, color=Spectral3[1], legend='Fragmentation')
p.line(x='MSNDATE', y='TONS_IC', line_width=2, source=source, color=Spectral3[2], legend='Incendiary')

p.yaxis.axis_label = 'Kilotons of Munitions Dropped'

show(p)


from IPython.display import Image
Image(filename='img/bokeh_plot_2.png') 




