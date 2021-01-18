#!/usr/bin/env python
# coding: utf-8

# ## References
# 
# - [1] https://seaborn.pydata.org/
# - [2] https://python-graph-gallery.com/seaborn/
# - [3] https://jakevdp.github.io/PythonDataScienceHandbook/04.14-visualization-with-seaborn.html
# - [4] https://www.datacamp.com/community/tutorials/seaborn-python-tutorial#sm
# - [5] https://www.mygreatlearning.com/blog/seaborn-tutorial/
# - [6] https://elitedatascience.com/python-seaborn-tutorial
# 
# Notes:
# 
# - Seaborn is based on matplotlib [1]
# - It allows to make your charts prettier, and facilitates some of the common data viz needs (like mapping color or faceting) [2]
# - Some Matplotlib problems that leads to Seaborn: [3]
#  - Matplotlib is relatively low level
#  - Matplotlib predated Pandas by more than a decade, but it's not designed to work with Pandas
# - Seaborn works great with pandas DataFrames. [4]

## Library
import ssl 
ssl._create_default_https_context = ssl._create_unverified_context


# ## Matplotlib vs Seaborn

# Import the necessary libraries
import matplotlib.pyplot as plt
import pandas as pd

# Initialize Figure and Axes object
fig, ax = plt.subplots()

# Load in data
tips = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv")

# Create violinplot
ax.violinplot(tips["total_bill"], vert=False)

# Show the plot
plt.show()


# Import the necessary libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
tips = sns.load_dataset("tips")

# Create violinplot
sns.violinplot(x = "total_bill", data=tips)

# Show the plot
plt.show()


# Import necessarily libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
titanic = sns.load_dataset("titanic")

# Set up a factorplot
g = sns.factorplot("class", "survived", "sex", data=titanic, kind="bar", palette="muted", legend=False)
                   
# Show plot
plt.show()


# ## Rotate Label Text in Seaborn

# Import the necessary libraries
import matplotlib.pyplot as plt
import seaborn as sns 
import numpy as np
import pandas as pd

# Initialize the data
x = 10 ** np.arange(1, 10)
y = x * 2
data = pd.DataFrame(data={'x': x, 'y': y})

# Create an lmplot
grid = sns.lmplot('x', 'y', data, size=7, truncate=True, scatter_kws={"s": 100})

# Rotate the labels on x-axis
grid.set_xticklabels(rotation=30)

# Show the plot
plt.show()


# ## relplot

import seaborn as sns
tips = sns.load_dataset("tips")
tips.head()


sns.relplot(data=tips, x="total_bill", y="tip")


sns.relplot(data=tips, x="total_bill", y="tip", hue="day")


sns.relplot(data=tips, x="total_bill", y="tip", hue="sex", col="day", col_wrap=2)


sns.relplot(data=tips, x="size", y="tip",kind="line",ci=None)


# ## Histogram

import seaborn as sns
from matplotlib import pyplot as plt
df = sns.load_dataset('iris')
sns.distplot(df['petal_length'],kde = False)


# ## Bar Plot

import matplotlib.pyplot as plt
import seaborn as sns
  
sns.set_context('paper')
 
# load dataset
titanic = sns.load_dataset('titanic')
# create plot
sns.barplot(x = 'embark_town', y = 'age', data = titanic,
            palette = 'PuRd',ci=None 
            )
plt.legend()
plt.show()
print(titanic.columns)


import matplotlib.pyplot as plt
import seaborn as sns
# load dataset
titanic = sns.load_dataset('titanic')
# create plot
sns.barplot(x = 'sex', y = 'survived', hue = 'class', data = titanic,
            palette = 'PuRd',
            order = ['male', 'female'],  
            capsize = 0.05,             
            saturation = 8,             
            errcolor = 'gray', errwidth = 2,  
            ci = 'sd'  
            )
plt.legend()
plt.show()


# ## Count Plot

import matplotlib.pyplot as plt
import seaborn as sns
  
sns.set_context('paper')
 
# load dataset
titanic = sns.load_dataset('titanic')
# create plot
sns.countplot(x = 'class', hue = 'who', data = titanic, palette = 'magma')
plt.title('Survivors')
plt.show()


# ## Point Plot

# importing required packages 
import seaborn as sns 
import matplotlib.pyplot as plt 
   
# loading dataset 
data = sns.load_dataset("tips") 
sns.pointplot(x="day", y="tip", data=data)
plt.show()


sns.pointplot(x="time", y="total_bill", hue="smoker",
                   data=data, palette="Accent")


# ## Joint Plot

import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style("dark")
tips=sns.load_dataset('tips')
sns.jointplot(x='total_bill', y='tip',data=tips)


# Add regression line to scatter plot and kernel density estimate to histogram
sns.jointplot(x='total_bill', y='tip', data=tips, kind='reg') 


# Display kernel density estimate instead of scatter plot and histogram
sns.jointplot(x='total_bill', y='tip', data=tips, kind='kde')


# Display hexagons instead of points in scatter plot
sns.jointplot(x='total_bill', y='tip', data=tips, kind='hex')


# ## Regplot

import seaborn as sns
tips = sns.load_dataset("tips")
ax = sns.regplot(x="total_bill", y="tip", data=tips)


sns.regplot(x="size", y="total_bill", data=tips, x_jitter=0.1)


import seaborn as sns
tips = sns.load_dataset("tips")
ax = sns.regplot(x="total_bill", y="tip", data=tips,ci=None)


# ## Lm Plot

import seaborn as sns
tips = sns.load_dataset("tips")
sns.lmplot(x="total_bill", y="tip", data=tips)


sns.lmplot(x="total_bill", y="tip", col="day", hue="day",
               data=tips, col_wrap=2, height=3)


# ## Cluster Map

import seaborn as sns
flights=sns.load_dataset("flights")
flights = flights.pivot("month", "year", "passengers")
sns.clustermap(flights,linewidths=.5,cmap="coolwarm")


import seaborn as sns
flights=sns.load_dataset("flights")
flights = flights.pivot("month", "year", "passengers")
sns.clustermap(flights,linewidths=.5,cmap="coolwarm",col_cluster=False)


# ## Pair Plot

import seaborn as sns
from matplotlib import pyplot as plt
df = sns.load_dataset('iris')
sns.set_style("ticks")
sns.pairplot(df,hue = 'species',diag_kind = "kde",kind = "scatter",palette = "husl")
plt.show()


# ## EliteDataScience Tutorial
# https://elitedatascience.com/python-seaborn-tutorial

# Matplotlib for additional customization
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# Seaborn for plotting and styling
import seaborn as sns


# Read dataset
df = pd.read_csv("data/Pokemon.csv", encoding= 'unicode_escape') #solution = https://stackoverflow.com/a/50538501/2670476
# df = pd.read_csv("https://elitedatascience.com/wp-content/uploads/2017/04/Pokemon.csv",sep=',')
df


# ### Overlaying Plot

# Setup Color Palletes
pkmn_type_colors = ['#78C850',  # Grass
                    '#F08030',  # Fire
                    '#6890F0',  # Water
                    '#A8B820',  # Bug
                    '#A8A878',  # Normal
                    '#A040A0',  # Poison
                    '#F8D030',  # Electric
                    '#E0C068',  # Ground
                    '#EE99AC',  # Fairy
                    '#C03028',  # Fighting
                    '#F85888',  # Psychic
                    '#B8A038',  # Rock
                    '#705898',  # Ghost
                    '#98D8D8',  # Ice
                    '#7038F8',  # Dragon
                   ]


# Set figure size with matplotlib
plt.figure(figsize=(10,6))
 
# Create plot
sns.violinplot(x='Type 1',
               y='Attack', 
               data=df, 
               inner=None, # Remove the bars inside the violins
               palette=pkmn_type_colors)
 
sns.swarmplot(x='Type 1', 
              y='Attack', 
              data=df, 
              color='k', # Make points black
              alpha=0.7) # and slightly transparent
 
# Set title with matplotlib
plt.title('Attack by Type')


# ### Swarm Plot

# Swarm plot with Pokemon color palette
sns.swarmplot(x='Type 1', y='Attack', data=df, 
              palette=pkmn_type_colors)


# Melt DataFrame
# Pre-format DataFrame
stats_df = df.drop(['Total', 'Stage', 'Legendary'], axis=1)

melted_df = pd.melt(stats_df, 
                    id_vars=["Name", "Type 1", "Type 2"], # Variables to keep
                    var_name="Stat") # Name of melted variable
melted_df.head()


# Swarmplot with melted_df
sns.swarmplot(x='Stat', y='value', data=melted_df, 
              hue='Type 1')


# 1. Enlarge the plot
plt.figure(figsize=(10,6))
 
sns.swarmplot(x='Stat', 
              y='value', 
              data=melted_df, 
              hue='Type 1', 
              split=True, # 2. Separate points by hue
              palette=pkmn_type_colors) # 3. Use Pokemon palette
 
# 4. Adjust the y-axis
plt.ylim(0, 260)
 
# 5. Place legend to the right
plt.legend(bbox_to_anchor=(1, 1), loc=2)


# ### Bar Plot

# Count Plot (a.k.a. Bar Plot)
sns.countplot(x='Type 1', data=df, palette=pkmn_type_colors)
 
# Rotate x-labels
plt.xticks(rotation=-45)


# ### Factor Plot

# Factor Plot
g = sns.factorplot(x='Type 1', 
                   y='Attack', 
                   data=df, 
                   hue='Stage',  # Color by stage
                   col='Stage',  # Separate by stage
                   kind='swarm') # Swarmplot
 
# Rotate x-axis labels
g.set_xticklabels(rotation=-45)
 
# Doesn't work because only rotates last plot
# plt.xticks(rotation=-45)


# ### Density Plot

# Density Plot
sns.kdeplot(df.Attack, df.Defense)


# ### Joint-Distribution Plot

# Joint Distribution Plot
sns.jointplot(x='Attack', y='Defense', data=df)




