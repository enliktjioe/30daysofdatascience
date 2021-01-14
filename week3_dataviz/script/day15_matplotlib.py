#!/usr/bin/env python
# coding: utf-8

# ## References
# 
# - https://towardsdatascience.com/matplotlib-tutorial-learn-basics-of-pythons-powerful-plotting-library-b5d1b8f67596
# - https://matplotlib.org/tutorials/introductory/pyplot.html
# - https://github.com/rougier/matplotlib-tutorial
# - https://www.datacamp.com/community/tutorials/matplotlib-tutorial-python
# - https://realpython.com/python-matplotlib-guide/

import ssl 
ssl._create_default_https_context = ssl._create_unverified_context

import matplotlib.pyplot as plt
import numpy as np


### Pie Chart

# Data is assumption
game_platform = ["PlayStation", "Nintendo", "Xbox", "Steam", "Stadia"]
market_share = [30, 25, 10, 30, 5]
Explode = [0,0.1,0,0,0]
plt.pie(market_share, explode=Explode, labels=game_platform, shadow=True, startangle=45)
plt.axis('equal')
plt.legend(title="List of Game Platform")
plt.show()


### Bar Graph
departments = ["Admin","Development","Methodology","Marketing"]
departments_employee_number = [5,8,8,4]

plt.bar(departments, departments_employee_number, color="blue")
plt.title("Bar Graph")
plt.xlabel("Departments")
plt.ylabel("Number of Employee")
plt.show()


# Horizontal bar graph
plt.barh(departments, departments_employee_number, color="blue")
plt.title("Bar Graph")
plt.xlabel("Departments")
plt.ylabel("Number of Employee")
plt.show()


# horizontal stack bar chart
departments = ["Admin","Development","Methodology","Marketing"]
departments_employee_male = [7,12,9,7]
departments_employee_female = [5,8,8,4]

index = np.arange(len(departments_employee_male))
width = 0.30

plt.bar(index, departments_employee_male, width, color='green', label='Male')
plt.bar(index+width, departments_employee_female, width, color='blue', label='Female')
plt.title('Horizontally Stacked Bar Graphs')

plt.ylabel("Number of People")
plt.xlabel("Departments")
plt.xticks(index+width/2, departments)

plt.legend(loc='best')
plt.show()


# vertically stacked
departments = ["Admin","Development","Methodology","Marketing"]
departments_employee_male = [7,12,9,7]
departments_employee_female = [5,8,8,4]

index = np.arange(len(departments_employee_male))
width = 0.30

plt.bar(index, departments_employee_male, width, color='green', label='Male')
plt.bar(index, departments_employee_female, width, color='blue', label='Female', bottom=departments_employee_male)
plt.title('Horizontally Stacked Bar Graphs')

plt.ylabel("Number of People")
plt.xlabel("Departments")
plt.xticks(index, departments)

plt.legend(loc='best')
plt.show()


plt.plot([1, 2, 3, 4])
plt.xlabel('some numbers x')
plt.ylabel('some numbers y')
plt.show()


plt.plot([1, 2, 3, 4], [1, 4, 9, 16])


# evenly sampled time at 200ms intervals
t = np.arange(0., 5., 0.2)

# red dashes, blue squares and green triangles
plt.plot(t, t, 'r--', t, t**2, 'bs', t, t**3, 'g^')
plt.show()


# ### Plotting with keyword string

data = {'a': np.arange(50),
        'c': np.random.randint(0, 50, 50),
        'd': np.random.randn(50)}
data['b'] = data['a'] + 10 * np.random.randn(50)
data['d'] = np.abs(data['d']) * 100

plt.scatter('a', 'b', c='c', s='d', data=data)
plt.xlabel('entry a')
plt.ylabel('entry b')
plt.show()


# ### Plotting with categorial variables

names = ['group_a', 'group_b', 'group_c']
values = [1, 10, 100]

plt.figure(figsize=(9, 3))

plt.subplot(131)
plt.bar(names, values)
plt.subplot(132)
plt.scatter(names, values)
plt.subplot(133)
plt.plot(names, values)
plt.suptitle('Categorical Plotting')
plt.show()


# ### Working with multiple figures and axes

def f(t):
    return np.exp(-t) * np.cos(2*np.pi*t)

t1 = np.arange(0.0, 5.0, 0.1)
t2 = np.arange(0.0, 5.0, 0.02)

plt.figure()
plt.subplot(211)
plt.plot(t1, f(t1), 'bo', t2, f(t2), 'k')

plt.subplot(212)
plt.plot(t2, np.cos(2*np.pi*t2), 'r--')
plt.show()


# ### Working with Text

mu, sigma = 100, 15
x = mu + sigma * np.random.randn(10000)

# the histogram of the data
n, bins, patches = plt.hist(x, 50, density=1, facecolor='g', alpha=0.75)


plt.xlabel('Smarts')
plt.ylabel('Probability')
plt.title('Histogram of IQ')
plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
plt.axis([40, 160, 0, 0.03])
plt.grid(True)
plt.show()


# ### Annotating text

ax = plt.subplot(111)

t = np.arange(0.0, 5.0, 0.01)
s = np.cos(2*np.pi*t)
line, = plt.plot(t, s, lw=2)

plt.annotate('local max', xy=(2, 1), xytext=(3, 1.5),
             arrowprops=dict(facecolor='black', shrink=0.05),
             )

plt.ylim(-2, 2)
plt.show()


# ### Logarithmic and other nonlinear axes

# Fixing random state for reproducibility
np.random.seed(19680801)

# make up some data in the open interval (0, 1)
y = np.random.normal(loc=0.5, scale=0.4, size=1000)
y = y[(y > 0) & (y < 1)]
y.sort()
x = np.arange(len(y))

# plot with various axes scales
plt.figure()

# linear
plt.subplot(221)
plt.plot(x, y)
plt.yscale('linear')
plt.title('linear')
plt.grid(True)

# log
plt.subplot(222)
plt.plot(x, y)
plt.yscale('log')
plt.title('log')
plt.grid(True)

# symmetric log
plt.subplot(223)
plt.plot(x, y - y.mean())
plt.yscale('symlog', linthresh=0.01)
plt.title('symlog')
plt.grid(True)

# logit
plt.subplot(224)
plt.plot(x, y)
plt.yscale('logit')
plt.title('logit')
plt.grid(True)
# Adjust the subplot layout, because the logit one may take more space
# than usual, due to y-tick labels like "1 - 10^{-3}"
plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,
                    wspace=0.35)

plt.show()


# ### Animation

# # New figure with white background
# fig = plt.figure(figsize=(6,6), facecolor='white')

# # New axis over the whole figure, no frame and a 1:1 aspect ratio
# ax = fig.add_axes([0,0,1,1], frameon=False, aspect=1)

# # Number of ring
# n = 50
# size_min = 50
# size_max = 50*50

# # Ring position
# P = np.random.uniform(0,1,(n,2))

# # Ring colors
# C = np.ones((n,4)) * (0,0,0,1)
# # Alpha color channel goes from 0 (transparent) to 1 (opaque)
# C[:,3] = np.linspace(0,1,n)

# # Ring sizes
# S = np.linspace(size_min, size_max, n)

# # Scatter plot
# scat = ax.scatter(P[:,0], P[:,1], s=S, lw = 0.5,
#                   edgecolors = C, facecolors='None')

# # Ensure limits are [0,1] and remove ticks
# ax.set_xlim(0,1), ax.set_xticks([])
# ax.set_ylim(0,1), ax.set_yticks([])


# def update(frame):
#     global P, C, S

#     # Every ring is made more transparent
#     C[:,3] = np.maximum(0, C[:,3] - 1.0/n)

#     # Each ring is made larger
#     S += (size_max - size_min) / n

#     # Reset ring specific ring (relative to frame number)
#     i = frame % 50
#     P[i] = np.random.uniform(0,1,2)
#     S[i] = size_min
#     C[i,3] = 1

#     # Update scatter object
#     scat.set_edgecolors(C)
#     scat.set_sizes(S)
#     scat.set_offsets(P)

#     # Return the modified object
#     return scat,

# from matplotlib.animation import FuncAnimation
# animation = FuncAnimation(fig, update, interval=10, blit=True, frames=200)
# animation.save('rain.gif', writer='imagemagick', fps=30, dpi=40)
# plt.show()


# # https://www.geeksforgeeks.org/matplotlib-animation-funcanimation-class-in-python/

# ## Error in macOS 10.12.6
# from matplotlib import pyplot as plt  
# import numpy as np  
# from matplotlib.animation import FuncAnimation  
  
# # initializing a figure in  
# # which the graph will be plotted  
# fig = plt.figure()  
  
# # marking the x-axis and y-axis  
# axis = plt.axes(xlim =(0, 4),  
#                 ylim =(-2, 2))  
  
# # initializing a line variable  
# line, = axis.plot([], [], lw = 3)  
  
# # data which the line will  
# # contain (x, y)  
# def init():  
#     line.set_data([], [])  
#     return line,  
  
# def animate(i):  
#     x = np.linspace(0, 4, 1000)  
  
#     # plots a sine graph  
#     y = np.sin(2 * np.pi * (x - 0.01 * i))  
#     line.set_data(x, y)  
      
#     return line,  
  
# anim = FuncAnimation(fig, animate,  
#                     init_func = init,  
#                     frames = 200,  
#                     interval = 20,  
#                     blit = True)  
  
# anim.save('continuousSineWave.mp4',  
#           writer = 'ffmpeg', fps = 30) 


# # https://jakevdp.github.io/blog/2012/08/18/matplotlib-animation-tutorial/

# """
# General Numerical Solver for the 1D Time-Dependent Schrodinger's equation.

# adapted from code at http://matplotlib.sourceforge.net/examples/animation/double_pendulum_animated.py

# Double pendulum formula translated from the C code at
# http://www.physics.usyd.edu.au/~wheat/dpend_html/solve_dpend.c

# author: Jake Vanderplas
# email: vanderplas@astro.washington.edu
# website: http://jakevdp.github.com
# license: BSD
# Please feel free to use and modify this, but keep the above information. Thanks!
# """

# from numpy import sin, cos
# import numpy as np
# import matplotlib.pyplot as plt
# import scipy.integrate as integrate
# import matplotlib.animation as animation

# class DoublePendulum:
#     """Double Pendulum Class

#     init_state is [theta1, omega1, theta2, omega2] in degrees,
#     where theta1, omega1 is the angular position and velocity of the first
#     pendulum arm, and theta2, omega2 is that of the second pendulum arm
#     """
#     def __init__(self,
#                  init_state = [120, 0, -20, 0],
#                  L1=1.0,  # length of pendulum 1 in m
#                  L2=1.0,  # length of pendulum 2 in m
#                  M1=1.0,  # mass of pendulum 1 in kg
#                  M2=1.0,  # mass of pendulum 2 in kg
#                  G=9.8,  # acceleration due to gravity, in m/s^2
#                  origin=(0, 0)): 
#         self.init_state = np.asarray(init_state, dtype='float')
#         self.params = (L1, L2, M1, M2, G)
#         self.origin = origin
#         self.time_elapsed = 0

#         self.state = self.init_state * np.pi / 180.
    
#     def position(self):
#         """compute the current x,y positions of the pendulum arms"""
#         (L1, L2, M1, M2, G) = self.params

#         x = np.cumsum([self.origin[0],
#                        L1 * sin(self.state[0]),
#                        L2 * sin(self.state[2])])
#         y = np.cumsum([self.origin[1],
#                        -L1 * cos(self.state[0]),
#                        -L2 * cos(self.state[2])])
#         return (x, y)

#     def energy(self):
#         """compute the energy of the current state"""
#         (L1, L2, M1, M2, G) = self.params

#         x = np.cumsum([L1 * sin(self.state[0]),
#                        L2 * sin(self.state[2])])
#         y = np.cumsum([-L1 * cos(self.state[0]),
#                        -L2 * cos(self.state[2])])
#         vx = np.cumsum([L1 * self.state[1] * cos(self.state[0]),
#                         L2 * self.state[3] * cos(self.state[2])])
#         vy = np.cumsum([L1 * self.state[1] * sin(self.state[0]),
#                         L2 * self.state[3] * sin(self.state[2])])

#         U = G * (M1 * y[0] + M2 * y[1])
#         K = 0.5 * (M1 * np.dot(vx, vx) + M2 * np.dot(vy, vy))

#         return U + K

#     def dstate_dt(self, state, t):
#         """compute the derivative of the given state"""
#         (M1, M2, L1, L2, G) = self.params

#         dydx = np.zeros_like(state)
#         dydx[0] = state[1]
#         dydx[2] = state[3]

#         cos_delta = cos(state[2] - state[0])
#         sin_delta = sin(state[2] - state[0])

#         den1 = (M1 + M2) * L1 - M2 * L1 * cos_delta * cos_delta
#         dydx[1] = (M2 * L1 * state[1] * state[1] * sin_delta * cos_delta
#                    + M2 * G * sin(state[2]) * cos_delta
#                    + M2 * L2 * state[3] * state[3] * sin_delta
#                    - (M1 + M2) * G * sin(state[0])) / den1

#         den2 = (L2 / L1) * den1
#         dydx[3] = (-M2 * L2 * state[3] * state[3] * sin_delta * cos_delta
#                    + (M1 + M2) * G * sin(state[0]) * cos_delta
#                    - (M1 + M2) * L1 * state[1] * state[1] * sin_delta
#                    - (M1 + M2) * G * sin(state[2])) / den2
        
#         return dydx

#     def step(self, dt):
#         """execute one time step of length dt and update state"""
#         self.state = integrate.odeint(self.dstate_dt, self.state, [0, dt])[1]
#         self.time_elapsed += dt

# #------------------------------------------------------------
# # set up initial state and global variables
# pendulum = DoublePendulum([180., 0.0, -20., 0.0])
# dt = 1./30 # 30 fps

# #------------------------------------------------------------
# # set up figure and animation
# fig = plt.figure()
# ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
#                      xlim=(-2, 2), ylim=(-2, 2))
# ax.grid()

# line, = ax.plot([], [], 'o-', lw=2)
# time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
# energy_text = ax.text(0.02, 0.90, '', transform=ax.transAxes)

# def init():
#     """initialize animation"""
#     line.set_data([], [])
#     time_text.set_text('')
#     energy_text.set_text('')
#     return line, time_text, energy_text

# def animate(i):
#     """perform animation step"""
#     global pendulum, dt
#     pendulum.step(dt)
    
#     line.set_data(*pendulum.position())
#     time_text.set_text('time = %.1f' % pendulum.time_elapsed)
#     energy_text.set_text('energy = %.3f J' % pendulum.energy())
#     return line, time_text, energy_text

# # choose the interval based on dt and the time to animate one step
# from time import time
# t0 = time()
# animate(0)
# t1 = time()
# interval = 1000 * dt - (t1 - t0)

# ani = animation.FuncAnimation(fig, animate, frames=300,
#                               interval=interval, blit=True, init_func=init)

# # save the animation as an mp4.  This requires ffmpeg or mencoder to be
# # installed.  The extra_args ensure that the x264 codec is used, so that
# # the video can be embedded in html5.  You may need to adjust this for
# # your system: for more information, see
# # http://matplotlib.sourceforge.net/api/animation_api.html
# ani.save('double_pendulum.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

# plt.show()


# ### Subplots

# Import the necessary packages and modules
import matplotlib.pyplot as plt
import numpy as np

# Create a Figure
fig = plt.figure()

# Set up Axes
ax = fig.add_subplot(111)

# Scatter the data
ax.scatter(np.linspace(0, 1, 5), np.linspace(0, 5, 5))

# Show the plot
plt.show()


# Import `pyplot` from `matplotlib`
import matplotlib.pyplot as plt

# Initialize the plot
fig = plt.figure(figsize=(20,10))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

# or replace the three lines of code above by the following line: 
#fig, (ax1, ax2) = plt.subplots(1,2, figsize=(20,10))

# Plot the data
ax1.bar([1,2,3],[3,4,5])
ax2.barh([0.5,1,2.5],[0,1,2])

# Show the plot
plt.show()


# Import `pyplot` from `matplotlib`
import matplotlib.pyplot as plt
import numpy as np

# Initialize the plot
fig = plt.figure()
ax1 = fig.add_subplot(131)
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133)

# Plot the data
ax1.bar([1,2,3],[3,4,5])
ax2.barh([0.5,1,2.5],[0,1,2])
ax2.axhline(0.45)
ax1.axvline(0.65)
ax3.scatter(np.linspace(0, 1, 5), np.linspace(0, 5, 5))

# Delete `ax3`
fig.delaxes(ax3)

# Show the plot
plt.show()


# Import the necessary packages and modules
import matplotlib.pyplot as plt
import numpy as np

# Prepare the data
x = np.linspace(0, 10, 100)

# Plot the data
plt.plot(x, x, label='linear')

# Add a legend
plt.legend()

# Show the plot
plt.show()


# ### Use ggplot2 style

# Import `pyplot` 
import matplotlib.pyplot as plt

# Set the style to `ggplot`
plt.style.use("ggplot")


# ### Plotting in pandas

import pandas as pd
import matplotlib.transforms as mtransforms

url = 'https://fred.stlouisfed.org/graph/fredgraph.csv?id=VIXCLS'
vix = pd.read_csv(url, index_col=0, parse_dates=True, na_values='.',
                  infer_datetime_format=True,
                  squeeze=True).dropna()
ma = vix.rolling('90d').mean()
state = pd.cut(ma, bins=[-np.inf, 14, 18, 24, np.inf],
               labels=range(4))

cmap = plt.get_cmap('RdYlGn_r')
ma.plot(color='black', linewidth=1.5, marker='', figsize=(8, 4),
        label='VIX 90d MA')
ax = plt.gca()  # Get the current Axes that ma.plot() references
ax.set_xlabel('')
ax.set_ylabel('90d moving average: CBOE VIX')
ax.set_title('Volatility Regime State')
ax.grid(False)
ax.legend(loc='upper center')
ax.set_xlim(xmin=ma.index[0], xmax=ma.index[-1])

trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
for i, color in enumerate(cmap([0.2, 0.4, 0.6, 0.8])):
    ax.fill_between(ma.index, 0, 1, where=state==i,
                    facecolor=color, transform=trans)
ax.axhline(vix.mean(), linestyle='dashed', color='xkcd:dark grey',
           alpha=0.6, label='Full-period mean', marker='')


plt.ioff()
x = np.arange(-4, 5)
y1 = x ** 2
y2 = 10 / (x ** 2 + 1)
fig, ax = plt.subplots()
ax.plot(x, y1, 'rx', x, y2, 'b+', linestyle='solid')
ax.fill_between(x, y1, y2, where=y2>y1, interpolate=True,
                color='green', alpha=0.3)
lgnd = ax.legend(['y1', 'y2'], loc='upper center', shadow=True)
lgnd.get_frame().set_facecolor('#ffb19a')
plt.show()

