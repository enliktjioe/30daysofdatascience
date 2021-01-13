#!/usr/bin/env python
# coding: utf-8

# ## References
# 
# - https://realpython.com/ggplot-python/

from plotnine.data import economics


economics


from plotnine.data import economics

from plotnine import ggplot, aes, geom_line


(

    ggplot(economics)  # What data to use

    + aes(x="date", y="pop")  # What variable to use

    + geom_line()  # Geometric object to use for drawing

)


# ##  Empty plot because not aesthetic and geom from Grammar of Graphics

from plotnine.data import mpg
from plotnine import ggplot

ggplot(mpg)


from plotnine.data import mpg
from plotnine import ggplot, aes

ggplot(mpg) + aes(x="class", y="hwy")


from plotnine.data import mpg
from plotnine import ggplot, aes, geom_point
ggplot(mpg) + aes(x="class", y="hwy") + geom_point()


from plotnine.data import mpg
from plotnine import ggplot, aes, geom_bar

ggplot(mpg) + aes(x="class") + geom_bar()


# Import our example dataset with the levels of Lake Huron 1875–1975
from plotnine.data import huron

huron


from plotnine.data import huron
from plotnine import ggplot, aes, stat_bin, geom_bar

ggplot(huron) + aes(x="level") + stat_bin(bins=10) + geom_bar()


from plotnine.data import huron
from plotnine import ggplot, aes, geom_histogram

ggplot(huron) + aes(x="level") + geom_histogram(bins=10)


from plotnine.data import huron
from plotnine import ggplot, aes, geom_boxplot

(
  ggplot(huron)
  + aes(x="factor(decade)", y="level")
  + geom_boxplot()
)


from plotnine.data import economics
from plotnine import ggplot, aes, scale_x_timedelta, labs, geom_line

(
    ggplot(economics)
    + aes(x="date", y="pop")
    + scale_x_timedelta(name="Years since 1970")
    + labs(title="Population Evolution", y="Population")
    + geom_line()
)


from plotnine.data import mpg
from plotnine import ggplot, aes, geom_bar

ggplot(mpg) + aes(x="class") + geom_bar()


from plotnine.data import mpg
from plotnine import ggplot, aes, geom_bar, coord_flip

ggplot(mpg) + aes(x="class") + geom_bar() + coord_flip()
from plotnine.data import mpg
from plotnine import ggplot, aes, geom_bar, coord_flip

ggplot(mpg) + aes(x="class") + geom_bar() + coord_flip()


# ## Facets: Plot Subsets of Data Into Panels in the Same Plot

from plotnine.data import mpg
from plotnine import ggplot, aes, facet_grid, labs, geom_point

(
    ggplot(mpg)
    + facet_grid(facets="year~class")
    + aes(x="displ", y="hwy")
    + labs(
        x="Engine Size",
        y="Miles per Gallon",
        title="Miles per Gallon for Each Year and Vehicle Class",
    )
    + geom_point()
)


from plotnine.data import mpg
from plotnine import ggplot, aes, facet_grid, labs, geom_point, theme_dark

(
    ggplot(mpg)
    + facet_grid(facets="year~class")
    + aes(x="displ", y="hwy")
    + labs(
        x="Engine Size",
        y="Miles per Gallon",
        title="Miles per Gallon for Each Year and Vehicle Class",
    )
    + geom_point()
    + theme_dark()
)


from plotnine.data import mpg
from plotnine import ggplot, aes, facet_grid, labs, geom_point, theme_xkcd

(
    ggplot(mpg)
    + facet_grid(facets="year~class")
    + aes(x="displ", y="hwy")
    + labs(
        x="Engine Size",
        y="Miles per Gallon",
        title="Miles per Gallon for Each Year and Vehicle Class",
    )
    + geom_point()
    + theme_xkcd()
)


# ## Multidimentional Data Visualization

from plotnine.data import mpg
from plotnine import ggplot, aes, labs, geom_point

(
    ggplot(mpg)
    + aes(x="cyl", y="hwy", color="class")
    + labs(
        x="Engine Cylinders",
        y="Miles per Gallon",
        color="Vehicle Class",
        title="Miles per Gallon for Engine Cylinders and Vehicle Classes",
    )
    + geom_point()
)


# ## Exporting to File

from plotnine.data import economics
from plotnine import ggplot, aes, geom_line

myPlot = ggplot(economics) + aes(x="date", y="pop") + geom_line()
myPlot.save("myplot.png", dpi=600)




