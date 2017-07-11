#!/usr/bin/python
# -*- coding: utf-8 -*-
from math import sin, cos, radians, pi

import numpy as np

from matplotlib.patches import Circle, FancyArrowPatch
import matplotlib.patheffects as PathEffects
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.mlab import griddata

from auttitude.geometry import project_equal_area, build_rotation_matrix


def clip_lines(data, z_tol=.1):
    """segment point pairs between inside and outside of primitive, for
    avoiding spurious lines when plotting circles."""
    z = np.transpose(data)[2]
    inside = z < z_tol
    results = []
    current = []
    for i, is_inside in enumerate(inside):
        if is_inside:
            current.append(data[i])
        elif current:
            results.append(current)
            current = []
    if current:
        results.append(current)
    return results


def net_grid(gc_spacing=10., sc_spacing=10., n=360, clean_caps=True):
    theta = np.linspace(0., 2*pi, n)
    gc_spacing, sc_spacing = radians(gc_spacing), radians(sc_spacing)
    if clean_caps:
        theta_gc = np.linspace(0. + sc_spacing, pi - sc_spacing, n)
    else:
        theta_gc = np.linspace(0., pi, n)
    gc_range = np.arange(0., pi + gc_spacing, gc_spacing)
    sc_range = np.arange(0., pi + sc_spacing, sc_spacing)
    i, j, k = np.eye(3)
    ik_circle = i[:, None]*np.sin(theta) + k[:, None]*np.cos(theta)
    great_circles = [(np.array((cos(alpha), .0, -sin(alpha)))[:, None]
                      *np.sin(theta_gc)
                      + j[:, None]*np.cos(theta_gc)).T
                     for alpha in gc_range] +\
                    [(np.array((cos(alpha), .0, -sin(alpha)))[:, None]
                      *np.sin(theta_gc)
                      + j[:, None]*np.cos(theta_gc)).T
                     for alpha in -gc_range]
    small_circles = [(ik_circle*sin(alpha) + j[:, None]*cos(alpha)).T
                     for alpha in sc_range]
    if clean_caps:
        theta_gc = np.linspace(-sc_spacing, sc_spacing, n)
        great_circles += [(np.array((cos(alpha), .0, -sin(alpha)))[:, None]
                           *np.sin(theta_gc)
                           + j[:, None]*np.cos(theta_gc)).T
                          for alpha in (0, pi/2.)]
        theta_gc = np.linspace(pi-sc_spacing, pi+sc_spacing, n)
        great_circles += [(np.array((cos(alpha), .0, -sin(alpha)))[:, None]
                           *np.sin(theta_gc)
                           + j[:, None]*np.cos(theta_gc)).T
                          for alpha in (0, pi/2.)]
    return great_circles, small_circles


class ProjectionPlot(object):
    point_defaults = {'marker': 'o',
                      'c': '#000000',
                      'ms': 3.0}

    circle_defaults = {"linewidths": 0.8,
                       "colors": "#4D4D4D",
                       "linestyles": "-"}

    contour_defaults = {"cmap": "Reds",
                        "linestyles": "-",
                        "antialiased": True}

    slip_defaults = {"lw": 1.0,
                     "ls": "-"}

    net_gc_defaults = {"linewidths":0.25,
                       "colors":"#808080",
                       "linestyles":"-"}

    net_sc_defaults = {"linewidths":0.25,
                       "colors":"#808080",
                       "linestyles":"-"}

    text_defaults = {"family": "sans-serif",
                     "size": "x-small",
                     "horizontalalignment": "center"}

    def __init__(self, axis=None, projection=project_equal_area, rotation=None):
        if rotation is not None:
            self.rotation = build_rotation_matrix(*rotation)
        else:
            self.rotation = None
        self.projection = projection
        if axis is None:
            from matplotlib import pyplot as plt
            self.axis = plt.gca()
            self.clear_diagram()
        else:
            self.axis = axis
        self.primitive = None

    def clear_diagram(self):
        """Clears the plot area and plot the primitive."""
        self.axis.cla()
        self.axis.axis('equal')
        self.axis.set_xlim(-1.1, 1.1)
        self.axis.set_ylim(-1.1, 1.1)
        self.axis.set_axis_off()
        self.plot_primitive()

    def plot_primitive(self):
        """Plots the primitive, center, NESW indicators and North if no
        rotation."""
        self.primitive = Circle((0, 0), radius=1, edgecolor='black',
                                fill=False, clip_box='None',
                                label='_nolegend_')
        self.axis.add_patch(self.primitive)
        # maybe add a dict for font options and such...
        if self.rotation is None:
            self.axis.text(0.01, 1.025, 'N', **self.text_defaults)
            x_cross = [0, 1, 0, -1, 0]
            y_cross = [0, 0, 1, 0, -1]
            self.axis.plot(x_cross, y_cross, 'k+', markersize=8,
                           label='_nolegend_')

    def project(self, data, invert_positive=True, rotate=True):
        if rotate and self.rotation is not None:
            return self.projection(self.rotate(data), invert_positive)
        else:
            return self.projection(data, invert_positive)

    def rotate(self, data):
        if self.rotation is not None:
            return self.rotation.dot(np.transpose(data)).T
        else:
            return data

    def plot_poles(self, vectors, **kwargs):
        """Plot points on the diagram. Accepts and passes aditional key word
        arguments to axis.plot."""
        X, Y = self.project(vectors)
        # use the default values if not user input
        # https://stackoverflow.com/a/6354485/1457481
        options = dict(self.point_defaults.items() + kwargs.items())
        self.axis.plot(X, Y, linestyle='', **options)

    def plot_circles(self, circles, **kwargs):
        """plot a list of circles, either great or small"""
        # use the default values if not user input
        # https://stackoverflow.com/a/6354485/1457481
        options = dict(self.circle_defaults.items() + kwargs.items())
        # should change this for better support of huge data
        projected_circles = [np.transpose(self.project(segment,
                                                       invert_positive=False,
                                                       rotate=False))
                             for circle in circles for segment in
                             clip_lines(self.rotate(circle))]
        circle_collection = LineCollection(projected_circles,
                                           **options)
        circle_collection.set_clip_path(self.primitive)
        self.axis.add_collection(circle_collection)

    def plot_contours(self, nodes, count, n_data, n_contours=10, minmax=True,
                      percentage=True, contour_mode='fillover', resolution=250,
                      **kwargs):
        """Plot contours of a spherical count. Parameters are the counting
        nodes, the actual counts and the number of data points. Returns the
        matplotlib contour object for creating colorbar."""
        if percentage:
            count = 100.*count/n_data
        if minmax:
            intervals = np.linspace(count.min(), count.max(), n_contours)
        else:
            intervals = np.linspace(0, count.max(), n_contours)
        xi = yi = np.linspace(-1.1, 1.1, resolution)
        # maybe preselect nodes here on z tolerance
        X, Y = self.project(nodes, invert_positive=False)
        zi = griddata(X, Y, count, xi, yi, interp='linear')
        # use the default values if not user input
        # https://stackoverflow.com/a/6354485/1457481
        options = dict(self.contour_defaults.items() + kwargs.items())

        contour_fill, contour_lines = None, None
        if contour_mode in ('fillover', 'fill'):
            contour_fill = self.axis.contourf(xi, yi, zi, intervals,
                                              **options)
            for collection in contour_fill.collections:
                collection.set_clip_path(self.primitive)
        if contour_mode != 'fill':
            contour_lines = self.axis.contour(xi, yi, zi, intervals, **options)
            for collection in contour_lines.collections:
                collection.set_clip_path(self.primitive)

        return contour_fill if contour_fill is not None else contour_lines

    def plot_text(self, vector, text, border=None, **kwargs):
        foreground = kwargs.pop("foreground", "w")
        options = dict(self.text_defaults.items() + kwargs.items())
        X, Y = self.project(vector)
        txt = self.axis.text(X, Y, text, **options)
        if border is not None:
            txt.set_path_effects([PathEffects.withStroke(
                linewidth=border,
                foreground=foreground)])

    def plot_net(self, gc_spacing=10., sc_spacing=10., n=360,
                  gc_options=None, sc_options=None, clean_caps=True,
                  plot_cardinal_points=True, cardinal_options=None):
        gc, sc = net_grid(gc_spacing, sc_spacing, n, clean_caps)
        gc_options = {} if gc_options is None else gc_options
        sc_options = {} if sc_options is None else sc_options
        cardinal_options = {} if cardinal_options is None else cardinal_options
        gc_options = dict(self.net_gc_defaults.items() + gc_options.items())
        self.plot_circles(gc, **gc_options)
        sc_options = dict(self.net_sc_defaults.items() + sc_options.items())
        self.plot_circles(sc, **sc_options)

        cardinal_options = dict([("verticalalignment", "center")]
                                + cardinal_options.items())
        if plot_cardinal_points and self.rotation is not None:
            cpoints = np.array(((0.0,1.0,0.0),
                                (1.0,0.0,0.0),
                                (0.0,-1.,0.0),
                                (-1.,0.0,0.0)))
            c_rotated = self.rotate(cpoints)
            for i, (point, name) in enumerate(zip(c_rotated, "NESW")):
                if point[2] > 0:
                    continue
                self.plot_text(cpoints[i], name, border=2.0, foreground='w',
                               **cardinal_options)

    def plot_slip_linear(self, planes, lines, sense=True, arrowsize=radians(10),
                         arrowcolor="#4D4D4D", footwall=False, **kwargs):
        options = dict(self.slip_defaults.items() + kwargs.items())
        for plane, line in zip(planes, lines):
            arrow_from = cos(arrowsize/2.)*plane + sin(arrowsize/2.)*line
            arrow_to = cos(-arrowsize/2.)*plane + sin(-arrowsize/2.)*line
            if footwall:
                arrow_from, arrow_to = arrow_to, arrow_from
            X, Y = self.project((arrow_from, arrow_to))
            if not sense:
                self.axis.add_line(Line2D(X, Y, c=arrowcolor,
                                          label='_nolegend_', **options))
            else:
                a, b = (X[0], Y[0]), (X[1], Y[1])
                self.axis.add_patch(FancyArrowPatch(a, b, shrinkA=0.0,
                    shrinkB=0.0, arrowstyle='->,head_length=2.5,head_width=1',
                    connectionstyle='arc3,rad=0.0', mutation_scale=2.0,
                    ec=arrowcolor, **options))

    def plot_slickenlines(self, planes, lines, sense=True,
                          arrowsize=radians(10), arrowcolor="#4D4D4D",
                          footwall=False, **kwargs):
        self.plot_slip_linear(lines, planes, sense, arrowsize, arrowcolor,
                              not footwall, **kwargs)
