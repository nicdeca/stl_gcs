from matplotlib import pyplot as plt
import numpy as np

np.random.seed(3)

# from stl_tool.stl import GOp, FOp, TasksOptimizer, ContinuousLinearSystem, BoxBound2d
from stl_tool.environment import Map
from stl_tool.polyhedron import Box2d

from stl_tool.planners import StlRRTStar
from json import loads
import os


##########################################################
# Create work space and mapo
##########################################################
workspace = Box2d(
    x=0, y=0, size=10
)  # square 10x10 (this is a 2d workspace, so the system it refers to must be 2d)
map = Map(
    workspace=workspace
)  # the map object contains the workpace, but it also contains the obstacles of your environment.

# load obstacles

# create obstacles
# some simple boxes
map.add_obstacle(Box2d(x=3, y=3, size=1))
map.add_obstacle(Box2d(x=4, y=-4, size=1))
map.add_obstacle(Box2d(x=-2, y=-1, size=1))
map.add_obstacle(Box2d(x=0, y=-2, size=1.5))
# you can create some rectangles even
map.add_obstacle(Box2d(x=0, y=3, size=[1, 3]))

map.draw()
