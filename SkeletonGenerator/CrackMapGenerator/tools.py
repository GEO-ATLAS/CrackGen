import math
import numpy as np
#辅助函数
def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
def is_point_in_obstacle(point, obstacles):
    for (ox, oy, r) in obstacles:
        if distance(point,(ox,oy))<= r:
            return True
    return False

def generate_valid_point(obstacles, map_size):
    while True:
        point = (np.random.randint(0, map_size[0]), np.random.randint(0, map_size[1]))
        if not is_point_in_obstacle(point, obstacles):
            return point
        
def check_circle_intersect(obstacles,x,y,r):
    for o in obstacles:
        d=distance(o,(x,y))
        if d<=(o[2]+r):
            return True