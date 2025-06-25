import numpy as np
from .tools import *
import cv2

class Node:
    def __init__(self, coord):
        self.coord = coord
        self.parent = None
        self.x=coord[0]
        self.y=coord[1]

class RRT:
    def __init__(self, start, goal,omap:np.ndarray,obstacles,max_iter=5000,step_len=2,delta=0.5,
                 goal_sample_rate=0.2,showObs=True,showStartGoal=True,
                 node_color=[255,255,0],obs_color=[0,0,255],showWin=False,mode=1,seed=None):
        if seed:
            assert isinstance(seed,int),'seed must be int'
            self.seed=seed
            np.random.seed(seed)
        self.s_start = Node(start)
        self.s_goal = Node(goal)
        self.map_size=omap.shape[:2]
        self.x_range = (0,self.map_size[0]-1)
        self.y_range = (0,self.map_size[1]-1)
        self.delta = delta
        self.nodes = [self.s_start]
        self.obstacles=obstacles
        self.max_iter=max_iter
        self.step_len=step_len
        self.goal_sample_rate=goal_sample_rate
        self.init_map =omap
        self.path=None
        self.node_color=node_color
        self.obs_color=obs_color
        self.showWin=showWin
        self.mode=mode

        if showObs:
            for obs in obstacles:
                cv2.circle(self.init_map, (obs[0],obs[1]), obs[2], obs_color, -1)  # 红色
        if showStartGoal:
            # 原来的起点终点标记
            size = 25
            cv2.drawMarker(self.init_map, start, (0, 255, 0),
                        markerType=cv2.MARKER_CROSS, markerSize=size, thickness=4)
            cv2.drawMarker(self.init_map, goal, (255, 255, 0),
                        markerType=cv2.MARKER_TRIANGLE_UP, markerSize=size, thickness=4)
            
            # 添加图例
            legend_x = 10  # 图例起始x坐标
            legend_y = 10  # 图例起始y坐标
            legend_spacing = 30  # 图例项之间的垂直间距
            
            # 绘制图例框
            padding = 10
            legend_height = 3 * legend_spacing + padding * 2
            legend_width = 200
            cv2.rectangle(self.init_map, 
                        (legend_x - padding, legend_y - padding),
                        (legend_x + legend_width, legend_y + legend_height),
                        (255, 255, 255), -1)  # 白色背景
            cv2.rectangle(self.init_map,
                        (legend_x - padding, legend_y - padding),
                        (legend_x + legend_width, legend_y + legend_height),
                        (0, 0, 0), 1)  # 黑色边框

            # 障碍物图例
            cv2.circle(self.init_map, (legend_x + 15, legend_y + 15), 10, obs_color, -1)
            cv2.putText(self.init_map, "Obstacle", (legend_x + 40, legend_y + 30),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 1)

            # 起点图例
            cv2.drawMarker(self.init_map, (legend_x + 15, legend_y + 15 + legend_spacing),
                        (0, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
            cv2.putText(self.init_map, "Start", (legend_x + 40, legend_y + 30 + legend_spacing),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 1)

            # 终点图例
            cv2.drawMarker(self.init_map, (legend_x + 15, legend_y + 15 + 2 * legend_spacing),
                        (255, 255, 0), markerType=cv2.MARKER_TRIANGLE_UP, markerSize=20, thickness=2)
            cv2.putText(self.init_map, "Goal", (legend_x + 40, legend_y + 30 + 2 * legend_spacing),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 1)
            
    
    def distance(self, p1, p2):
        return distance(p1,p2)

    def generate_random_node(self):
        if np.random.random() > self.goal_sample_rate:
            return Node((np.random.uniform(self.x_range[0] + self.delta, self.x_range[1] - self.delta),
                         np.random.uniform(self.y_range[0] + self.delta, self.y_range[1] - self.delta)))

        return self.s_goal

    def is_collision_free(self, node):
        for (ox, oy, r) in self.obstacles:
            if self.distance((ox, oy), node.coord) <= r+self.delta:
                return False
        return True

    def get_nearest_node(self, node):
        '''获取最近点'''
        nearest_node = self.nodes[0]
        min_dist = self.distance(node.coord, nearest_node.coord)
        for n in self.nodes:
            d = self.distance(node.coord, n.coord)
            if d < min_dist:
                nearest_node = n
                min_dist = d
        return nearest_node
        
    @staticmethod
    def get_distance_and_angle(node_start, node_end):
        dx = node_end.coord[0] - node_start.coord[0]
        dy = node_end.coord[1] - node_start.coord[1]
        return math.hypot(dx, dy), math.atan2(dy, dx)
    
    def steer(self, node_start, node_end):
        dist, theta = self.get_distance_and_angle(node_start, node_end)

        dist = min(self.step_len, dist)
        node_new = Node((int(node_start.x + dist * math.cos(theta)),
                         int(node_start.y + dist * math.sin(theta))))
        node_new.parent = node_start

        return node_new

    def planning(self):
        for i in range(self.max_iter):
            random_node = self.generate_random_node()
            if not self.is_collision_free(random_node):
                #随机点发生碰撞
                continue
            nearest_node = self.get_nearest_node(random_node)
            new_node = self.steer(nearest_node, random_node)
            if self.is_collision_free(new_node):
                # 绘制随机探索的点
                #调换xy位置
                if self.mode==1:
                    self.init_map[new_node.coord[1], new_node.coord[0]] = self.node_color  # 表示随机探索的点
                else:
                    cv2.circle(self.init_map, new_node.coord, max(int(self.step_len/2),1), self.node_color, -1)  # 用圆绘制随机点
                if self.showWin:
                    cv2.imshow('Map', self.init_map)
                    cv2.waitKey(1)
                self.nodes.append(new_node)
                if self.distance(new_node.coord, self.s_goal.coord) < 1.5:
                    self.s_goal.parent = new_node
                    self.nodes.append(self.s_goal)
                    break

        path = []
        node = self.s_goal
        while node.parent is not None:
            path.append(node.coord)
            node = node.parent
        path.append(self.s_start.coord)
        path.reverse()
        self.path=path
        return path
    
    def plotPath(self,color:list=[255, 255, 255]):
        if self.path:
            for (x, y) in self.path:
                self.init_map[y, x] = color  # 路径