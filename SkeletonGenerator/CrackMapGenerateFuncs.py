# #!/usr/bin/env python
# # -*- encoding: utf-8 -*-
import cv2
import numpy as np
import random
import math
from CrackMapGenerator.tools import *
from CrackMapGenerator.RRT import *
import os

def ensure_dir(directory):
    """确保目录存在，如果不存在则创建"""
    if not os.path.exists(directory):
        os.makedirs(directory)

def draw_obstacles(map_image, obstacles, color=(0, 255, 0)):
    """Draw obstacles with green circles"""
    vis_map = map_image.copy()
    for x, y, r in obstacles:
        cv2.circle(vis_map, (int(x), int(y)), int(r), color, 2)
        cv2.circle(vis_map, (int(x), int(y)), 1, (0, 0, 255), -1)
    return vis_map

def draw_points(map_image, points, start_color=(255, 0, 0), end_color=(0, 0, 255)):
    """Draw start and end points with different colors"""
    vis_map = map_image.copy()
    for start, end in points:
        # Start point
        cv2.circle(vis_map, (int(start[0]), int(start[1])), 4, start_color, -1)
        cv2.putText(vis_map, 'S', (int(start[0])+5, int(start[1])+5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, start_color, 1)
        
        # End point
        cv2.circle(vis_map, (int(end[0]), int(end[1])), 4, end_color, -1)
        cv2.putText(vis_map, 'E', (int(end[0])+5, int(end[1])+5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, end_color, 1)
    return vis_map

def single_crack_generate(start, goal, omap, obstacles, max_iter=5000, step_len=2, delta=0.5,
                         goal_sample_rate=0.2, showObs=True, node_color=[255,255,255], 
                         showMainPath=False, mode=1, save_steps=100):
    # 创建保存过程图像的目录
    process_dir = "generate_image_log/process"
    ensure_dir(process_dir)
    
    # 创建基础地图
    base_map = omap.copy()
    
    # 初始化 RRT
    rrt = RRT(start, goal, base_map, obstacles, max_iter=max_iter, 
                step_len=step_len, delta=delta, goal_sample_rate=goal_sample_rate,
                showObs=showObs, node_color=node_color, mode=mode)
    
    # 修改 planning 方法来保存过程
    def modified_planning(self):
        step_count = 0
        for i in range(self.max_iter):
            random_node = self.generate_random_node()
            if not self.is_collision_free(random_node):
                continue
                
            nearest_node = self.get_nearest_node(random_node)
            new_node = self.steer(nearest_node, random_node)
            
            if self.is_collision_free(new_node):
                # 绘制随机探索的点
                if self.mode == 1:
                # Mode 1: 单点模式
                # 只在探索点的确切位置绘制一个像素点
                # 产生细线条的效果，更适合模拟细小裂缝
                # 缺点是在大尺寸图像中可能不够明显
                    self.init_map[new_node.coord[1], new_node.coord[0]] = self.node_color
                else:
                                    # Mode 2: 圆形模式
                # 在探索点位置绘制一个实心圆
                # 圆的半径与步长相关，为 step_len/2（最小为1像素）
                # 产生更粗的、更容易看见的路径
                # 适合模拟较宽的裂缝或在大尺寸图像中展示路径
                    cv2.circle(self.init_map, new_node.coord, max(int(self.step_len/2),1), self.node_color, -1)
                
                # 保存过程图像
                if step_count < save_steps:
                    # 保存当前地图状态
                    filename = f"{process_dir}/step_{step_count:03d}.jpg"
                    cv2.imwrite(filename, self.init_map)
                    cv2.imshow('Crack Generation Process', self.init_map)
                    cv2.waitKey(1)
                    step_count += 1
                
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
        self.path = path
        return path

    # 替换原始的 planning 方法
    rrt.planning = modified_planning.__get__(rrt, RRT)
    
    # 运行规划
    rrt.planning()
    
    if showMainPath:
        rrt.plotPath([125,125,125])
    
    return rrt.init_map

def obstaclesGenerator(map_width, map_height, num_obstacles=100, max_r=20, min_r=2, aratio=-1, seed=2024):
    obstacles = []
    np.random.seed(seed)
    if aratio > 0 and aratio < 1:
        totalArea = map_height * map_width
        goalArea = aratio * totalArea
        area = 0
        while area < goalArea:
            x, y = np.random.randint(0, map_width), np.random.randint(0, map_height)
            r = np.random.randint(min_r, max_r)
            if len(obstacles) == 0:
                obstacles.append((x, y, r))
                area += (np.pi * r * r)
            else:
                if not check_circle_intersect(obstacles, x, y, r):
                    obstacles.append((x, y, r))
                    area += (np.pi * r * r)
    else:
        c = 0
        while c < num_obstacles:
            x, y = np.random.randint(0, map_width), np.random.randint(0, map_height)
            r = np.random.randint(min_r, max_r)
            if len(obstacles) == 0:
                obstacles.append((x, y, r))
                c += 1
            else:
                if not check_circle_intersect(obstacles, x, y, r):
                    obstacles.append((x, y, r))
                    c += 1
    return obstacles


if __name__ == "__main__":
    map_width = 512
    map_height = 512
    num_obstacles = 100
    np.random.seed(2024)
    
    # 初始化黑色背景
    init_map = np.zeros((map_width, map_height, 3), dtype=np.uint8)
    
    step_len = 1.414
    mode = 2
    
    # 生成障碍物
    obstacles = obstaclesGenerator(map_width, map_height, num_obstacles)
    
    # 设置起点和终点
    start = (251, 210)
    goal = (200, 500)
    
    # 生成裂缝地图并保存前100步过程
    showObs = True
    omap = single_crack_generate(start, goal, init_map, obstacles,
                               step_len=step_len, showObs=showObs,
                               mode=mode, showMainPath=True,
                               save_steps=2000)
    
    # 保存最终结果
    cv2.imwrite("generate_image_log/final_map.jpg", omap)
    
    # 显示最终结果
    cv2.imshow('Final Crack Map', omap)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 


