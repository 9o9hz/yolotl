#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import glob
import numpy as np
import torch
import cv2
import rclpy
from rclpy.node import Node
import argparse
from ultralytics import YOLO
from sensor_msgs.msg import Image
from std_msgs.msg import Float32, Bool
from ament_index_python.packages import get_package_share_directory

# [시스템 설정] Conda 환경 우선 로드
if 'CONDA_PREFIX' in os.environ:
    conda_site = glob.glob(os.path.join(os.environ['CONDA_PREFIX'], 'lib', 'python*', 'site-packages'))
    if conda_site:
        sys.path.insert(0, conda_site[0])

class AdvancedLaneFollower(Node):
    def __init__(self, opt):
        super().__init__('advanced_lane_follower')
        self.opt = opt
        
        # 1. Model & Device Setup
        self.device = torch.device('cuda:0' if torch.cuda.is_available() and opt.device != 'cpu' else 'cpu')
        self.model = YOLO(opt.weights)
        self.model.to(self.device)
        
        # 2. BEV Params (0.004... 값들은 이전 코드에서 가져옴)
        self.m_per_pixel_y = 0.004015625
        self.y_offset_m = 1.83
        self.m_per_pixel_x = 0.00278125
        self.wheelbase = 0.75
        self.lookahead = 2.10

        try:
            params = np.load(opt.param_file)
            self.M = cv2.getPerspectiveTransform(params['src_points'], params['dst_points'])
            self.bev_w, self.bev_h = int(params['warp_w']), int(params['warp_h'])
        except:
            self.get_logger().error("BEV Param file not found!")
            sys.exit(1)

        # 3. ROS Pub/Sub
        self.pub_steering = self.create_publisher(Float32, 'auto_steer_angle_lane', 1)
        self.sub_image = self.create_subscription(Image, '/usb_cam/image_raw', self.image_callback, 1)

    def image_to_vehicle(self, u, v):
        x_v = (self.bev_h - v) * self.m_per_pixel_y + self.y_offset_m
        y_v = (self.bev_w / 2 - u) * self.m_per_pixel_x
        return x_v, y_v

    def image_callback(self, msg):
        try:
            # ROS 2 Image Message to OpenCV (Manual conversion)
            np_arr = np.frombuffer(msg.data, dtype=np.uint8)
            if np_arr.size == (msg.width * msg.height * 2): # YUYV
                img = cv2.cvtColor(np_arr.reshape((msg.height, msg.width, 2)), cv2.COLOR_YUV2BGR_YUYV)
            else:
                img = np_arr.reshape((msg.height, msg.width, 3))
                if 'rgb' in msg.encoding.lower(): img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            self.process_advanced(img)
        except Exception as e:
            self.get_logger().error(f"Callback error: {e}")

    def process_advanced(self, img):
        # 1. Inference (Detect First)
        results = self.model(img, conf=self.opt.conf_thres, verbose=False)
        result = results[0]
        
        # 2. Make Mask & BEV
        mask = np.zeros(result.orig_shape[:2], dtype=np.uint8)
        if result.masks is not None:
            for m_tensor in result.masks.data:
                m_np = (m_tensor.cpu().numpy() * 255).astype(np.uint8)
                mask = np.maximum(mask, cv2.resize(m_np, (img.shape[1], img.shape[0])))
        
        bev_mask = cv2.warpPerspective(mask, self.M, (self.bev_w, self.bev_h))
        bev_img = cv2.warpPerspective(img, self.M, (self.bev_w, self.bev_h))
        
        # 3. Path Planning (Pure Pursuit)
        steering_angle = 0.0
        goal_pt = None
        
        # Thinning & Connected Components (이전 코드 로직)
        skeleton = cv2.ximgproc.thinning(bev_mask) if hasattr(cv2, 'ximgproc') else bev_mask
        ys, xs = np.where(skeleton > 0)
        
        if len(ys) > 10:
            coeff = np.polyfit(ys, xs, 2)
            # Find Goal Point (Lookahead)
            min_dist = float('inf')
            for y in range(0, self.bev_h, 5):
                x = np.polyval(coeff, y)
                if 0 <= x < self.bev_w:
                    xv, yv = self.image_to_vehicle(x, y)
                    dist = np.sqrt(xv**2 + yv**2)
                    if abs(dist - self.lookahead) < min_dist:
                        min_dist = abs(dist - self.lookahead)
                        goal_pt = (int(x), int(y))
                        xv_g, yv_g = xv, yv
            
            if goal_pt:
                alpha = np.arctan2(yv_g, xv_g)
                ld = np.sqrt(xv_g**2 + yv_g**2)
                steering_angle = np.degrees(np.arctan((2 * self.wheelbase * np.sin(alpha)) / ld))
                cv2.circle(bev_img, goal_pt, 10, (0, 255, 0), -1)
                cv2.line(bev_img, (self.bev_w//2, self.bev_h), goal_pt, (0, 255, 0), 2)

        self.pub_steering.publish(Float32(data=float(steering_angle)))

        # 4. Dashboard Visualization (상단 원본 / 하단 2분할)
        p1 = cv2.resize(result.plot(), (self.bev_w*2, int(img.shape[0]*(self.bev_w*2/img.shape[1]))))
        p2 = bev_mask
        if len(p2.shape) == 2: p2 = cv2.cvtColor(p2, cv2.COLOR_GRAY2BGR)
        p3 = bev_img
        
        bottom = np.hstack((cv2.resize(p2, (self.bev_w, self.bev_h)), p3))
        dashboard = np.vstack((p1, bottom))
        
        cv2.putText(dashboard, f"Steer: {steering_angle:.2f} deg", (20, 60), 1, 2, (0, 255, 255), 2)
        cv2.imshow("Advanced Dashboard", dashboard)
        cv2.waitKey(1)

def main():
    rclpy.init()
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default='weights2.pt')
    parser.add_argument('--param-file', default='bev_params_y_5.npz')
    parser.add_argument('--conf-thres', type=float, default=0.3)
    parser.add_argument('--device', default='0')
    opt, _ = parser.parse_known_args()
    
    node = AdvancedLaneFollower(opt)
    rclpy.spin(node)
    cv2.destroyAllWindows()
    rclpy.shutdown()

if __name__ == '__main__':
    main()