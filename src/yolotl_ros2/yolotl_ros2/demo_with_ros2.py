#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import glob
import numpy as np
import cv2
import torch
import argparse
from math import atan2, degrees, sqrt

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32, Bool
from ament_index_python.packages import get_package_share_directory
from ultralytics import YOLO

# Conda 라이브러리 충돌 방지
if 'CONDA_PREFIX' in os.environ:
    conda_site = glob.glob(os.path.join(os.environ['CONDA_PREFIX'], 'lib', 'python*', 'site-packages'))
    if conda_site: sys.path.insert(0, conda_site[0])

# --- 유틸리티 함수 (ROS 1 코드 원본 유지) ---
def polyfit_lane(points_y, points_x, order=2):
    if len(points_y) < 5: return None
    try: return np.polyfit(points_y, points_x, order)
    except: return None

def morph_close(binary_mask, ksize=5):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    return cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

def remove_small_components(binary_mask, min_size=300):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    cleaned = np.zeros_like(binary_mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_size: cleaned[labels == i] = 255
    return cleaned

def keep_top2_components(binary_mask, min_area=300):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    if num_labels <= 1: return np.zeros_like(binary_mask)
    comps = []
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area: comps.append((i, stats[i, cv2.CC_STAT_AREA]))
    comps.sort(key=lambda x: x[1], reverse=True)
    cleaned = np.zeros_like(binary_mask)
    for i in range(min(len(comps), 2)):
        idx = comps[i][0]
        cleaned[labels == idx] = 255
    return cleaned

def final_filter(bev_mask):
    f2 = morph_close(bev_mask, ksize=5)
    f3 = remove_small_components(f2, min_size=1000)
    f4 = keep_top2_components(f3, min_area=300)
    return f4

def overlay_polyline(image, coeff, color=(0, 0, 255), step=4, thickness=2):
    if coeff is None: return image
    h, w = image.shape[:2]
    draw_points = []
    for y in range(0, h, step):
        x = np.polyval(coeff, y)
        if 0 <= x < w: draw_points.append((int(x), int(y)))
    if len(draw_points) > 1:
        cv2.polylines(image, [np.array(draw_points, dtype=np.int32)], False, color, thickness)
    return image

class LaneFollowerNode(Node):
    def __init__(self, opt):
        super().__init__('lane_follower_node')
        self.opt = opt
        self.get_logger().info("[Visualizer] Launching Original ROS1 Style (3 Windows)...")

        # 1. 경로 설정
        try:
            pkg_share = get_package_share_directory('yolotl_ros2')
            weights_path = os.path.join(pkg_share, 'config', opt.weights)
            param_path = os.path.join(pkg_share, 'config', opt.param_file)
            if not os.path.exists(weights_path):
                weights_path = os.path.join(os.getcwd(), 'src/yolotl_ros2/config', opt.weights)
            if not os.path.exists(param_path):
                param_path = os.path.join(os.getcwd(), 'src/yolotl_ros2/config', opt.param_file)
        except:
            weights_path, param_path = opt.weights, opt.param_file

        # 2. 모델 로드
        self.device = torch.device('cuda:0' if torch.cuda.is_available() and opt.device != 'cpu' else 'cpu')
        self.model = YOLO(weights_path).to(self.device)

        # 3. BEV 파라미터
        try:
            params = np.load(param_path)
            self.bev_params = {'src_points': params['src_points'], 'dst_points': params['dst_points']}
            self.bev_w, self.bev_h = int(params['warp_w']), int(params['warp_h'])
        except Exception as e:
            self.get_logger().error(f"Param Error: {e}"); sys.exit(1)

        # 4. 파라미터 초기화
        self.m_per_pixel_y, self.y_offset_m, self.m_per_pixel_x = 0.0025, 1.25, 0.003578125
        self.tracked_lanes = {'left': {'coeff': None, 'age': 0}, 'right': {'coeff': None, 'age': 0}}
        self.tracked_center_path = {'coeff': None}
        self.SMOOTHING_ALPHA = 0.6
        self.MAX_LANE_AGE = 7
        self.L = 0.73
        self.THROTTLE_MIN, self.THROTTLE_MAX = 0.4, 0.6
        self.MIN_LOOKAHEAD_DISTANCE, self.MAX_LOOKAHEAD_DISTANCE = 1.75, 2.35
        self.current_throttle = self.THROTTLE_MIN

        # 5. ROS Setup
        self.pub_steering = self.create_publisher(Float32, 'auto_steer_angle_lane', 1)
        self.pub_lane_status = self.create_publisher(Bool, 'lane_detection_status', 1)
        self.sub_image = self.create_subscription(Image, '/usb_cam/image_raw', self.image_callback, 1)
        self.sub_throttle = self.create_subscription(Float32, 'auto_throttle', self.throttle_callback, 1)

    def throttle_callback(self, msg):
        self.current_throttle = np.clip(msg.data, self.THROTTLE_MIN, self.THROTTLE_MAX)

    def do_bev_transform(self, image):
        M = cv2.getPerspectiveTransform(self.bev_params['src_points'], self.bev_params['dst_points'])
        return cv2.warpPerspective(image, M, (self.bev_w, self.bev_h), flags=cv2.INTER_LINEAR)

    def image_to_vehicle(self, pt_bev):
        u, v = pt_bev
        x_vehicle = (self.bev_h - v) * self.m_per_pixel_y + self.y_offset_m
        y_vehicle = (self.bev_w / 2 - u) * self.m_per_pixel_x
        return x_vehicle, y_vehicle

    def image_callback(self, msg):
        try:
            # YUYV 포맷 에러 방지용 변환 코드
            np_arr = np.frombuffer(msg.data, dtype=np.uint8)
            if np_arr.size == (msg.width * msg.height * 2): 
                img = cv2.cvtColor(np_arr.reshape((msg.height, msg.width, 2)), cv2.COLOR_YUV2BGR_YUYV)
            elif np_arr.size == (msg.width * msg.height * 3):
                img = np_arr.reshape((msg.height, msg.width, 3))
                if 'rgb' in msg.encoding.lower(): img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            else: return
            self.process_image(np.ascontiguousarray(img))
        except Exception as e:
            self.get_logger().error(f"Img Error: {e}")

    def process_image(self, im0s):
        # [변수 초기화] 에러 방지
        steer_deg = None
        goal_point_bev = None
        is_detected = False
        final_l, final_r = None, None

        # 1. BEV Transform & Inference
        bev_image_input = self.do_bev_transform(im0s)
        results = self.model(bev_image_input, imgsz=self.opt.img_size, conf=self.opt.conf_thres, 
                            iou=self.opt.iou_thres, device=self.device, verbose=False)
        result = results[0]

        # 2. Masking
        combined_mask_bev = np.zeros(result.orig_shape[:2], dtype=np.uint8)
        if result.masks is not None:
            conf = result.boxes.conf
            for i, mask_tensor in enumerate(result.masks.data):
                if conf[i] >= 0.5:
                    m = (mask_tensor.cpu().numpy() * 255).astype(np.uint8)
                    if m.shape != result.orig_shape[:2]:
                        m = cv2.resize(m, (result.orig_shape[1], result.orig_shape[0]))
                    combined_mask_bev = np.maximum(combined_mask_bev, m)
        
        final_mask = final_filter(combined_mask_bev)
        bev_im_for_drawing = bev_image_input.copy()

        # 3. Detect & Track Lanes
        num, labels, stats, _ = cv2.connectedComponentsWithStats(final_mask, connectivity=8)
        dets = []
        if num > 1:
            for i in range(1, num):
                if stats[i, cv2.CC_STAT_AREA] >= 100:
                    ys, xs = np.where(labels == i)
                    coeff = polyfit_lane(ys, xs)
                    if coeff is not None:
                        dets.append({'coeff': coeff, 'x_bottom': np.polyval(coeff, self.bev_h - 1)})
        dets.sort(key=lambda c: c['x_bottom'])

        # Tracking Logic
        l_trk, r_trk = self.tracked_lanes['left'], self.tracked_lanes['right']
        cur_l, cur_r = None, None
        
        if len(dets) == 2: cur_l, cur_r = dets[0], dets[1]
        elif len(dets) == 1:
            d = dets[0]
            dl = abs(d['x_bottom'] - np.polyval(l_trk['coeff'], self.bev_h-1)) if l_trk['coeff'] is not None else float('inf')
            dr = abs(d['x_bottom'] - np.polyval(r_trk['coeff'], self.bev_h-1)) if r_trk['coeff'] is not None else float('inf')
            if dl < dr: cur_l = d
            else: cur_r = d
            if l_trk['coeff'] is None and r_trk['coeff'] is None:
                if d['x_bottom'] < self.bev_w/2: cur_l = d
                else: cur_r = d

        for trk, cur in zip([l_trk, r_trk], [cur_l, cur_r]):
            if cur:
                trk['coeff'] = cur['coeff'] if trk['coeff'] is None else (self.SMOOTHING_ALPHA * cur['coeff'] + (1-self.SMOOTHING_ALPHA)*trk['coeff'])
                trk['age'] = 0
            else:
                trk['age'] += 1
                if trk['age'] > self.MAX_LANE_AGE: trk['coeff'] = None

        final_l, final_r = l_trk['coeff'], r_trk['coeff']
        is_detected = (final_l is not None) or (final_r is not None)
        self.pub_lane_status.publish(Bool(data=is_detected))

        # 4. Steering Logic
        if is_detected:
            c_pts = []
            lw_px = 1.5 / self.m_per_pixel_x
            for y in range(self.bev_h-1, self.bev_h//2, -1):
                xc = None
                if final_l is not None and final_r is not None: xc = (np.polyval(final_l, y) + np.polyval(final_r, y))/2
                elif final_l is not None: xc = np.polyval(final_l, y) + lw_px/2
                elif final_r is not None: xc = np.polyval(final_r, y) - lw_px/2
                if xc is not None: c_pts.append([xc, y])
            
            tgt_coeff = polyfit_lane(np.array(c_pts)[:,1], np.array(c_pts)[:,0]) if len(c_pts)>10 else None
            
            if tgt_coeff is not None:
                ct = self.tracked_center_path
                ct['coeff'] = tgt_coeff if ct['coeff'] is None else (self.SMOOTHING_ALPHA*tgt_coeff + (1-self.SMOOTHING_ALPHA)*ct['coeff'])
            
            if self.tracked_center_path['coeff'] is not None:
                norm_thr = (self.current_throttle - self.THROTTLE_MIN)/(self.THROTTLE_MAX - self.THROTTLE_MIN + 1e-6)
                lookahead = self.MIN_LOOKAHEAD_DISTANCE + (self.MAX_LOOKAHEAD_DISTANCE - self.MIN_LOOKAHEAD_DISTANCE)*norm_thr
                
                for yb in range(self.bev_h-1, -1, -1):
                    xb = np.polyval(self.tracked_center_path['coeff'], yb)
                    xv, yv = self.image_to_vehicle((xb, yb))
                    if sqrt(xv**2 + yv**2) >= lookahead:
                        goal_point_bev = (int(xb), int(yb))
                        rad = atan2(2.0*self.L*yv, xv**2 + yv**2)
                        steer_deg = np.clip(-degrees(rad), -25.0, 25.0)
                        self.pub_steering.publish(Float32(data=steer_deg))
                        break

        # 5. [시각화] 창 3개, 리사이즈 없음 (NO RESIZE, NO BLACK BARS)
        # 이미지 크기 그대로 출력
        
        # [Window 1] 원본 카메라 (640x480)
        cv2.imshow("Original Camera View", im0s)
        
        # [Window 2] 탐지 결과 (BEV 크기 그대로)
        annotated_bev = result.plot()
        cv2.imshow("Roboflow Detections (on BEV)", annotated_bev)
        
        # [Window 3] 최종 경로 (BEV 크기 그대로)
        if final_l is not None: overlay_polyline(bev_im_for_drawing, final_l, (255, 0, 0), step=2) 
        if final_r is not None: overlay_polyline(bev_im_for_drawing, final_r, (0, 0, 255), step=2) 
        if self.tracked_center_path['coeff'] is not None:
            overlay_polyline(bev_im_for_drawing, self.tracked_center_path['coeff'], (0, 255, 0), step=2, thickness=3) 

        if goal_point_bev is not None:
            cv2.circle(bev_im_for_drawing, goal_point_bev, 10, (0, 255, 255), -1)

        # 텍스트 정보
        s_txt = f"Steer: {steer_deg:.1f} deg" if steer_deg is not None else "Steer: N/A"
        l_val = self.MIN_LOOKAHEAD_DISTANCE + (self.MAX_LOOKAHEAD_DISTANCE - self.MIN_LOOKAHEAD_DISTANCE)*((self.current_throttle-self.THROTTLE_MIN)/(self.THROTTLE_MAX-self.THROTTLE_MIN+1e-6))
        
        cv2.putText(bev_im_for_drawing, s_txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(bev_im_for_drawing, f"Lane Detected: {is_detected}", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(bev_im_for_drawing, f"Lookahead: {l_val:.2f}m", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(bev_im_for_drawing, f"Throttle: {self.current_throttle:.2f}", (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        cv2.imshow("Final Path & Logs (on BEV)", bev_im_for_drawing)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default='weights2.pt')
    parser.add_argument('--param-file', default='bev_params_y_5.npz')
    parser.add_argument('--img-size', type=int, default=640)
    parser.add_argument('--conf-thres', type=float, default=0.6)
    parser.add_argument('--iou-thres', type=float, default=0.5)
    parser.add_argument('--device', default='0')
    opt, _ = parser.parse_known_args()
    
    node = LaneFollowerNode(opt)
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        cv2.destroyAllWindows()
        rclpy.shutdown()

if __name__ == '__main__':
    main()