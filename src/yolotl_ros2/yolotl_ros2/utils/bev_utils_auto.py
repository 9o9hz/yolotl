#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import argparse
import os

# 전역 변수 설정
src_points = []
max_points = 4
current_frame_width = 0 

def mouse_callback(event, x, y, flags, param):
    global src_points, current_frame_width
    if event == cv2.EVENT_LBUTTONDOWN:
        if current_frame_width == 0:
            print("[WARNING] 프레임 너비가 설정되지 않았습니다.")
            return

        if len(src_points) < max_points:
            if len(src_points) == 0: # 첫 번째 클릭 (좌하단)
                src_points.append((x, y))
                symmetric_x = current_frame_width - 1 - x
                src_points.append((symmetric_x, y))
                print(f"[INFO] 좌하단 추가: ({x}, {y}), 우하단 자동추가: ({symmetric_x}, {y})")
            elif len(src_points) == 2: # 세 번째 클릭 (좌상단)
                src_points.append((x, y))
                symmetric_x = current_frame_width - 1 - x
                src_points.append((symmetric_x, y))
                print(f"[INFO] 좌상단 추가: ({x}, {y}), 우상단 자동추가: ({symmetric_x}, {y})")
                print("[INFO] 4개 점 선택 완료. 's'를 눌러 저장하거나 'r'로 초기화하세요.")
        else:
            print("[WARNING] 이미 4개의 점이 선택되었습니다.")

def run_bev_setup(source='0', warp_w=640, warp_h=640, out_npz='bev_params.npz', out_txt='selected_points.txt'):
    global src_points, current_frame_width
    src_points = [] # 초기화
    
    is_image = False
    cap = None
    static_img = None

    # 소스 판별 로직
    if source.isdigit():
        cap = cv2.VideoCapture(int(source), cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        current_frame_width = 1280
    else:
        ext = os.path.splitext(source)[1].lower()
        if ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            static_img = cv2.imread(source)
            is_image = True
            current_frame_width = static_img.shape[1]
        else:
            cap = cv2.VideoCapture(source)
            current_frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    dst_points = np.float32([[0, warp_h], [warp_w, warp_h], [0, 0], [warp_w, 0]])

    cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
    cv2.namedWindow("BEV", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Original", mouse_callback)

    while True:
        if is_image:
            frame = static_img.copy()
        else:
            ret, frame = cap.read()
            if not ret: break
        
        disp = frame.copy()
        for i, pt in enumerate(src_points):
            cv2.circle(disp, pt, 5, (0, 255, 0), -1)
        
        if len(src_points) == 4:
            cv2.polylines(disp, [np.array(src_points, dtype=np.int32)], True, (0,0,255), 2)
            M = cv2.getPerspectiveTransform(np.float32(src_points), dst_points)
            bev_result = cv2.warpPerspective(frame, M, (warp_w, warp_h))
            cv2.imshow("BEV", bev_result)

        cv2.imshow("Original", disp)
        key = cv2.waitKey(30) & 0xFF
        
        if key == ord('q'): break
        elif key == ord('r'): src_points = []
        elif key == ord('s') and len(src_points) == 4:
            np.savez(out_npz, src_points=np.float32(src_points), dst_points=dst_points, warp_w=warp_w, warp_h=warp_h)
            with open(out_txt, 'w') as f:
                for pt in src_points: f.write(f"{pt[0]}, {pt[1]}\n")
            print(f"[INFO] 저장 완료: {out_npz}")
            break

    if cap: cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='0')
    parser.add_argument('--warp-width', type=int, default=640)
    parser.add_argument('--warp-height', type=int, default=640)
    parser.add_argument('--out-npz', type=str, default='bev_params_3.npz')
    parser.add_argument('--out-txt', type=str, default='selected_bev_src_points_3.txt')
    args = parser.parse_args()
    
    run_bev_setup(args.source, args.warp_width, args.warp_height, args.out_npz, args.out_txt)