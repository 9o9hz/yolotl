#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Utility Script: BEV (Birds-Eye View) Parameter Setup - Y-Axis Auto Alignment
---------------------------------------------------
ROS 2 호환 버전: 4개의 점을 선택하되, 우측 점들의 y좌표를 좌측 점과 동일하게 자동 정렬합니다.
"""

import cv2
import numpy as np
import argparse
import os

# 전역 변수: 선택된 4개의 좌표
src_points = []
max_points = 4

def mouse_callback(event, x, y, flags, param):
    """
    마우스 클릭 이벤트를 처리합니다.
    우측 점(2번째, 4번째) 클릭 시 y좌표를 이전 좌측 점의 y좌표로 고정합니다.
    """
    global src_points
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(src_points) < max_points:
            point_order = ["Left-Bottom", "Right-Bottom", "Left-Top", "Right-Top"]
            current_point_index = len(src_points)
            final_point = (x, y)

            # 우측 점(Index 1, 3) 클릭 시 y좌표 자동 정렬 로직
            if current_point_index == 1:   # Right-Bottom 선택 시
                y_bottom = src_points[0][1] # Left-Bottom의 y좌표 가져오기
                final_point = (x, y_bottom)
            elif current_point_index == 3: # Right-Top 선택 시
                y_top = src_points[2][1]    # Left-Top의 y좌표 가져오기
                final_point = (x, y_top)

            src_points.append(final_point)
            print(f"[INFO] Added {point_order[current_point_index]} point: {final_point} ({len(src_points)}/{max_points})")

            if len(src_points) == max_points:
                print("[INFO] 모든 점 선택 완료. 's'를 눌러 저장하거나 'r'로 리셋하세요.")
        else:
            print("[WARNING] 이미 4개의 점이 선택되었습니다. 'r'을 눌러 리셋하세요.")

def run_bev_y_auto_setup(source='0', warp_w=640, warp_h=640, out_npz='bev_params_y_auto.npz', out_txt='selected_points_y_auto.txt'):
    global src_points
    src_points = [] # 초기화
    
    is_image = False
    cap = None
    static_img = None

    # 입력 소스 판별 (카메라 번호 또는 파일 경로)
    if source.isdigit():
        cap = cv2.VideoCapture(int(source), cv2.CAP_V4L2)
        if not cap.isOpened():
            print(f"[ERROR] 카메라를 열 수 없습니다: {source}")
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        print(f"[INFO] 실시간 웹캠 모드 (1280x720)")
    else:
        ext = os.path.splitext(source)[1].lower()
        if ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            static_img = cv2.imread(source)
            is_image = True
            print(f"[INFO] 단일 이미지 모드: {source}")
        else:
            cap = cv2.VideoCapture(source)
            print(f"[INFO] 비디오 파일 모드: {source}")

    dst_points = np.float32([
        [0,       warp_h],    # Bottom-left
        [warp_w,  warp_h],    # Bottom-right
        [0,       0],         # Top-left
        [warp_w,  0]          # Top-right
    ])

    cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
    cv2.namedWindow("BEV", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Original", mouse_callback)

    print("\n[사용 방법 - Y축 자동 정렬 모드]")
    print("1. 클릭 순서: 좌하 -> 우하(y고정) -> 좌상 -> 우상(y고정)")
    print("2. 's' 키: 저장 후 종료 / 'r' 키: 리셋 / 'q' 키: 취소\n")

    while True:
        if is_image:
            frame = static_img.copy()
        else:
            ret, frame = cap.read()
            if not ret:
                break
        
        disp = frame.copy()
        point_labels = ["1 (L-Bot)", "2 (R-Bot)", "3 (L-Top)", "4 (R-Top)"]
        
        for i, pt in enumerate(src_points):
            cv2.circle(disp, pt, 5, (0, 255, 0), -1)
            cv2.putText(disp, point_labels[i], (pt[0]+5, pt[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
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
    parser.add_argument('--out-npz', type=str, default='bev_params_7.npz')
    parser.add_argument('--out-txt', type=str, default='selected_bev_src_points_7.txt')
    args = parser.parse_args()
    
    run_bev_y_auto_setup(args.source, args.warp_width, args.warp_height, args.out_npz, args.out_txt)