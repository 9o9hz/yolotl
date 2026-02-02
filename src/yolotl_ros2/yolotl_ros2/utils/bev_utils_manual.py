#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Utility Script: BEV (Birds-Eye View) Parameter Setup - Manual Version
---------------------------------------------------
ROS 2 호환 버전: 수동으로 4개의 점을 선택하여 BEV 파라미터를 생성합니다.
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
    마우스 클릭 이벤트를 처리하여 순차적으로 4개의 점을 받습니다.
    """
    global src_points
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(src_points) < max_points:
            src_points.append((x, y))
            point_order = ["Left-Bottom", "Right-Bottom", "Left-Top", "Right-Top"]
            current_point_index = len(src_points) - 1
            print(f"[INFO] Added {point_order[current_point_index]} point: ({x}, {y}) ({len(src_points)}/{max_points})")

            if len(src_points) == max_points:
                print("[INFO] 모든 점이 선택되었습니다. 's'를 눌러 저장하거나 'r'로 리셋하세요.")
        else:
            print("[WARNING] 이미 4개의 점이 선택되었습니다. 리셋하려면 'r'을 누르세요.")

def run_bev_manual_setup(source='0', warp_w=640, warp_h=640, out_npz='bev_params_manual.npz', out_txt='selected_points_manual.txt'):
    global src_points
    src_points = [] # 함수 호출 시 좌표 초기화
    
    is_image = False
    cap = None
    static_img = None

    # 입력 소스 판별
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
        if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
            static_img = cv2.imread(source)
            if static_img is None:
                print(f"[ERROR] 이미지 파일을 열 수 없습니다: {source}")
                return
            is_image = True
            print(f"[INFO] 단일 이미지 모드: {source}")
        else:
            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                print(f"[ERROR] 비디오 파일을 열 수 없습니다: {source}")
                return
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

    print("\n[사용 방법]")
    print("1. 클릭 순서: 좌하 -> 우하 -> 좌상 -> 우상")
    print("2. 'r' 키: 좌표 리셋")
    print("3. 's' 키: 파라미터 저장 및 종료")
    print("4. 'q' 키: 저장하지 않고 종료\n")

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
            label = point_labels[i] if i < len(point_labels) else f"{i+1}"
            cv2.putText(disp, label, (pt[0]+5, pt[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
        if len(src_points) == 4:
            cv2.polylines(disp, [np.array(src_points, dtype=np.int32)], True, (0,0,255), 2)
            M = cv2.getPerspectiveTransform(np.float32(src_points), dst_points)
            bev_result = cv2.warpPerspective(frame, M, (warp_w, warp_h))
            cv2.imshow("BEV", bev_result)

        cv2.imshow("Original", disp)
        key = cv2.waitKey(30) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('r'):
            src_points = []
        elif key == ord('s'):
            if len(src_points) < 4:
                print("[WARNING] 4개의 점을 모두 선택해야 저장 가능합니다.")
            else:
                np.savez(out_npz, src_points=np.float32(src_points), dst_points=dst_points, warp_w=warp_w, warp_h=warp_h)
                try:
                    with open(out_txt, 'w') as f:
                        f.write("# Selected BEV Points\n")
                        for i, pt in enumerate(src_points):
                            f.write(f"{pt[0]}, {pt[1]} # {point_labels[i]}\n")
                    print(f"[INFO] 저장 완료: {out_npz}, {out_txt}")
                except Exception as e:
                    print(f"[ERROR] 파일 저장 중 오류 발생: {e}")
                break

    if cap: cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='0')
    parser.add_argument('--warp-width', type=int, default=640)
    parser.add_argument('--warp-height', type=int, default=640)
    parser.add_argument('--out-npz', type=str, default='bev_params_manual.npz')
    parser.add_argument('--out-txt', type=str, default='selected_bev_src_points_manual.txt')
    args = parser.parse_args()
    
    run_bev_manual_setup(args.source, args.warp_width, args.warp_height, args.out_npz, args.out_txt)