#!/usr/bin/env bash
set -e

unset AMENT_PREFIX_PATH
unset COLCON_PREFIX_PATH
unset PYTHONPATH

source /opt/ros/humble/setup.bash

cd ~/yolotl
source .venv/bin/activate
python -c "import torch; print('venv torch:', torch.__version__)"

rm -rf build install log
colcon build --symlink-install --cmake-args -DPython3_EXECUTABLE="$(which python)"

source ~/yolotl/install/setup.bash
ros2 run yolotl_ros2 lane_follower


#실행전
cd ~/yolotl
source .venv/bin/activate
export PYTHONPATH="$HOME/yolotl/.venv/lib/python3.10/site-packages:$PYTHONPATH"
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 run yolotl_ros2 lane_follower

ros2 run usb_cam usb_cam_node_exe --ros-args -p video_device:=/dev/video2 -p pixel_format:=yuyv2rgb
ros2 bag play ~/Downloads/test13 -l
