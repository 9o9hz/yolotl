
#실행전
ros2 run yolotl_ros2 lane_follower



카메라/백파일 input
ros2 run usb_cam usb_cam_node_exe --ros-args -p video_device:=/dev/video2 -p pixel_format:=yuyv2rgb
ros2 bag play ~/Downloads/test13 -l



토픽 list
j:~$ ros2 topic list
/auto_steer_angle_lane
/auto_throttle
/drivable_area
/events/read_split
/image_raw/compressed
/lane_detection_status
/lane_path
/lookahead_distance
/parameter_events
/rosout
j:~$ 
