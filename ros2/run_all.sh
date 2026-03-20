#!/usr/bin/env bash
set -e

source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=0

need_pkg() { dpkg -s "$1" >/dev/null 2>&1 || echo "$1"; }

PKGS=()
for p in \
  tmux \
  ros-humble-image-transport \
  ros-humble-image-transport-plugins \
  ros-humble-web-video-server \
  ros-humble-apriltag-ros \
  ros-humble-apriltag-msgs \
  ros-humble-cv-bridge \
  python3-opencv
do
  miss="$(need_pkg "$p" || true)"
  [ -n "$miss" ] && PKGS+=("$miss")
done

if [ "${#PKGS[@]}" -ne 0 ]; then
  echo "[install] missing packages: ${PKGS[*]}"
  export DEBIAN_FRONTEND=noninteractive
  apt update
  apt install -y "${PKGS[@]}"
fi

# apt로 설치된 패키지 환경 반영
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=0

# ---- video device auto pick ----
VIDEO_DEV=""
if ls /dev/video* >/dev/null 2>&1; then
  VIDEO_DEV="$(ls /dev/video* 2>/dev/null | sort -V | head -n 1)"
fi

if [ -z "$VIDEO_DEV" ]; then
  echo "[FATAL] No /dev/video* devices found inside container."
  echo "Hint: docker run에 --device /dev/videoX 를 넣어야 합니다."
  exit 1
fi

echo "[INFO] Using VIDEO_DEV=$VIDEO_DEV"

# ---- silence calibration warning by creating empty file (optional but handy) ----
# v4l2_camera가 기본으로 참조하는 경로가 /root/.ros/camera_info 아래라서 미리 만들어둠
mkdir -p /root/.ros/camera_info
CAL_FILE="/root/.ros/camera_info/integrated_webcam_hd:_integrate.yaml"
if [ ! -f "$CAL_FILE" ]; then
  # 빈 파일이라 "정확한 보정"은 아니지만, 파일 없음 경고는 줄어듦
  touch "$CAL_FILE"
fi

# ---- tmux ----
SESSION=run
tmux has-session -t "$SESSION" 2>/dev/null && tmux kill-session -t "$SESSION"
tmux new-session -d -s "$SESSION" -n cam

# 0) cam
# 카메라가 MJPG 지원(네 로그에 있음)하므로 기본을 MJPG로 둠.
# 만약 화면이 깨지거나 토픽이 안 나오면 MJPG -> YUYV로 되돌리면 됨.
tmux send-keys -t "$SESSION:cam" \
"source /opt/ros/humble/setup.bash; export ROS_DOMAIN_ID=0; \
echo '[cam] device=${VIDEO_DEV}'; \
ros2 run v4l2_camera v4l2_camera_node --ros-args \
  -p video_device:=${VIDEO_DEV} \
  -p pixel_format:=YUYV \
  -p image_size:='[640,480]' \
  -p time_per_frame:='[1,15]'" C-m

# 1) republish: /image_raw -> /image_raw_be
tmux new-window -t "$SESSION" -n repub
tmux send-keys -t "$SESSION:repub" \
"source /opt/ros/humble/setup.bash; export ROS_DOMAIN_ID=0; \
ros2 run image_transport republish raw --ros-args \
  --remap in:=/image_raw \
  --remap out:=/image_raw_be" C-m

# 2) apriltag detector
tmux new-window -t "$SESSION" -n apriltag
tmux send-keys -t "$SESSION:apriltag" \
"source /opt/ros/humble/setup.bash; export ROS_DOMAIN_ID=0; \
ros2 run apriltag_ros apriltag_node --ros-args \
  --remap image_rect:=/image_raw_be \
  --remap camera_info:=/camera_info \
  -p tag_family:=tag36h11 \
  -p tag_size:=0.162" C-m

# 3) overlay
tmux new-window -t "$SESSION" -n overlay
tmux send-keys -t "$SESSION:overlay" \
"source /opt/ros/humble/setup.bash; export ROS_DOMAIN_ID=0; \
python3 /scripts/apriltag_overlay.py" C-m

# 4) web
tmux new-window -t "$SESSION" -n web
tmux send-keys -t "$SESSION:web" \
"source /opt/ros/humble/setup.bash; export ROS_DOMAIN_ID=0; \
ros2 run web_video_server web_video_server" C-m

tmux attach -t "$SESSION"