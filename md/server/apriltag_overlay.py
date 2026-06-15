#!/usr/bin/env python3
"""
apriltag_overlay.py

기능
- /image_raw_be (카메라 이미지)와 /detections (AprilTag 검출 결과)를 구독
- 이미지 위에 태그 코너/중심/ID 텍스트를 그려서
- /apriltag_overlay 토픽으로 다시 publish

전제
- ROS 2 Humble
- apriltag_msgs (AprilTagDetectionArray)
- cv_bridge, OpenCV

실행 예)
  source /opt/ros/humble/setup.bash
  export ROS_DOMAIN_ID=0
  python3 apriltag_overlay.py

Windows에서 보기 예)
- web_video_server 실행 후
- 브라우저에서 http://<RASPI_IP>:8080/ 접속
- /apriltag_overlay 토픽 선택
"""

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from apriltag_msgs.msg import AprilTagDetectionArray

from cv_bridge import CvBridge
import cv2


class AprilTagOverlay(Node):
    """
    타이머 기반으로 가장 최근 이미지에 가장 최근 detections를 얹어서 publish 하는 방식.
    - 정확한 타임 싱크(동기화)를 강제하지 않음 -> 구현이 단순하고 대부분의 디버깅엔 충분
    - 만약 완전한 동기화를 원하면 message_filters(ApproximateTimeSynchronizer) 방식으로 변경 가능
    """

    def __init__(self):
        super().__init__("apriltag_overlay")

        # ROS Image <-> OpenCV Mat 변환 도구
        self.bridge = CvBridge()

        # 가장 최근에 받은 입력을 저장해두는 버퍼
        self.last_image_msg: Image | None = None
        self.last_dets_msg: AprilTagDetectionArray | None = None

        # 이미지 구독: 그려줄 "바탕" 이미지
        # - 원본이 /image_raw라면 여기 토픽명을 /image_raw로 바꿔도 됨
        self.create_subscription(Image, "/image_raw_be", self.on_image, 10)

        # AprilTag 검출 결과 구독
        self.create_subscription(AprilTagDetectionArray, "/detections", self.on_detections, 10)

        # 오버레이 결과 이미지 publish
        self.pub = self.create_publisher(Image, "/apriltag_overlay", 10)

        # 주기적으로(20Hz) 마지막 이미지+마지막 detections를 합성해 publish
        # - detections가 없으면 그냥 원본 이미지가 그대로 publish됨
        self.timer = self.create_timer(0.05, self.on_timer)

        self.get_logger().info("apriltag_overlay node started. Publishing to /apriltag_overlay")

    def on_image(self, msg: Image) -> None:
        """이미지 토픽 콜백: 가장 최근 이미지를 저장"""
        self.last_image_msg = msg

    def on_detections(self, msg: AprilTagDetectionArray) -> None:
        """detections 토픽 콜백: 가장 최근 검출 결과를 저장"""
        self.last_dets_msg = msg

    def on_timer(self) -> None:
        """
        타이머 콜백:
        - 마지막으로 받은 이미지가 없으면 아무것도 하지 않음
        - 있으면 OpenCV로 변환하고, 마지막 detections를 그린 뒤, 다시 Image로 publish
        """
        if self.last_image_msg is None:
            return

        # ROS Image -> OpenCV BGR 이미지로 변환
        # 주의: 입력 encoding이 rgb8/bgr8 등 다양한데, desired_encoding을 bgr8로 통일
        try:
            cv_img = self.bridge.imgmsg_to_cv2(self.last_image_msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().warn(f"imgmsg_to_cv2 failed: {e}")
            return

        # detections가 아직 한 번도 안 들어왔다면 빈 리스트로 처리(=원본만 publish)
        dets = self.last_dets_msg.detections if self.last_dets_msg is not None else []

        # 각 태그에 대해 코너/중심/텍스트를 그림
        for det in dets:
            # det.corners: 4개의 점(x,y). 태그 사각형의 꼭짓점들
            pts = [(int(c.x), int(c.y)) for c in det.corners]

            # 코너가 4개일 때만 사각형처럼 연결해서 그림
            if len(pts) == 4:
                # 코너를 선으로 연결(초록색)
                for i in range(4):
                    cv2.line(cv_img, pts[i], pts[(i + 1) % 4], (0, 255, 0), 4)

                # 코너 점을 더 눈에 띄게(노란색 원)
                for p in pts:
                    cv2.circle(cv_img, p, 6, (0, 255, 255), -1)

                # (선택) 축정렬 bounding box도 그리고 싶다면 아래 주석 해제
                # xs = [p[0] for p in pts]
                # ys = [p[1] for p in pts]
                # cv2.rectangle(cv_img, (min(xs), min(ys)), (max(xs), max(ys)), (255, 0, 255), 2)

            # det.centre: 태그 중심점
            cx, cy = int(det.centre.x), int(det.centre.y)
            cv2.circle(cv_img, (cx, cy), 6, (0, 0, 255), -1)

            # 라벨: "family:id" 형태
            # 예) "tag36h11:2"
            label = f"{det.family}:{det.id}"
            cv2.putText(
                cv_img,
                label,
                (cx + 10, cy - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 0, 0),
                2,
                cv2.LINE_AA,
            )

        # OpenCV -> ROS Image로 변환해 publish
        out_msg = self.bridge.cv2_to_imgmsg(cv_img, encoding="bgr8")

        # 타임스탬프/프레임은 원본 이미지 header를 그대로 복사
        out_msg.header = self.last_image_msg.header

        self.pub.publish(out_msg)


def main() -> None:
    """ROS2 노드 엔트리포인트"""
    rclpy.init()
    node = AprilTagOverlay()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()