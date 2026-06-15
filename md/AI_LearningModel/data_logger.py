import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist

from cv_bridge import CvBridge

import cv2
import os
import csv
import time


class DataLogger(Node):

    def __init__(self):
        super().__init__('data_logger')

        self.bridge = CvBridge()

        self.latest_linear = 0.0
        self.latest_angular = 0.0

        self.dataset_dir = "dataset"
        self.image_dir = os.path.join(self.dataset_dir,"images")

        os.makedirs(self.image_dir, exist_ok=True)

        self.csv_path = os.path.join(
            self.dataset_dir,
            "labels.csv"
        )

        if not os.path.exists(self.csv_path):
            with open(self.csv_path,"w",newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "image",
                    "linear_speed",
                    "angular_speed",
                    "timestamp"
                ])

        self.image_count = 0

        self.create_subscription(
            Twist,
            "/cmd_vel",
            self.cmd_callback,
            10
        )

        self.create_subscription(
            Image,
            "/camera/image_raw",
            self.image_callback,
            10
        )

        self.get_logger().info(
            "Data logger started."
        )


    def cmd_callback(self,msg):
        self.latest_linear = msg.linear.x
        self.latest_angular = msg.angular.z


    def image_callback(self,msg):

        try:
            frame = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding='bgr8'
            )

            # optional crop
            h,w,_ = frame.shape
            frame = frame[int(h*0.4):,:]

            # resize for learning
            frame = cv2.resize(
                frame,
                (160,120)
            )

            filename = f"{self.image_count:06d}.jpg"

            image_path = os.path.join(
                self.image_dir,
                filename
            )

            cv2.imwrite(
                image_path,
                frame
            )

            with open(
                self.csv_path,
                "a",
                newline=''
            ) as f:

                writer = csv.writer(f)

                writer.writerow([
                    filename,
                    self.latest_linear,
                    self.latest_angular,
                    time.time()
                ])

            self.image_count += 1

            if self.image_count % 100 == 0:
                self.get_logger().info(
                    f"{self.image_count} samples saved"
                )

        except Exception as e:
            self.get_logger().error(
                str(e)
            )


def main(args=None):

    rclpy.init(args=args)

    node = DataLogger()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()