#!/usr/bin/env python3

# ----  robot 1217 ----

import rospy
import cv2
import numpy as np

from sensor_msgs.msg import CompressedImage, LaserScan
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge

class LineTracerWithObstacleAvoidance:
    def __init__(self):
        rospy.init_node("line_tracer_with_obstacle_avoidance")

        self.pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        rospy.Subscriber("/usb_cam/image_raw/compressed", CompressedImage, self.camera_cb)
        rospy.Subscriber("/scan", LaserScan, self.lidar_cb)

        # --- 디버깅용 타이머 추가 (1초 주기로 상태 보고) ---
        rospy.Timer(rospy.Duration(1.0), self.debug_log_cb)

        self.bridge = CvBridge()

        # =====================
        # Parameters
        # =====================
        self.speed = 0.22
        self.robot_width = 0.20 
        self.safe_distance = 0.45 

        self.scan_ranges = []
        self.front_dist = 999.0

        # FSM
        self.state = "LANE"
        self.state_start = 0.0
        self.escape_angle = 0.0
        self.current_twist = Twist() # 현재 명령 저장용 (로그용)

        # Escape logic memory
        self.left_escape_count = 0
        self.force_right_escape = 0

    # ============================================================
    # DEBUG LOG CALLBACK (1초마다 실행)
    # ============================================================
    def debug_log_cb(self, event):
        log_msg = f"\n[STATUS REPORT]\n" \
                  f"- State: {self.state}\n" \
                  f"- Front Dist: {self.front_dist:.3f}m\n" \
                  f"- Cmd Vel: x={self.current_twist.linear.x:.2f}, z={self.current_twist.angular.z:.2f}\n"
        
        if self.state == "ESCAPE":
            log_msg += f"- Escape Angle Target: {np.rad2deg(self.escape_angle):.1f} deg\n"
        
        rospy.loginfo(log_msg)

    def lidar_cb(self, scan):
        raw = np.array(scan.ranges)
        raw = np.where(raw < 0.05, 9.0, raw)
        self.scan_ranges = raw

        front_indices = np.concatenate([np.arange(0, 21), np.arange(340, 360)])
        front_values = raw[front_indices]
        
        valid_front = front_values[(front_values > 0.1) & (front_values < 10.0)]
        if len(valid_front) > 0:
            self.front_dist = np.min(valid_front)
        else:
            self.front_dist = 999.0

    def camera_cb(self, msg):
        now = rospy.Time.now().to_sec()

        if self.state == "LANE" and self.front_dist < self.safe_distance:
            self.state = "BACK"
            self.state_start = now
            return

        if self.state == "BACK":
            self.back_control(now)
            return

        if self.state == "ESCAPE":
            self.escape_control(now)
            return

        # LANE MODE
        try:
            frame = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
            h, w = frame.shape[:2]
            roi_near = frame[int(h * 0.55):h, :]
            hsv_near = cv2.cvtColor(roi_near, cv2.COLOR_BGR2HSV)
            
            lower_white = np.array([0, 0, 180])
            upper_white = np.array([180, 40, 255])
            mask_near = cv2.inRange(hsv_near, lower_white, upper_white)
            contours_near, _ = cv2.findContours(mask_near, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

            twist = Twist()
            if len(contours_near) > 0:
                c = max(contours_near, key=cv2.contourArea)
                M = cv2.moments(c)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    error = cx - (w // 2)
                    twist.linear.x = self.speed
                    twist.angular.z = -error / 150.0
            else:
                twist.linear.x = 0.1
                twist.angular.z = 0.2 
            
            self.current_twist = twist # 로그용 저장
            self.pub.publish(twist)
        except Exception as e:
            pass

    def back_control(self, now):
        twist = Twist()
        if now - self.state_start < 1.2:
            twist.linear.x = -0.2
            self.current_twist = twist
            self.pub.publish(twist)
        else:
            target_angle = self.find_best_gap()
            self.escape_angle = self.apply_escape_direction_logic(target_angle)
            self.state = "ESCAPE"
            self.state_start = now

    def escape_control(self, now):
        twist = Twist()
        if now - self.state_start < 1.5:
            twist.linear.x = 0.18
            twist.angular.z = self.escape_angle * 1.5
            self.current_twist = twist
            self.pub.publish(twist)
        else:
            self.state = "LANE"

    def find_best_gap(self):
        if len(self.scan_ranges) == 0:
            return 0.0

        scan_data = np.array(self.scan_ranges)
        angles = np.arange(-80, 80)
        distances = []
        for a in angles:
            dist = scan_data[a % 360]
            distances.append(dist if dist > 0.1 else 0.0)
        
        distances = np.array(distances)
        min_pass_dist = 0.6 
        accessible = (distances > min_pass_dist).astype(int)
        
        max_width = 0
        best_idx = 80 
        curr_width = 0
        for i, val in enumerate(accessible):
            if val == 1:
                curr_width += 1
                if curr_width > max_width:
                    max_width = curr_width
                    best_idx = i - (curr_width // 2)
            else:
                curr_width = 0
        
        return np.deg2rad(angles[best_idx])

    def apply_escape_direction_logic(self, angle):
        if self.force_right_escape > 0:
            self.force_right_escape -= 1
            return 0.8

        if angle > 0.1: 
            self.left_escape_count += 1
            if self.left_escape_count >= 3:
                self.force_right_escape = 2
                self.left_escape_count = 0
                return -0.8
        else:
            self.left_escape_count = 0
        return angle

if __name__ == "__main__":
    try:
        LineTracerWithObstacleAvoidance()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass