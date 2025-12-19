#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np
import math
from sensor_msgs.msg import CompressedImage, LaserScan
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge

# 1253,
# ip 121. 

class LimoMissionIntegrated:
    def __init__(self):
        rospy.init_node('limo_mission_integrated', anonymous=True)
        
        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        
        # === [설정] 상태 머신 ===
        self.mode = "LINE"  # 초기 모드: 라인 트레이싱
        
        # === [설정] 장애물 감지 및 모드 전환 ===
        self.trigger_dist = 1.0       # 이 거리 안에 장애물 있으면 감지 시작
        self.trigger_count = 0        # 노이즈 필터링용 카운터
        self.trigger_limit = 5        # 5번 연속 감지 시 모드 전환
        
        # === [설정] 라인 트레이싱 (새로운 로직) ===
        self.bridge = CvBridge()
        self.line_speed = 0.3        # 라인 주행 속도
        
        # HSV 색상 범위 (검은색 + 노란색)
        self.black_lower = np.array([102, 0, 60])
        self.black_upper = np.array([164, 86, 136])
        self.black2_lower = np.array([126, 25, 45])
        self.black2_upper = np.array([167, 89, 108])
        self.black3_lower = np.array([125, 29, 26])
        self.black3_upper = np.array([171, 100, 78])
        self.yellow_lower = np.array([14, 17, 153])
        self.yellow_upper = np.array([35, 167, 255])
        
        # Bird's Eye View 파라미터
        self.margin_x = 150
        self.margin_y = 350
        
        # 조향 제어 (Smoothing)
        self.steer_weight = 2.0
        self.steer_alpha = 0.35
        self.steer_max = 1.20
        self.steer_rate = 0.14
        self.steer = 0.0
        self.steer_f = 0.0

        # === [설정] Gap Follower (장애물 회피) ===
        self.gap_speed = 0.22
        self.gap_kp = 1.5
        self.safe_dist = 1.0
        self.view_limit_left = 15.0   
        self.view_limit_right = -80.0 
        
        # === [설정] 충돌 감지 및 후진 ===
        self.corner_safe_dist_default = 0.25  
        self.corner_safe_dist_narrow = 0.12   
        self.wall_ignore_y = 0.5
        self.recover_start_time = 0
        self.recover_duration = 1.5
        self.recover_dir = 0
        self.recover_rot_speed = 0.4

        # === Subscribers ===
        # [수정됨] 카메라 토픽 변경: /usb_cam/... -> /camera/rgb/...
        # rospy.Subscriber('/camera/rgb/image_raw/compressed', CompressedImage, self.image_callback)
        # * * 
        rospy.Subscriber("/usb_cam/image_raw/compressed", CompressedImage, self.camera_cb)
        rospy.Subscriber('/scan', LaserScan, self.scan_callback)
        
        # 종료 시 로봇 정지를 위한 hook 등록
        rospy.on_shutdown(self.shutdown_hook)

        rospy.loginfo("===== LIMO Integrated Mission Started =====")
        rospy.loginfo("Mode: LINE (Yellow/Black) -> Detect Obstacle -> Mode: GAP")

    def shutdown_hook(self):
        """노드 종료 시 로봇을 정지시킵니다."""
        stop_twist = Twist()
        self.cmd_pub.publish(stop_twist)
        rospy.loginfo("LIMO Stopped.")

    # ---------------------------------------------------------
    # 1. 라인 트레이싱 콜백
    # ---------------------------------------------------------
    def image_callback(self, msg):
        if self.mode != "LINE": 
            return # GAP 모드일 때는 카메라 무시

        try:
            # CompressedImage -> OpenCV 변환
            cv_img = self.bridge.compressed_imgmsg_to_cv2(msg)
            y, x, _ = cv_img.shape
            hsv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)
            
            # 1. 필터링
            yellow_filter = cv2.inRange(hsv_img, self.yellow_lower, self.yellow_upper)
            b1 = cv2.inRange(hsv_img, self.black_lower, self.black_upper)
            b2 = cv2.inRange(hsv_img, self.black2_lower, self.black2_upper)
            b3 = cv2.inRange(hsv_img, self.black3_lower, self.black3_upper)
            black_filter = cv2.bitwise_or(b1, cv2.bitwise_or(b2, b3))
            
            # 노란색 or 검은색 인식
            target_filter = cv2.bitwise_or(black_filter, yellow_filter)

            # 2. Bird's Eye View 변환
            src_pts = np.float32([(30, y), (self.margin_x, self.margin_y), (x - self.margin_x, self.margin_y), (x - 30, y)])
            dst_margin_x = 120
            dst_pts = np.float32([(dst_margin_x, y), (dst_margin_x, 0), (x - dst_margin_x, 0), (x - dst_margin_x, y)])
            matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
            warp_img = cv2.warpPerspective(target_filter, matrix, (x, y))

            # 3. 조향각 계산 (Moments)
            M = cv2.moments(warp_img)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                error = (x // 2) - cx
                
                steer_raw = (error * math.pi / x) * self.steer_weight
                steer_raw = float(np.clip(steer_raw, -self.steer_max, self.steer_max))

                # Smoothing
                self.steer_f = (1.0 - self.steer_alpha) * self.steer_f + self.steer_alpha * steer_raw
                d = self.steer_f - self.steer
                d = float(np.clip(d, -self.steer_rate, self.steer_rate))
                self.steer = float(np.clip(self.steer + d, -self.steer_max, self.steer_max))
            else:
                # 라인을 잃어버렸을 때 기존 조향 유지 (급격한 회전 방지)
                self.steer = float(np.clip(self.steer, -0.45, 0.45))

            # 주행 명령
            twist = Twist()
            twist.linear.x = self.line_speed
            twist.angular.z = self.steer
            self.cmd_pub.publish(twist)

        except Exception as e:
            rospy.logerr(f"Image Processing Error: {e}")

    # ---------------------------------------------------------
    # 2. LiDAR 콜백 (모드 전환 + 장애물 회피 + 후진)
    # ---------------------------------------------------------
    def scan_callback(self, msg):
        # A. 후진(RECOVER) 모드 처리
        if self.mode == "RECOVER":
            if rospy.Time.now().to_sec() - self.recover_start_time > self.recover_duration:
                rospy.loginfo("[RECOVER] Done. Go Gap.")
                self.mode = "GAP"
            else:
                twist = Twist()
                twist.linear.x = -0.2
                rot = self.recover_rot_speed
                # 부딪힌 반대 방향으로 회전하며 후진
                twist.angular.z = rot if self.recover_dir == -1 else -rot
                self.cmd_pub.publish(twist)
            return

        ranges = msg.ranges
        angle_min = msg.angle_min
        angle_inc = msg.angle_increment
        
        front_min = 10.0
        path_clear_dist = 10.0
        left_corner_min = 10.0
        right_corner_min = 10.0
        scan_data = []

        for i, r in enumerate(ranges):
            if np.isinf(r) or r < 0.1: continue
            
            raw_angle = angle_min + i * angle_inc
            robot_angle = raw_angle 
            
            deg = math.degrees(robot_angle)

            # 1. 정면 트리거 데이터 수집 (+/- 10도)
            if abs(deg) < 10:
                if r < front_min: front_min = r
                if r < path_clear_dist: path_clear_dist = r
            
            # 2. GAP 모드용 데이터 (뒤쪽 버림)
            if abs(deg) > 90: continue
            
            # 시야각 필터
            if self.view_limit_right < deg < self.view_limit_left:
                scan_data.append((robot_angle, min(r, 3.0)))

            # 벽과의 거리 계산 (y축 거리)
            oy = r * math.sin(robot_angle)
            if abs(oy) > self.wall_ignore_y: continue

            # 코너 충돌 감지
            if 20 < deg < 60:
                if r < left_corner_min: left_corner_min = r
            elif -60 < deg < -20:
                if r < right_corner_min: right_corner_min = r

        # B. 모드 전환 로직 (LINE -> GAP)
        if self.mode == "LINE":
            if front_min < self.trigger_dist:
                self.trigger_count += 1
            else:
                self.trigger_count = 0 # 연속 감지 끊기면 리셋

            if self.trigger_count >= self.trigger_limit:
                rospy.logwarn(f"!!! OBSTACLE CONFIRMED ({front_min:.2f}m) -> GAP MODE !!!")
                self.mode = "GAP"
                # 잠깐 정지 후 모드 전환
                stop = Twist()
                for _ in range(5): 
                    self.cmd_pub.publish(stop)
                    rospy.sleep(0.05)
            
            return # LINE 모드일 땐 여기서 종료 (LiDAR로 주행 안 함)

        # C. GAP 모드 & 충돌 방지
        current_safe_limit = self.corner_safe_dist_default
        if path_clear_dist > 0.4: 
            current_safe_limit = self.corner_safe_dist_narrow

        if left_corner_min < current_safe_limit:
            rospy.logwarn(f"!!! LEFT HIT ({left_corner_min:.2f}m) -> BACKUP !!!")
            self.mode = "RECOVER"
            self.recover_dir = 1
            self.recover_start_time = rospy.Time.now().to_sec()
            return
        elif right_corner_min < current_safe_limit:
            rospy.logwarn(f"!!! RIGHT HIT ({right_corner_min:.2f}m) -> BACKUP !!!")
            self.mode = "RECOVER"
            self.recover_dir = -1
            self.recover_start_time = rospy.Time.now().to_sec()
            return

        # Gap Finding Algorithm
        scan_data.sort(key=lambda x: x[0])
        best_gap_center = 0.0
        max_gap_len = 0
        current_start = -1
        current_len = 0
        
        for i in range(len(scan_data)):
            _, dist = scan_data[i]
            if dist > self.safe_dist:
                if current_start == -1: current_start = i
                current_len += 1
            else:
                if current_start != -1:
                    if current_len > max_gap_len:
                        max_gap_len = current_len
                        s, e = current_start, i-1
                        best_gap_center = scan_data[int((s+e)/2)][0]
                    current_start = -1
                    current_len = 0
                    
        if current_start != -1 and current_len > max_gap_len:
            s, e = current_start, len(scan_data)-1
            best_gap_center = scan_data[int((s+e)/2)][0]
            max_gap_len = current_len

        twist = Twist()
        # 충분한 갭이 발견되었을 때
        if max_gap_len > 5:
            twist.linear.x = self.gap_speed
            twist.angular.z = self.gap_kp * best_gap_center
            
            # 코너가 가까우면 회피 가중치 부여
            if right_corner_min < 0.40:
                twist.angular.z += 0.3 
            elif left_corner_min < 0.40:
                twist.angular.z -= 0.3

            twist.angular.z = np.clip(twist.angular.z, -1.2, 1.2)
        else:
            # 갈 곳이 없으면 제자리 회전 혹은 정지
            twist.linear.x = 0.0
            twist.angular.z = -0.6
            
        self.cmd_pub.publish(twist)

if __name__ == '__main__':
    try:
        LimoMissionIntegrated()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass