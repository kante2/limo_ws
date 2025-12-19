#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Refactor Controller (p10 제거 / front median만 사용)

구조:
1) LiDAR CB: front_distance_m(전방거리) + gap_target_angle_rad 저장
   - front는 2번째 코드 방식: raw[:N] + raw[-N:] 의 median
2) Camera CB: cone / line 결과 저장
3) Control Timer: 여기서만 /cmd_vel publish
4) Debug Timer: ROSINFO 1초 주기 출력

후진 트리거:
- (lidar_fresh) and (front_distance_m < obstacle_emergency_threshold_m)  -> BACK
- BACK 들어가면 emergency_back_duration_sec 동안 무조건 후진 유지
"""

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import CompressedImage, LaserScan
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


class LimoMissionController:
    def __init__(self):
        rospy.init_node("limo_mission_controller_refactor")

        # ---------------------------
        # Pub/Sub
        # ---------------------------
        self.cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        rospy.Subscriber("/scan", LaserScan, self.lidar_cb, queue_size=1)
        rospy.Subscriber("/usb_cam/image_raw/compressed", CompressedImage, self.camera_cb, queue_size=1)
        self.bridge = CvBridge()

        # ---------------------------
        # Timers
        # ---------------------------
        self.control_rate_hz = 20.0
        rospy.Timer(rospy.Duration(1.0 / self.control_rate_hz), self.control_timer_cb)
        rospy.Timer(rospy.Duration(1.0), self.debug_timer_cb)

        # ---------------------------
        # Latest cmd (debug)
        # ---------------------------
        self.last_cmd_linear_mps = 0.0
        self.last_cmd_angular_rps = 0.0
        self.control_mode = "INIT"

        # ---------------------------
        # Sensor freshness watchdog
        # ---------------------------
        self.last_lidar_stamp = 0.0
        self.last_camera_stamp = 0.0
        self.lidar_stale_sec = 0.5
        self.camera_stale_sec = 0.5

        # ---------------------------
        # Line-follow parameters
        # ---------------------------
        self.line_follow_speed_mps = 0.30
        self.line_search_turn_rps = 0.25
        self.line_steer_gain = 0.010
        self.line_min_pixel_count = 5
        self.line_turn_limit_rps = 0.85

        # ---------------------------
        # Cone-follow parameters
        # ---------------------------
        self.cone_follow_speed_mps = 0.15
        self.cone_steer_gain = 0.005
        self.cone_min_contour_area = 200

        # ---------------------------
        # Obstacle / Gap parameters
        # ---------------------------
        self.obstacle_emergency_threshold_m = 0.30  # 이하면 BACK
        self.obstacle_follow_threshold_m = 0.2     # 이하면 GAP (원하면 0.30으로 맞춰도 됨)

        self.emergency_back_duration_sec = 3.0
        self.emergency_back_speed_mps = -0.15

        self.gap_follow_speed_mps = 0.15
        self.gap_steer_gain = 1.2
        self.gap_turn_limit_rps = 1.2

        # GAP 계산용 FOV(각도 기반)
        self.gap_search_fov_deg = 90.0
        self.gap_free_distance_m = 0.35

        # LiDAR filtering
        self.lidar_min_valid_range_m = 0.05
        self.lidar_max_range_clip_m = 3.0

        # ✅ FRONT: 2번째 코드처럼 raw[:N] + raw[-N:]
        self.front_check_bins = 10  # 10 -> 15/20 늘리면 정면을 더 넓게 봄

        # ---------------------------
        # BACK state
        # ---------------------------
        self.emergency_state = "NONE"  # NONE / BACK
        self.emergency_start_time = 0.0

        # ---------------------------
        # [1] LiDAR outputs
        # ---------------------------
        self.front_distance_m = 999.0
        self.front_distance_min_m = 999.0  # 참고용
        self.front_sample_count = 0

        self.gap_target_angle_rad = 0.0
        self.best_gap_score = 0.0
        self.best_gap_width_bins = 0

        # ---------------------------
        # [2] Camera outputs
        # ---------------------------
        self.cone_visible = False
        self.cone_cmd_v = 0.0
        self.cone_cmd_w = 0.0
        self.cone_count = 0
        self.cone_target_x = -1

        self.line_visible = False
        self.line_cmd_w = 0.0
        self.line_center_offset_px = 0.0
        self.line_col_max = 0

        rospy.loginfo("=== Start: FRONT median only (no p10) ===")

    # ============================================================
    # 1) LiDAR CB: FRONT(2nd style median) + GAP(angle style) 저장
    # ============================================================
    def lidar_cb(self, scan: LaserScan):
        self.last_lidar_stamp = rospy.Time.now().to_sec()

        raw_ranges = np.array(scan.ranges, dtype=np.float32)
        n = raw_ranges.size
        if n == 0:
            self.front_distance_m = 999.0
            self.front_distance_min_m = 999.0
            self.front_sample_count = 0
            self.gap_target_angle_rad = 0.0
            return

        # sanitize ranges
        ranges_m = raw_ranges.copy()
        ranges_m[np.isnan(ranges_m) | np.isinf(ranges_m)] = 0.0
        ranges_m[ranges_m < self.lidar_min_valid_range_m] = 0.0
        ranges_m = np.clip(ranges_m, 0.0, self.lidar_max_range_clip_m)

        # ----------------------------
        # FRONT: raw[:N] + raw[-N:] median (2번째 코드)
        # ----------------------------
        N = int(clamp(self.front_check_bins, 1, max(1, n // 2)))
        front_zone = np.concatenate([ranges_m[:N], ranges_m[-N:]])
        front_valid = front_zone[front_zone > 0.0]
        self.front_sample_count = int(front_valid.size)

        if front_valid.size > 0:
            self.front_distance_m = float(np.median(front_valid))
            self.front_distance_min_m = float(np.min(front_valid))
        else:
            self.front_distance_m = 999.0
            self.front_distance_min_m = 999.0

        # ----------------------------
        # GAP: angle 기반 Follow-the-Gap
        # ----------------------------
        angles = scan.angle_min + np.arange(n, dtype=np.float32) * scan.angle_increment
        angles = (angles + np.pi) % (2.0 * np.pi) - np.pi  # wrap 대비

        gap_mask = np.abs(angles) <= np.deg2rad(self.gap_search_fov_deg)
        gap_ranges = ranges_m[gap_mask]
        gap_angles = angles[gap_mask]

        if gap_ranges.size < 10:
            self.gap_target_angle_rad = 0.0
            self.best_gap_score = 0.0
            self.best_gap_width_bins = 0
            return

        window = 5
        kernel = np.ones(window, dtype=np.float32) / float(window)
        smooth = np.convolve(gap_ranges, kernel, mode="same")

        free = smooth >= self.gap_free_distance_m

        best_score = -1.0
        best_seg = None
        i, L = 0, free.size
        while i < L:
            if not free[i]:
                i += 1
                continue
            j = i
            while j < L and free[j]:
                j += 1
            seg = smooth[i:j]
            score = float(len(seg) * np.mean(seg)) if len(seg) > 0 else 0.0
            if score > best_score:
                best_score = score
                best_seg = (i, j)
            i = j

        if best_seg is None:
            self.gap_target_angle_rad = 0.0
            self.best_gap_score = 0.0
            self.best_gap_width_bins = 0
            return

        s, e = best_seg
        self.best_gap_width_bins = int(e - s)
        self.best_gap_score = float(best_score)

        local_best = int(np.argmax(smooth[s:e]))
        target_idx = s + local_best
        self.gap_target_angle_rad = float(gap_angles[target_idx])

    # ============================================================
    # 2) Camera CB: cone/line 결과 저장
    # ============================================================
    def camera_cb(self, msg: CompressedImage):
        self.last_camera_stamp = rospy.Time.now().to_sec()

        try:
            frame = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
            h, w = frame.shape[:2]

            roi_y = int(h * 0.5)
            roi = frame[roi_y:, :]
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

            # Cone detection (red)
            lower_r1 = np.array([0, 100, 80])
            upper_r1 = np.array([10, 255, 255])
            lower_r2 = np.array([170, 100, 80])
            upper_r2 = np.array([180, 255, 255])

            mask1 = cv2.inRange(hsv, lower_r1, upper_r1)
            mask2 = cv2.inRange(hsv, lower_r2, upper_r2)
            red_mask = cv2.bitwise_or(mask1, mask2)
            red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

            contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            cone_centers_x = []
            for cnt in contours:
                if cv2.contourArea(cnt) > self.cone_min_contour_area:
                    M = cv2.moments(cnt)
                    if M["m00"] != 0:
                        cone_centers_x.append(int(M["m10"] / M["m00"]))

            if len(cone_centers_x) > 0:
                cone_centers_x.sort()
                self.cone_visible = True
                self.cone_count = len(cone_centers_x)

                if len(cone_centers_x) >= 2:
                    target_x = (cone_centers_x[0] + cone_centers_x[-1]) // 2
                else:
                    single_x = cone_centers_x[0]
                    target_x = (w - 100) if (single_x < w // 2) else 100

                self.cone_target_x = int(target_x)

                pixel_error = (w // 2) - target_x
                angular_cmd = float(pixel_error) * self.cone_steer_gain
                angular_cmd = clamp(angular_cmd, -1.2, 1.2)

                self.cone_cmd_v = float(self.cone_follow_speed_mps)
                self.cone_cmd_w = float(angular_cmd)

                self.line_visible = False
                return
            else:
                self.cone_visible = False
                self.cone_cmd_v = 0.0
                self.cone_cmd_w = 0.0
                self.cone_count = 0
                self.cone_target_x = -1

            # Line detection (black)
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            kernel = np.ones((3, 3), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

            col_sum = np.sum(binary > 0, axis=0)
            max_col = int(np.max(col_sum)) if col_sum.size > 0 else 0
            self.line_col_max = int(max_col)

            if max_col < self.line_min_pixel_count:
                self.line_visible = False
                self.line_cmd_w = 0.0
                self.line_center_offset_px = 0.0
                return

            thresh = max(self.line_min_pixel_count, int(max_col * 0.3))
            candidates = np.where(col_sum >= thresh)[0]
            if candidates.size == 0:
                self.line_visible = False
                self.line_cmd_w = 0.0
                self.line_center_offset_px = 0.0
                return

            x_idx = np.arange(len(col_sum))
            line_center_x = float(np.sum(x_idx[candidates] * col_sum[candidates]) /
                                  np.sum(col_sum[candidates]))

            offset_px = line_center_x - (w / 2.0)
            self.line_center_offset_px = float(offset_px)

            angular_cmd = -self.line_steer_gain * offset_px
            angular_cmd = clamp(angular_cmd, -self.line_turn_limit_rps, self.line_turn_limit_rps)

            self.line_visible = True
            self.line_cmd_w = float(angular_cmd)

        except Exception as e:
            rospy.logerr(f"[camera_cb] Error: {e}")
            self.cone_visible = False
            self.line_visible = False

    # ============================================================
    # 3) Control Timer: 여기서만 publish
    # ============================================================
    def control_timer_cb(self, event):
        now = rospy.Time.now().to_sec()
        twist = Twist()

        lidar_fresh = (now - self.last_lidar_stamp) <= self.lidar_stale_sec
        camera_fresh = (now - self.last_camera_stamp) <= self.camera_stale_sec

        if not lidar_fresh and not camera_fresh:
            self.control_mode = "STOP(no_sensors)"
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            self._publish_and_store(twist)
            return

        # (A) BACK 유지
        if self.emergency_state == "BACK":
            if (now - self.emergency_start_time) < self.emergency_back_duration_sec:
                self.control_mode = "BACK(emergency)"
                twist.linear.x = float(self.emergency_back_speed_mps)
                twist.angular.z = 0.0
                self._publish_and_store(twist)
                return
            else:
                self.emergency_state = "NONE"

        # (B) BACK 트리거 (median 기반)
        if lidar_fresh and (self.front_distance_m < self.obstacle_emergency_threshold_m):
            self.emergency_state = "BACK"
            self.emergency_start_time = now

            self.control_mode = "BACK(start)"
            twist.linear.x = float(self.emergency_back_speed_mps)
            twist.angular.z = 0.0
            self._publish_and_store(twist)
            return

        # (C) 콘 우선(원래 느낌 유지)
        if camera_fresh and self.cone_visible:
            self.control_mode = "CONE"
            twist.linear.x = float(self.cone_cmd_v)
            twist.angular.z = float(self.cone_cmd_w)
            self._publish_and_store(twist)
            return

        # (D) GAP / LINE / SEARCH
        # obstacle_follow_threshold_m -> 20cm 으로 수정된 부분
        if lidar_fresh and (self.front_distance_m < self.obstacle_follow_threshold_m):
            self.control_mode = "GAP"
            twist.linear.x = float(self.gap_follow_speed_mps)
            twist.angular.z = float(clamp(self.gap_steer_gain * self.gap_target_angle_rad,
                                          -self.gap_turn_limit_rps, self.gap_turn_limit_rps))
        elif camera_fresh and self.line_visible:
            self.control_mode = "LINE"
            twist.linear.x = float(self.line_follow_speed_mps)
            twist.angular.z = float(self.line_cmd_w)
        else:
            self.control_mode = "SEARCH"
            twist.linear.x = 0.0
            twist.angular.z = float(self.line_search_turn_rps)

        self._publish_and_store(twist)

    def _publish_and_store(self, twist: Twist):
        self.last_cmd_linear_mps = float(twist.linear.x)
        self.last_cmd_angular_rps = float(twist.angular.z)
        self.cmd_pub.publish(twist)

    # ============================================================
    # 4) Debug Timer: 1초 주기 출력
    # ============================================================
    def debug_timer_cb(self, event):
        gap_deg = float(self.gap_target_angle_rad * 180.0 / np.pi)

        rospy.loginfo(
            f"[DBG] mode={self.control_mode:12s} state={self.emergency_state:4s} "
            f"front(med={self.front_distance_m:5.2f},min={self.front_distance_min_m:5.2f}) "
            f"(N={self.front_check_bins},valid={self.front_sample_count}) "
            f"gap={gap_deg:6.1f}deg(w={self.best_gap_width_bins:3d},score={self.best_gap_score:7.1f}) | "
            f"cone={int(self.cone_visible)}(n={self.cone_count},tx={self.cone_target_x}) | "
            f"line={int(self.line_visible)}(max={self.line_col_max},off={self.line_center_offset_px:6.1f},w={self.line_cmd_w:6.2f}) | "
            f"cmd(v={self.last_cmd_linear_mps:5.2f},w={self.last_cmd_angular_rps:6.2f})"
        )


if __name__ == "__main__":
    try:
        node = LimoMissionController()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
