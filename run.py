import cv2
import cupy as cp
import numpy as np
import time
import threading
import asyncio
import aioserial
from queue import Queue
from sklearn.linear_model import RANSACRegressor
from collections import deque
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


class LKAS:
    def __init__(self):
        self.nwindows = 10
        self.img_x = 320
        self.img_y = 180
        self.frame_skip = 1
        self.scale_factor = 0.5
        self.min_area = 50
        self.pid = PIDController(kp=0.1, ki=0.01, kd=0.05)
        self.previous_time = time.time()

        roi_size = min(self.img_x, self.img_y) // 2
        center_x = self.img_x // 2
        center_y = self.img_y // 2

        self.roi_points = [
            (center_x - roi_size // 0.6, center_y + roi_size // 4),
            (center_x + roi_size // 0.75, center_y + roi_size // 4),
            (center_x + roi_size // 3, center_y - roi_size // 2.1),
            (center_x - roi_size // 0.8, center_y - roi_size // 2.1)
        ]

        self.prev_path = None
        self.smoothing_factor = 0.8

        self.ransac_left = RANSACRegressor(residual_threshold=2.5)
        self.ransac_right = RANSACRegressor(residual_threshold=2.5)

        self.left_line_memory = deque(maxlen=10)
        self.right_fit = None
        self.lane_width = None

        self.M = None
        self.Minv = None

        self.previous_left_fitx = None
        self.previous_right_fitx = None
        self.lane_memory = 5

        self.memory_pool = cp.get_default_memory_pool()
        self.pinned_memory_pool = cp.get_default_pinned_memory_pool()

        self.yellow_lower = cp.array([0.09, 0.45, 0.45])
        self.yellow_upper = cp.array([0.11, 1.0, 1.0])
        self.white_lower = cp.array([0.0, 0.0, 0.90])
        self.white_upper = cp.array([1.0, 0.10, 1.0])

        self.frame_count = 0

    def create_roi_mask(self, img):
        mask = cp.zeros_like(img)
        mask_np = cp.asnumpy(mask)
        roi_points_np = cp.asnumpy(cp.array(self.roi_points, dtype=cp.int32))
        cv2.fillPoly(mask_np, [roi_points_np], (255, 255, 255))
        return cp.array(mask_np)

    def rgb_to_hsv(self, img):
        max_val = cp.max(img, axis=0)
        min_val = cp.min(img, axis=0)
        delta = max_val - min_val

        hsv = cp.zeros_like(img)

        hue = cp.zeros_like(max_val)
        mask = delta != 0
        idx = (img[0] == max_val) & mask
        hue[idx] = (img[1][idx] - img[2][idx]) / delta[idx] % 6
        idx = (img[1] == max_val) & mask
        hue[idx] = (img[2][idx] - img[0][idx]) / delta[idx] + 2
        idx = (img[2] == max_val) & mask
        hue[idx] = (img[0][idx] - img[1][idx]) / delta[idx] + 4
        hue /= 6
        hue[hue < 0] += 1

        hsv[0] = hue
        hsv[1] = cp.where(max_val == 0, cp.zeros_like(max_val), delta / max_val)
        hsv[2] = max_val

        return hsv

    def process_image(self, img):
        with cp.cuda.Stream():
            img_gpu = cp.asarray(img.astype(np.float32) / 255.0)
            img_hsv = self.rgb_to_hsv(img_gpu.transpose(2, 0, 1))

            white_mask = ((img_hsv >= self.white_lower[:, None, None]) & (
                    img_hsv <= self.white_upper[:, None, None])).all(axis=0)

            blend_mask = white_mask
            blend_mask = blend_mask[cp.newaxis, :, :].repeat(3, axis=0)
            blend_color = img_gpu.transpose(2, 0, 1) * blend_mask.astype(cp.float32)

            result = cp.asnumpy(blend_color.transpose(1, 2, 0))

        return result

    def detect_color(self, img):
        img_gpu = cp.asarray(img)
        return self.process_image(img_gpu)

    def img_binary(self, blend_line):
        with cp.cuda.Stream():
            gray_gpu = cp.asarray(cv2.cvtColor(blend_line, cv2.COLOR_BGR2GRAY))
            binary_gpu = cp.where(gray_gpu != 0, 255, 0).astype(cp.uint8)
            binary = cp.asnumpy(binary_gpu)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv2.contourArea(contour) < self.min_area:
                cv2.drawContours(binary, [contour], 0, 0, -1)

        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        return cp.asarray(binary)

    def ransac_polyfit(self, x, y):
        if len(x) < 10:
            return None
        X = cp.array(x).reshape(-1, 1)
        Y = cp.array(y)
        try:
            self.ransac_left.fit(cp.asnumpy(X), cp.asnumpy(Y))
            if self.ransac_left.score(cp.asnumpy(X), cp.asnumpy(Y)) < 0.7:
                return None
            return cp.poly1d(cp.append(self.ransac_left.estimator_.coef_, [self.ransac_left.estimator_.intercept_]))
        except:
            return None

    def validate_lane_width(self, left_fitx, right_fitx):
        lane_width = cp.mean(right_fitx - left_fitx)
        expected_width = self.img_x * 0.5
        return 0.9 * expected_width < lane_width < 1.5 * expected_width 

    def window_search(self, binary_line):
        histogram = cp.sum(binary_line[binary_line.shape[0] // 2:, :], axis=0)
        midpoint = histogram.shape[0] // 2
        left_x_base = cp.argmax(histogram[:midpoint])
        right_x_base = cp.argmax(histogram[midpoint:]) + midpoint

        out_img = cp.dstack((binary_line, binary_line, binary_line)) * 255

        nonzero = binary_line.nonzero()
        nonzeroy = cp.array(nonzero[0])
        nonzerox = cp.array(nonzero[1])

        left_lane_inds = ((nonzerox > (left_x_base - 50)) & (nonzerox < (left_x_base + 50)))
        right_lane_inds = ((nonzerox > (right_x_base - 50)) & (nonzerox < (right_x_base + 50)))

        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        left_fit = self.ransac_polyfit(cp.asnumpy(lefty), cp.asnumpy(leftx))
        right_fit = self.ransac_polyfit(cp.asnumpy(righty), cp.asnumpy(rightx))

        ploty = cp.linspace(0, binary_line.shape[0] - 1, binary_line.shape[0])

        if left_fit is None and self.previous_left_fitx is not None:
            left_fitx = self.previous_left_fitx
        elif left_fit is not None:
            left_fitx = left_fit(ploty)
        else:
            left_fitx = None

        if right_fit is None and self.previous_right_fitx is not None:
            right_fitx = self.previous_right_fitx
        elif right_fit is not None:
            right_fitx = right_fit(ploty)
        else:
            right_fitx = None

        if left_fitx is not None and right_fitx is not None:
            if not self.validate_lane_width(left_fitx, right_fitx):
                left_fitx, right_fitx = self.previous_left_fitx, self.previous_right_fitx

        if left_fitx is not None:
            self.previous_left_fitx = left_fitx
        if right_fitx is not None:
            self.previous_right_fitx = right_fitx

        out_img[lefty, leftx] = [255, 0, 0]
        out_img[righty, rightx] = [0, 0, 255]

        return cp.asnumpy(out_img), left_fitx, right_fitx, ploty

    def generate_path(self, left_fitx, right_fitx, ploty):
        center_line = left_fitx + (right_fitx - left_fitx) / 2
        path = cp.column_stack((center_line, ploty))
        if self.prev_path is not None:
            path = self.smoothing_factor * self.prev_path + (1 - self.smoothing_factor) * path
        self.prev_path = path
        return path

    def calculate_steering_angle(self, path):
        current_time = time.time()
        dt = current_time - self.previous_time
        self.previous_time = current_time

        current_pos = cp.array([self.img_x / 2, self.img_y])
        target_pos = path[-1]  # 野껋럥以덌옙占� 占쎌빘��
        error = target_pos[0] - current_pos[0]

        steering_angle = self.pid.update(error, dt)

        if hasattr(self, 'previous_steering_angle'):
            max_change = 5 
            steering_angle = cp.clip(steering_angle,
                                     self.previous_steering_angle - max_change * dt,
                                     self.previous_steering_angle + max_change * dt)

        self.previous_steering_angle = steering_angle
        return cp.clip(steering_angle, -30, 30)

    def initialize_bird_eye_view(self):
        src = cp.float32(self.roi_points)
        dst = cp.float32([
            [0, self.img_y],
            [self.img_x, self.img_y],
            [self.img_x, 0],
            [0, 0]
        ])
        self.M = cv2.getPerspectiveTransform(cp.asnumpy(src), cp.asnumpy(dst))
        self.Minv = cv2.getPerspectiveTransform(cp.asnumpy(dst), cp.asnumpy(src))

    def bird_eye_view(self, img):
        return cv2.warpPerspective(img, self.M, (self.img_x, self.img_y), flags=cv2.INTER_LINEAR)

    def reverse_bird_eye_view(self, img):
        return cv2.warpPerspective(img, self.Minv, (self.img_x, self.img_y), flags=cv2.INTER_LINEAR)

    def check_center_line(self, binary_img):
        center_width = int(self.img_x * 0.2)
        center_start = (self.img_x - center_width) // 2
        center_end = center_start + center_width

        center_region = binary_img[:, center_start:center_end]
        white_pixel_ratio = cp.sum(center_region) / (center_region.size * 255)

        return 0.15 < white_pixel_ratio < 0.4

    def process_frame(self, frame):
        try:
            self.frame_count += 1
            frame = cv2.resize(frame, (self.img_x, self.img_y))
            frame = cv2.GaussianBlur(frame, (3, 3), 0)

            if self.M is None:
                self.initialize_bird_eye_view()

            roi_mask = self.create_roi_mask(frame)
            roi_img = cv2.bitwise_and(frame, cp.asnumpy(roi_mask))

            bird_eye = self.bird_eye_view(roi_img)

            blend_img = self.detect_color(bird_eye)
            binary_img = self.img_binary(blend_img)

            is_center_line = self.check_center_line(binary_img)

            out_img, left_fitx, right_fitx, ploty = self.window_search(binary_img)

            if left_fitx is None or right_fitx is None:
                return frame, out_img, 0, "No lanes detected"

            path = self.generate_path(left_fitx, right_fitx, ploty)
          
            steering_angle = self.calculate_steering_angle(path)

            if is_center_line:
                steering_angle = 2.49

            colored_lanes = cp.zeros_like(cp.asarray(bird_eye))
            left_points = cp.array([cp.transpose(cp.vstack([left_fitx, ploty]))])
            right_points = cp.array([cp.flipud(cp.transpose(cp.vstack([right_fitx, ploty])))])
            points = cp.hstack((left_points, right_points))
            cv2.fillPoly(cp.asnumpy(colored_lanes), cp.asnumpy(points).astype(int), (0, 255, 0))

            lane_img = self.reverse_bird_eye_view(cp.asnumpy(colored_lanes))

            result = cv2.addWeighted(frame, 1, lane_img, 0.3, 0)
            cv2.polylines(result, [cp.asnumpy(cp.array(self.roi_points, dtype=cp.int32))], True, (0, 255, 0), 2)

            warped_path = cv2.perspectiveTransform(cp.asnumpy(path.reshape(-1, 1, 2)), self.Minv).reshape(-1, 2)
            cv2.polylines(result, [warped_path.astype(int)], False, (0, 255, 255), 2)

            center_bottom = (self.img_x * 45 // 100, self.img_y)
            path_end = warped_path[-1].astype(int)
            cv2.line(result, center_bottom, tuple(path_end), (0, 0, 255), 2)

            dx = path_end[0] - center_bottom[0]
            dy = path_end[1] - center_bottom[1]
            angle_with_bottom = cp.arctan2(dy, dx) * 180 / cp.pi
            steering_angle = angle_with_bottom + 90  
          
            status = "Center line" if is_center_line else "Following lanes"
            cv2.putText(result, f"Status: {status}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(result, f"Steering: {steering_angle:.2f}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(result, f"Angle with bottom: {angle_with_bottom:.2f}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            if self.frame_count % 100 == 0:
                self.memory_pool.free_all_blocks()
                self.pinned_memory_pool.free_all_blocks()

            return result, out_img, cp.asnumpy(steering_angle), status

        except Exception as e:
            logging.error(f"Error in process_frame: {e}")
            return frame, frame, 0, "Error occurred"

    def map_steering_angle_to_level(self, steering_angle):
        levels = [-16, -11, -4, 0, 9, 17, 26]
        for i in range(0,4):
            if steering_angle <= levels[i] and levels[i] <= 0:
                return i + 1 
        for i in range(6,3,-1):
            if steering_angle >= levels[i] and levels[i] >= 0:
                return i + 1

class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.previous_error = 0
        self.integral = 0

    def update(self, error, dt):
        self.integral += error * dt
        derivative = (error - self.previous_error) / dt
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.previous_error = error
        return output


class ArduinoCommunicator:
    def __init__(self, port='COM7', baudrate=115200):
        self.port = port
        self.baudrate = baudrate
        self.serial = None
        self.queue = asyncio.Queue()
        self.loop = asyncio.get_event_loop() 
        self.thread = threading.Thread(target=self.run_async_loop, daemon=True)
        self.thread.start()
        time.sleep(2)

    def run_async_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self.communicate())

    async def connect(self):
        try:
            self.serial = aioserial.AioSerial(port=self.port, baudrate=self.baudrate)
            logging.info("Connected to Arduino")
        except Exception as e:
            logging.error(f"Failed to connect to Arduino: {e}")

    async def communicate(self):
        await self.connect()
        while True:
            if self.serial:
                try:
                    data = await self.queue.get()
                    await self.serial.write_async(data.encode())
                    await asyncio.sleep(0.01)
                except Exception as e:
                    logging.error(f"Error in Arduino communication: {e}")
                    await asyncio.sleep(1)
            else:
                await asyncio.sleep(1)

    def send_data(self, data):
        self.loop.call_soon_threadsafe(self.queue.put_nowait, data)


def process_frames(lkas, frame_queue, result_queue, arduino_comm, stop_event):
    frame_count = 0
    while not stop_event.is_set():
        if not frame_queue.empty():
            frame = frame_queue.get()
            try:
                result, out_img, steering_angle, status = lkas.process_frame(frame)
                frame_count += 1
                if frame_count % lkas.frame_skip == 0:
                    steering_level = lkas.map_steering_angle_to_level(steering_angle)
                    arduino_comm.send_data(str(steering_level))

                if result_queue.qsize() < 3:
                    result_queue.put((result, out_img, steering_angle, status))
                else:
                    try:
                        result_queue.get_nowait()
                    except Queue.Empty:
                        pass
                    result_queue.put((result, out_img, steering_angle, status))
            except Exception as e:
                logging.error(f"Error processing frame: {e}")
        else:
            time.sleep(0.01)


def read_frames(cap, frame_queue, stop_event):
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            logging.info("비디오 없음")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        if frame_queue.qsize() < 5:
            frame_queue.put(frame)
        else:
            try:
                frame_queue.get_nowait()
            except Queue.Empty:
                pass
            frame_queue.put(frame)
        time.sleep(0.01)


def main():
    lkas = LKAS()
    frame_queue = Queue(maxsize=10)
    result_queue = Queue(maxsize=5)
    stop_event = threading.Event()

    arduino_comm = ArduinoCommunicator()

    remember = 3

    cap = cv2.VideoCapture("/path/your/video")
    if not cap.isOpened():
        logging.error("열수없음")
        exit()

    cv2.namedWindow('Result', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Lane Detection', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Result', 640, 360)
    cv2.resizeWindow('Lane Detection', 640, 360)

    threading.Thread(target=read_frames, args=(cap, frame_queue, stop_event), daemon=True).start()
    threading.Thread(target=process_frames, args=(lkas, frame_queue, result_queue, arduino_comm, stop_event),
                     daemon=True).start()

    last_time = time.time()
    fps = 24  

    try:
        while True:
            current_time = time.time()
            elapsed = current_time - last_time
            if elapsed < 1.0 / fps:
                time.sleep(1.0 / fps - elapsed)
                continue

            if not result_queue.empty():
                try:
                    result, out_img, angle_with_bottom, status = result_queue.get(timeout=1)

                    cv2.putText(result, f"Steering Angle: {angle_with_bottom:.2f}", (10, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    cv2.imshow('Result', result)
                    cv2.imshow('Lane Detection', out_img)
                    # logging.info(f"Status: {status}, Steering Angle: {steering_angle:.2f}")

                    last_time = time.time()

                    if(status == "Center line"):
                        steering_level = 4
                        time.sleep(2)
                    elif(status == "Following lanes"):
                        steering_level = lkas.map_steering_angle_to_level(angle_with_bottom)
                        
                    if(steering_level != None):
                        arduino_comm.send_data(str(steering_level))
                        print(steering_level)
                        remember = steering_level
                    else:
                        arduino_comm.send_data(str(remember))
                        print(remember)

                except Exception as e:
                    logging.error(f"Error processing result: {e}")
                    continue

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        stop_event.set()
        cap.release()
        cv2.destroyAllWindows()

        while not frame_queue.empty():
            frame_queue.get()
        while not result_queue.empty():
            result_queue.get()


if __name__ == "__main__":
    main()
