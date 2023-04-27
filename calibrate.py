import random

import cv2
import numpy as np
import scipy
import math
from KUKA import YouBot
import time
import threading as thr


def mouse(event, x, y, flags, param):
    global mouseX, mouseY, vid_coords, click
    if event == 1:
        mouseX, mouseY = x, y
        if conv.any():
            new = np.dot(conv, np.array([[x, y, 1]]).T)
            # print(new / new[-1])
            click = new / new[-1]
        if vid_coords.any():
            vid_coords = np.append(vid_coords, np.array([[x, y, 1]]), axis=0)
        else:
            vid_coords = np.array([[x, y, 1]])


def mouse_point(event, x, y, flags, param):
    global point
    if event == 1:
        point[0], point[1] = x, y
        new = np.dot(conv, np.array([[x, y, 1]]).T)
        print(new / new[-1])


vid_coords = np.array(False)
point = (0, 0)
click = np.array([])

correction = [0, 0, 0]

corners_coords = np.array([[0, 0, 1], [2100, 0, 1], [2100, 2700, 1], [0, 2700, 1]])
mouseX, mouseY = 0, 0
vid = cv2.VideoCapture(2)
cv2.namedWindow('image')
cv2.setMouseCallback('image', mouse)
vid.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
vid.set(cv2.CAP_PROP_FPS, 30)
vid.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

blue_lower_bound = np.array([33, 162, 113])
blue_upper_bound = np.array([180, 255, 255])
yellow_lower_bound = np.array([0, 143, 156])
yellow_upper_bound = np.array([180, 255, 255])

robot = YouBot('192.168.88.23', ros=True, offline=False, camera_enable=False)

conv = np.array(False)
conv = np.array([[9.89893448e-03, -6.48379525e+00, 2.33262205e+03],
                 [5.78828171e+00, 1.55971661e+00, -1.46446993e+03],
                 [1.33476341e-05, 1.09126595e-03, 1.00000000e+00]])
if not conv.any():
    while True:
        ret, frame = vid.read()
        if vid_coords.any():
            for i in range(vid_coords.shape[0]):
                cv2.circle(frame, (vid_coords[i, 0], vid_coords[i, 1]), 5, (255, 0, 0), -1)
            if vid_coords.shape[0] == 4:
                break
        cv2.imshow('image', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # calculate HOMOGRAPHY
    A = []
    for i in range(4):
        xs, ys, _ = vid_coords[i, :]
        xd, yd, _ = corners_coords[i, :]
        A.append([xs, ys, 1, 0, 0, 0, -xd * xs, -xd * ys, -xd])
        A.append([0, 0, 0, xs, ys, 1, -yd * xs, -yd * ys, -yd])

    A = np.array(A)
    conv = scipy.linalg.eig(A.T @ A)[1][:, -1]
    conv /= conv[-1]
    conv = conv.reshape(3, 3)
    print(conv)

test_rect = (np.linalg.inv(conv) @ corners_coords.T).T
tr_del = test_rect[:, 2]
test_rect = test_rect[:, :2]
tr1, tr2, tr3, tr4 = test_rect
test_rect = np.array([tr1 / tr_del[0], tr2 / tr_del[1], tr3 / tr_del[2], tr4 / tr_del[3]], np.int32)


def find_robot(hsv):
    mask = cv2.inRange(hsv, blue_lower_bound, blue_upper_bound)
    mask = cv2.GaussianBlur(mask, (15, 15), 0)
    ret, mask = cv2.threshold(mask, 95, 255, cv2.THRESH_BINARY)
    M = cv2.moments(mask)
    try:
        cent_X = int(M["m10"] / M["m00"])
        cent_Y = int(M["m01"] / M["m00"])
    except:
        cent_X = 0
        cent_Y = 0
    mask = cv2.inRange(hsv, yellow_lower_bound, yellow_upper_bound)
    mask = cv2.GaussianBlur(mask, (15, 15), 0)
    ret, mask = cv2.threshold(mask, 95, 255, cv2.THRESH_BINARY)
    M = cv2.moments(mask)
    try:
        back_X = int(M["m10"] / M["m00"])
        back_Y = int(M["m01"] / M["m00"])
    except:
        back_X = 0
        back_Y = 0
    return cent_X, cent_Y, back_X, back_Y


def find_robot_abs():
    frame_lock.acquire()
    my_hsv = hsv
    frame_lock.release()
    cent_X, cent_Y, back_X, back_Y = find_robot(my_hsv)
    r_coord = np.array([cent_X, cent_Y, 1, back_X, back_Y, 1]).reshape((2, 3))
    # print(r_coord.T)
    r_coord = np.dot(conv, (r_coord.T)).T
    yfc, xfc, den_f, yrc, xrc, den_r = r_coord.reshape(6)
    xfc /= den_f
    yfc /= den_f
    xrc /= den_r
    yrc /= den_r
    ang = math.atan2(yfc - yrc, xfc - xrc)
    return xfc, yfc, ang, xrc, yrc


def full_logs(delta_t, odom1, cam_pos1, f, s, r):
    global robot
    xr, yr, zr, = robot.increment
    odom2 = np.array([xr * 1000, yr * 1000, zr])
    print('odom2',np.array([xr * 1000, yr * 1000, zr]))
    x, y, ang, x1, y1 = find_robot_abs()
    cam_pos2 = np.array((x1, y1, ang))
    print('campos2', np.array((x1, y1, ang)))
    delta_odom = abs(odom2 - odom1)
    delta_cam_pos = abs(cam_pos2 - cam_pos1)
    #s_log = str(f) + ', ' + str(s) + ', ' + str(r) + '; ' + str(delta_odom[0]) + ' ,' + str(delta_odom[1]) + ' ,' + str(
    #    delta_odom[2]) + '; ' + str(delta_cam_pos[0]) + ' ,' + str(delta_cam_pos[1]) + ' ,' + str(
    #    delta_cam_pos[2]) + '; ' + str(delta_t) + ';' + '\n'
    # s_log = str(f) + ', ' + str(s) + ', ' + str(r) + '; ' + ",".join(delta_odom) + ';' + ".".join(delta_cam_pos) + '; ' + str(delta_t) + ';' +'\n'
    s_log = ('{}, {}, {}; ' + '{}, ' * len(delta_odom) + '; ' + '{}, ' * len(delta_cam_pos) + '; ' + '{};\n').format(f, s, r, *delta_odom, *delta_cam_pos, delta_t)
    print(s_log)
    file = open("logs_KUKA", 'a')
    file.write(s_log)
    file.close()


def go_to_point():
    print('zashel v func')
    global inv_rob_mat
    global robot
    global x, y, x1, y1

    xr, yr, zr, = robot.increment  # одометрия
    print('odom0',np.array([xr * 1000, yr * 1000, zr]))
    # robot.move_base(-0.1, 0, 0)
    going = False
    while True:

        if not going:
            going = True
            f = random.randint(-30, 30) / 100
            s = random.randint(-30, 30) / 100
            r = random.randint(-30, 30) / 100

        if going:
            t1 = time.time()

            x, y, ang, x1, y1 = find_robot_abs()  # точки на поле
            xr, yr, zr, = robot.increment  # одометрия
            print('odom1',np.array([xr * 1000, yr * 1000, zr]))
            cam_pos1 = np.array((x1, y1, ang))
            print('campos1', np.array((x1, y1, ang)))
            odom1 = np.array([xr * 1000, yr * 1000, zr])
            t3 = time.time()
            while 2700 > x1 > 0 and 2100 > y1 > 0 and t3-t1<20:
                robot.move_base(f, s, r)
                x, y, ang, x1, y1 = find_robot_abs()
                t3 = time.time()
            else:
                t2 = time.time()
                robot.move_base(0, 0, 0)
                print('stop')
                time.sleep(1)
                delta_t = round(t2 - t1, 3)
                full_logs(delta_t, odom1, cam_pos1, f, s, r)
                robot.move_base(-f, -s, -r)
                time.sleep(delta_t)
                robot.move_base(0, 0, 0)
                time.sleep(1)
                full_logs(delta_t, odom1, cam_pos1, -f, -s, -r)
                x, y, ang, x1, y1 = find_robot_abs()
                going = False


ret, frame = vid.read()
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

frame_lock = thr.Lock()

moving_thr = thr.Thread(target=go_to_point)
moving_thr.start()

while True:
    frame_lock.acquire()
    ret, frame = vid.read()
    frame_lock.release()
    # cv2.createTrackbar()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    frame = cv2.polylines(frame, [test_rect], True, (255, 255, 255), 1)
    cent_X, cent_Y, back_X, back_Y = find_robot(hsv)
    xfc, yfc, ang, xrc, yrc = find_robot_abs()
    cv2.polylines(frame, [test_rect], True, (255, 255, 255), 1)
    cv2.putText(frame, f"x: {round(xrc)}", (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1,
                cv2.LINE_AA)
    cv2.putText(frame, f"y: {round(yrc)}", (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1,
                cv2.LINE_AA)
    if 'robot' in globals() and robot:
        xr, yr, zr, = robot.increment
        cv2.putText(frame, f"x2: {round(xr * 1000)}", (300, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1,
                    cv2.LINE_AA)
        cv2.putText(frame, f"y2: {round(yr * 1000)}", (300, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1,
                    cv2.LINE_AA)
        cv2.putText(frame, f"ang2: {zr}", (300, 75), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"ang: {round(ang, 5)}", (0, 75), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (255, 255, 255), 1, cv2.LINE_AA)
    cv2.circle(frame, (cent_X, cent_Y), 2, (255, 255, 255), -1)
    cv2.circle(frame, (back_X, back_Y), 2, (255, 255, 0), -1)
    cv2.circle(frame, (mouseX, mouseY), 10, (0, 163, 249), -1)
    cv2.imshow('image', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        robot.move_base(0, 0, 0)
        break

vid.release()
cv2.destroyAllWindows()
