import cv2
import numpy as np
import scipy
import math
from KUKA import KUKA
import time
import threading as thr


def mouse(event, x, y, flags, param):
    global mouseX, mouseY, vid_coords
    if event == 1:
        mouseX, mouseY = x, y
        if conv.any():
            new = np.dot(conv, np.array([[x, y, 1]]).T)
            print(new / new[-1])
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

correction = [0, 0, 0]

corners_coords = np.array([[0, 0, 1], [2500, 0, 1], [2500, 4550, 1], [0, 4550, 1]])
mouseX, mouseY = 0, 0
vid = cv2.VideoCapture(0)
cv2.namedWindow('image')
cv2.setMouseCallback('image', mouse)
vid.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
vid.set(cv2.CAP_PROP_FPS, 30)
vid.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

blue_lower_bound = np.array([33, 75, 95])
blue_upper_bound = np.array([180, 255, 255])
yellow_lower_bound = np.array([14, 170, 95])
yellow_upper_bound = np.array([95, 255, 255])

conv = np.array(False)
conv = np.array([[-1.21395537e+00, -1.85796898e+01,  5.29410053e+03],
 [ 8.54023496e+00,  4.45781499e+00, -1.64141440e+03],
 [-7.28641955e-05,  3.16672104e-03,  1.00000000e+00]])
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



    ret, frame = vid.read()
    cv2.setMouseCallback('image', mouse_point)
    print(point)
    cv2.circle(frame, point, 5, (255, 255, 0), -1)


# draw test rectangle
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


centering = 1
need_ros_restart = False
inv_rob_mat = np.array(False)


def center_robot():
    global centering
    global inv_rob_mat
    global need_ros_restart
    global robot
    if not robot.going_to_target_pos:
        print("check pos")
        x, y, ang, _, _ = find_robot_abs()
        x -= 1650
        y -= 1650
        odom = np.array([xr * 1000, yr * 1000, zr])
        cam_pos = np.array((x, y, ang))
        targ = odom - cam_pos
        xi, yi, angi = targ
        xi /= 1000
        yi /= 1000
        if math.sqrt(xi ** 2 + yi ** 2) > 0.5:
            need_ros_restart = True
        print("correct error")
        print(targ)
        robot.go_to(xi, yi, angi, prec=0.01, k=2, initial_speed=1)
        centering += 1


def rand_pos():
    x, y, ang = np.random.random(3)
    x = (x - 0.5) * 0.5  # * 3
    y = (y - 0.5) * 0.5  # * 3
    ang = (ang - 0.5) * 2  # * 3.1415
    return x, y, ang


def grab_data_from_frame():
    global xr, yr, zr, ret, frame, hsv, cent_X, cent_Y, back_X, back_Y, xfc, yfc, ang, xrc, yrc, robot
    frame_lock.acquire()
    my_hsv = hsv
    frame_lock.release()
    if robot.increment:
        xr, yr, zr, = robot.increment
    else:
        xr, yr, zr, = 0, 0, 0
    cent_X, cent_Y, back_X, back_Y = find_robot(my_hsv)
    xfc, yfc, ang, xrc, yrc = find_robot_abs()


robot = None


def log_odom():
    global centering
    global inv_rob_mat
    global need_ros_restart
    global robot
    global xr, yr, zr, ret, frame, hsv, cent_X, cent_Y, back_X, back_Y, xfc, yfc, ang, xrc, yrc
    robot = KUKA('192.168.88.23', ros=False, offline=False)
    now_going = True
    last_grab = 0
    xr, yr, zr = 0, 0, 0
    cent_X, cent_Y, back_X, back_Y = 0, 0, 0, 0
    xfc, yfc, ang = 0, 0, 0
    xrc, yrc = 0, 0
    while True:
        grab_data_from_frame()
        if not 3300 > xfc > 0 or not 3300 > yfc > 0 and not centering:
            print("centering")
            centering = 1
        if centering == 1:
            center_robot()
        elif centering == 2:
            if not robot.going_to_target_pos:
                centering = 0
                now_going = False
                last_grab = time.time()
                print("grab")
                time.sleep(1)
                if need_ros_restart:
                    need_ros_restart = False
                    del robot
                    time.sleep(2)
                    robot = KUKA('192.168.88.23', ros=True, offline=False)
                    time.sleep(1)

        if not 2500 > xfc > 800 or not 2500 > yfc > 800 or time.time() - last_grab > 4:
            if not centering:
                last_grab = time.time()
                robot.move_base(0, 0, 0)
                time.sleep(1)
                print("grab")
                grab_data_from_frame()
                # robot.go_to(0, 0, 0, prec=0.2, k=4, initial_speed=0.6)
                now_going = True
                centering = 1
        if not now_going:
            last_grab = time.time()
            x, y, z = point[0], point[1], point[2]
            print("rand")
            robot.move_base(x, y, z)
            now_going = True


ret, frame = vid.read()
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

frame_lock = thr.Lock()
odom_thr = thr.Thread(target=log_odom, args=())
odom_thr.start()

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
    cv2.putText(frame, f"x: {round(xfc)}", (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1,
                cv2.LINE_AA)
    cv2.putText(frame, f"y: {round(yfc)}", (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1,
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
    cv2.imshow('image', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        robot.move_base(0, 0, 0)
        break
vid.release()
cv2.destroyAllWindows()
