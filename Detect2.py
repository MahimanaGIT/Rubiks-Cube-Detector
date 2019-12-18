import math
import time
import imutils
import cv2.cv2 as cv2
import numpy as np
import scipy.spatial as sp

global frame

'''
TODO:
1. Remove noise using average area and perimeter of the square after detecting each of the one.
2. Find the missing squares and the grid. Detect the rubiks cube if there is a grid of 3x3 or 4x4 
being constructed from the detected squares
3. distance between adjacent squares is the average edge length
4. de duplicacy of contours
'''

# Variables
colors = {
    "red": (255, 0, 0),
    "green": (0, 255, 0),
    "blue": (0, 0, 255),
    "white": (255, 255, 255),
    "orange": (250, 128, 0),
    "yellow": (255, 255, 0)
}

squares = []

dilate_kernel = np.ones((2, 2), np.uint8)

# Initializations
cap = cv2.VideoCapture(0)  # Read once for getting frame size
cap.set(cv2.CAP_PROP_FRAME_WIDTH,1080)
_, frame = cap.read()
frame_height, frame_width, _ = frame.shape
crop_image = frame

# Enumerate Color names
color_names = np.array(list(colors.keys()))
color_vals = np.array(list(colors.values()))

# https://stackoverflow.com/a/22478139/9799700
colors_tree = sp.KDTree(color_vals)

class Square:
    x = 0
    y = 0
    area = 0
    radius = 0
    approx = 0
    perimeter = 0
    contour = [[]]
    color_name = ""
    color = (0, 0, 0)

def getMidPoint(pt1, pt2):
    return (int((pt1[0] + pt2[0])/2), int((pt1[1] + pt2[1])/2))


def clean_squares(black, img, centroidSquare, squares):
    global crop_image

    new_squares = []

    if len(squares) > 4:  # minimum 4 Squares detected
        print("Rubiks Cube")
        largest_len_to_go = max(centroidSquare.radius,
                                int(squares[0].radius * 5))

        miny = max(0, centroidSquare.y - largest_len_to_go)
        maxy = min(frame_height, centroidSquare.y + largest_len_to_go)

        minx = max(0, int(centroidSquare.x - largest_len_to_go))
        maxx = min(frame_width, int(centroidSquare.x + largest_len_to_go))
        # print(str(miny) + ',' + str(maxy) + ',' + str(minx) + ',' + str(maxx))
        # crop_img = img[miny:maxy, minx:maxx]
        part_img = np.zeros_like(img)
        part_img[miny:maxy, minx:maxx] = 255

        crop_image = cv2.bitwise_and(img, part_img)
        
        # if maxx-minx > 0 and maxy-miny > 0:
        #     cv2.imshow("cropped", crop_img)
        slope=0
        for i in range(len(squares)):
            sq=squares[i]
            if minx <= sq.x and sq.x <= maxx and miny <= sq.y and sq.y <= maxy:
                p1=getMidPoint(sq.approx[0][0], sq.approx[1][0])
                p2=getMidPoint(sq.approx[2][0], sq.approx[3][0])
                try:
                    slope=(slope+(p1[1]-p2[1])/(p1[0]-p2[0]))/2
                except:
                    pass
        print(slope)
        for i in range(len(squares)):
            sq=squares[i]
            # To remove squares which are far away from centroid
            centroidSquare.area=0
            centroidSquare.perimeter=0
            if minx <= sq.x and sq.x <= maxx and miny <= sq.y and sq.y <= maxy:
                centroidSquare.area+=sq.area
                centroidSquare.perimeter+=sq.perimeter
        
        for i in range(len(squares)):
            sq=squares[i]
            if sq.area<centroidSquare.area*1.8 and sq.area>centroidSquare.area*0.2 and  sq.perimeter<centroidSquare.perimeter*1.8 and sq.perimeter>centroidSquare.perimeter*0.2:
                new_squares.append(sq)
                # p1=getMidPoint(sq.approx[0][0], sq.approx[3][0])
                # p2=getMidPoint(sq.approx[1][0], sq.approx[2][0])
                # p3=getMidPoint(sq.approx[0][0], sq.approx[1][0])
                # p4=getMidPoint(sq.approx[2][0], sq.approx[3][0])
                cv2.line(black, getMidPoint(sq.approx[0][0], sq.approx[1][0]),
                         getMidPoint(sq.approx[2][0], sq.approx[3][0]), (127), thickness=2)
                cv2.line(black, getMidPoint(sq.approx[0][0], sq.approx[3][0]),
                         getMidPoint(sq.approx[1][0], sq.approx[2][0]), (127), thickness=2)
                # cv2.line(black,(sq.approx[2][0][0],sq.approx[3][0][1]),(sq.approx[3][0][0],sq.approx[2][0][1]), (127), thickness=2)
                # cv2.line(black,(sq.approx[0][0][0],sq.approx[0][0][1]), (sq.approx[1][0][0],sq.approx[1][0][1]), (127), thickness=2)
               
                
                # cv2.line(black, p1, (p2[0],int((slope*(p2[0]-p1[0]))+p1[1])), (127), thickness=2)
                # cv2.line(black, p3, (p4[0],int((-1/slope)*(p4[0]-p3[0])+p3[1])), (127), thickness=2)

                # cv2.line(black, tuple(sq.approx[0][0]), tuple(
                #     sq.approx[1][0]), (150), thickness=2)
                # cv2.line(black, tuple(sq.approx[1][0]), tuple(
                #     sq.approx[2][0]), (150), thickness=2)
                # cv2.line(black, tuple(sq.approx[2][0]), tuple(
                #     sq.approx[3][0]), (150), thickness=2)
                # cv2.line(black, tuple(sq.approx[3][0]), tuple(
                #     sq.approx[0][0]), (150), thickness=2)

                try:
                    # can use the approx contour, make a mask and then avg to get mean color
                    clr = img[sq.y, sq.x]
                except:
                    clr = (0, 0, 0)

                sq.color_name, sq.color = getNearestColor(
                    (int(clr[2]), int(clr[1]), int(clr[0])))

        # cvFitLine
        print(len(new_squares))
        # lines = cv2.HoughLines(black, 1, np.pi / 180, 110)

        # if lines is not None:
        #     for i in range(0, len(lines)):
        #         rho = lines[i][0][0]
        #         theta = lines[i][0][1]
        #         a = math.cos(theta)
        #         b = math.sin(theta)
        #         x0 = a * rho
        #         y0 = b * rho
        #         pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
        #         pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
        #         cv2.line(black, pt1, pt2, (255), 3, cv2.LINE_AA)

        linesP = cv2.HoughLinesP(black, 1, np.pi / 180, 50, None, 50, 1000)
        # print(linesP)
        if linesP is not None:
            for i in range(0, len(linesP)):
                l = linesP[i][0]
                cv2.line(black, (l[0], l[1]), (l[2], l[3]),
                         (255), 2, cv2.LINE_AA)
        
        # linesP = cv2.HoughLinesP(black, 1, np.pi / 180, 50, None, 50, 1000)
        # # print(linesP)
        # if linesP is not None:
        #     for i in range(0, len(linesP)):
        #         l = linesP[i][0]
        #         cv2.line(black, (l[0], l[1]), (l[2], l[3]),
        #                  (255), 2, cv2.LINE_AA)

        # TODO get unkown squares by using the lines to get the center point intersection of 2 lines
        # TODO get rotation direction of cube using optical flow

    return new_squares


def drawSquares(img, squares):
    for sq in squares:
        cv2.drawContours(img, [sq.approx], -1, sq.color, 2)
        cv2.putText(img, sq.color_name,
                    (sq.x, sq.y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 255, 0))


def getNearestColor(colr):
    _, indx = colors_tree.query(colr)
    col = tuple((int(color_vals[indx][2]), int(
        color_vals[indx][1]), int(color_vals[indx][0])))
    return color_names[indx],  col


def angle_cos(p0, p1, p2):
    d1, d2 = (p0-p1).astype('float'), (p2-p1).astype('float')
    return abs(np.dot(d1, d2) / np.sqrt(np.dot(d1, d1)*np.dot(d2, d2)))


# Optical Flow

previous_pos = None
previous_black = None
mask = None

lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


def opticalFlow(black, frame, squares):
    global previous_black, previous_pos, mask
    pos = []
    for sq in squares:
        pos.append([(sq.x, sq.y)])

    pos = np.float32([tr[-1] for tr in pos]).reshape(-1, 1, 2)

    if len(pos) > 0:
        # init or regenerate points
        if previous_black is None or previous_pos is None or len(previous_pos) == 0:
            next_points, status, _ = cv2.calcOpticalFlowPyrLK(
                black, black, pos, None, **lk_params)
            previous_pos = next_points
            mask = np.zeros_like(frame)
        else:
            next_points, status, _ = cv2.calcOpticalFlowPyrLK(
                previous_black, black, previous_pos, None)

        if next_points is not None:
            good_new = next_points[status == 1]
            good_old = previous_pos[status == 1]

            previous_pos = good_new.reshape(-1, 1, 2)

            for _, (new, old) in enumerate(zip(good_new, good_old)):
                # Returns a contiguous flattened array as (x, y) coordinates for new point
                a, b = new.ravel()
                # Returns a contiguous flattened array as (x, y) coordinates for old point
                c, d = old.ravel()
                # Draws line between new and old position with green color and 2 thickness
                mask = cv2.line(mask, (a, b), (c, d), (255), 4)
                # Draws filled circle (thickness of -1) at new position with green color and radius of 3
                frame = cv2.circle(frame, (a, b), 3, (255), -1)

    previous_black = black


def detect_squares(frame):
    global crop_image, mask, dilate_kernel, squares

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    black = gray.copy()
    black[:] = 0
    test=black.copy()
    # gray = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,1)
    edges = cv2.Canny(gray, 50, 220, apertureSize=3)
    edges = cv2.dilate(edges, dilate_kernel, iterations=1)

    squares = []

    centroidSquare = Square()

    contours, hierarchy = cv2.findContours(
        edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    try:
        hierarchy = hierarchy[0]
    except:
        pass

    for ind in range(len(contours)):
        cnt = contours[ind]

        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, False)

        approx = cv2.approxPolyDP(cnt, 0.1 * perimeter, True)

        if area > 150 and area < 3300 and perimeter < 200:
            if len(approx) == 4:
                # # Skip if this contour has a parent
                # if hierarchy[ind][3] == -1:
                #     cv2.drawContours(
                #         frame, [approx], -1, (255, 255, 0), thickness=-1)
                #     continue
                cv2.drawContours(test, [approx], -1, (255), thickness=-1)
                ((x, y), radius) = cv2.minEnclosingCircle(cnt)
                # # OR
                # M = cv2.moments(cnt)
                # x = int(M['m10']/M['m00'])
                # y = int(M['m01']/M['m00'])
                # radius = 20

                if(radius <= 0):
                    radius = 1

                squr = Square()
                squr.x = int(x)
                squr.y = int(y)
                squr.area = area
                squr.contour = cnt
                squr.approx = approx
                squr.radius = int(radius)
                squr.perimeter = perimeter

                squares.append(squr)

                # try:
                #     clr = frame[squr.y, squr.x]
                # # Skip squares whose color we cant get (mostly center going out of the frame)
                # except:
                #     continue
                # squr.color_name, squr.color = getNearestColor(
                #     (int(clr[2]), int(clr[1]), int(clr[0])))

                # center = (squr.x, squr.y)

                # # cv2.drawContours(black, [approx], -1, (255), thickness=-1)
                # cv2.circle(frame, center, int(radius), squr.color, -1)

                # cv2.putText(frame, squr.color_name,
                #             center,
                #             cv2.FONT_HERSHEY_SIMPLEX,
                #             1,
                #             (255, 255, 0))

                # Doing average for radius, area and x,y
                centroidSquare.x += x
                centroidSquare.y += y
                centroidSquare.area += area
                centroidSquare.radius += radius

    length = len(squares)
    if length > 0:  # calc centeroid if squares are detected
        centroidSquare.x = int(centroidSquare.x/length)
        centroidSquare.y = int(centroidSquare.y/length)
        # This should not work as its not linear
        centroidSquare.area = int(centroidSquare.area/length)
        centroidSquare.radius = int(centroidSquare.radius/length)

    squares = clean_squares(black, frame, centroidSquare, squares)

    opticalFlow(black, frame, squares)
    
    # Draw optical flow
    if mask is not None:
        frame = cv2.add(frame, mask)
    
    drawSquares(frame, squares)
    # Draw centroid
    #cv2.circle(frame, (centroidSquare.x, centroidSquare.y),
     #          centroidSquare.radius, (255, 255, 255), 1)

    # print(len(contours))
    cv2.imshow('black', black)
    cv2.imshow('Out', frame)
    cv2.imshow('test', test)
    # time.sleep(0.1)

if(__name__ == '__main__'):
    while(True): 
        _, frame = cap.read()
        detect_squares(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break