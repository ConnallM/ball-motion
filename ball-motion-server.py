import cv2
import imutils
import math

cap = cv2.VideoCapture(0)
if not cap.isOpened():
 print("Cannot open camera")
 exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

def detectBall(start_frame):


    ret, frame = cap.read()
    cv2.imshow('Capture', frame)
    frame = imutils.resize(frame, width=500)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    difference = cv2.absdiff(frame_gray, start_frame)
    bilateral_filtered_image = cv2.bilateralFilter(difference, 5, 175, 175)
    thresh = cv2.threshold(bilateral_filtered_image, 4, 255, cv2.THRESH_BINARY)[1]

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #cv2.drawContours(frame, contours, -1, (255, 0, 0), 2)
    cv2.imshow('Thresh', thresh)

    contour_list = []
    radii = []
    if thresh.sum() > 300:
        for contour in contours:
            approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
            area = cv2.contourArea(contour)
            if ((len(approx) > 8) and (len(approx) < 23) and (area > 300)):
                (x_axis, y_axis), radius = cv2.minEnclosingCircle(contour)
                if cv2.contourArea(contour) > 0:
                    area_ratio = math.pi * radius ** 2 / cv2.contourArea(contour)
                    if (area_ratio < 1.5 and area_ratio > 1):
                        contour_list.append(contour)
                        radii.append(radius)
                        cv2.circle(frame, (int(x_axis), int(y_axis)), int(radius), (0, 255, 0), 2)


        cv2.imshow('Ball Detected', frame)

    return frame_gray

def main():
    ret, start_frame = cap.read()
    start_frame = imutils.resize(start_frame, width=500)
    start_frame = cv2.cvtColor(start_frame, cv2.COLOR_BGR2GRAY)
    start_frame = cv2.GaussianBlur(start_frame, (21, 21), 0)
    while (cap.isOpened()):
        start_frame = detectBall(start_frame)
        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


main()

