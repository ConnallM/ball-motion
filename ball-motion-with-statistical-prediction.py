# Import modules
import cv2
import imutils
import math
import numpy as np
import time
import warnings
from scipy import stats
from cv2.xphoto import createSimpleWB

# Set up constants for frame size
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Declare global variables
# Have to use global variables because the function is called recursively
points = []
latest_prediction = []
coefficients = []
refresh_time = time.time()
launch_angle_deg = 0
initial_velocity = 0


# # Returns a quadratic function
# def quadratic_curve(x, a, b, c):
#     # Use numpy to get a quadratic function
#     return np.polyval([a, b, c], x)

def clean_data(x, y):
    # Get the z-scores to filter out erroneous points
    z_scores = np.abs(stats.zscore(y))
    # print(z_scores)
    threshold = 1  # Adjust based on requirements
    outliers = np.where(z_scores > threshold)[0]

    # Remove outliers
    x_cleaned = np.delete(x, outliers)
    y_cleaned = np.delete(y, outliers)

    return x_cleaned, y_cleaned

# Detect the prescence of a ball in flight and draw the path and predict its future motion (should be broken into multiple functions probably)
def detect_ball(start_frame, cap):
    # Use globals
    global refresh_time
    global points
    global latest_prediction
    global coefficients

    global launch_angle_deg
    global initial_velocity

    # Read a frame from the camera, and process it
    _ret, frame = cap.read()
    # cv2.imshow('Capture', frame)
    frame = imutils.resize(frame, width=500)
    # Create a SimpleWB object (White Balance)
    wb = createSimpleWB()

    # Adjust the white balance of the image
    balanced_frame = wb.balanceWhite(frame)

    # Normalize pixel values to the range [0, 1]
    normalized_frame = balanced_frame / 255.0

    # Apply gamma correction
    corrected_frame = np.power(normalized_frame, 1.5)

    # Scale the pixel values back to the range [0, 255]
    corrected_frame = (corrected_frame * 255).astype(np.uint8)

    frame_gray = cv2.cvtColor(corrected_frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.addWeighted(frame_gray, alpha=0.5, src2=np.zeros_like(frame_gray), beta=0.5, gamma=0)

    difference = cv2.absdiff(frame_gray, start_frame)
    bilateral_filtered_image = cv2.bilateralFilter(difference, 5, 175, 175)
    thresh = cv2.threshold(bilateral_filtered_image, 16, 255, cv2.THRESH_BINARY)[1]
    cv2.imshow('Thresh', thresh)

    # Find image contours
    contours, _hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # If there is sufficient movement, check all contours to see if its a ball or not
    if thresh.sum() > 300:
        for contour in contours:
            approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
            area = cv2.contourArea(contour)

            # Filter contours by size
            if ((len(approx) > 8) and (len(approx) < 23) and (area > 500)):

                # Fit an ellipse to contours suspected to be a ball in flight and determine how elliptical it is
                ellipse = cv2.fitEllipse(contour)
                area_ratio = math.pi * ellipse[1][0] / 2 * ellipse[1][1] / 2 / cv2.contourArea(contour)
                radius = min(ellipse[1]) / 2

                # Check if the object resembles an ellipse closely enough to filter out unwanted contours further
                if 1.2 > area_ratio > 1:
                    # print(area_ratio)

                    # Draw enclosing circle to visualize, reset the refresh time, and append the center to an array
                    #cv2.circle(frame, (int(ellipse[0][0]), int(ellipse[0][1])), int(radius), (0, 255, 0), 2)
                    cv2.circle(frame, (int(ellipse[0][0]), int(ellipse[0][1])), 5, (0, 255, 0), -1)
                    refresh_time = time.time()
                    points.append((int(ellipse[0][0]), int(ellipse[0][1])))

                    # if len(points) > 0:
                    # drawTangent(frame, points[len(points) - 1], (int(ellipse[0][0]), int(ellipse[0][1])))

                    # if len(points) > 0:
                    # if math.sqrt(abs(ellipse[0][0] - points[len(points) - 1][0]) ** 2 + abs(ellipse[0][1] - points[len(points) - 1][1]) ** 2) < 200:
                    # points.append((int(ellipse[0][0]), int(ellipse[0][1])))
                    # refresh_time = time.time()
                    # else:
                    # points.append((int(ellipse[0][0]), int(ellipse[0][1])))
                    # refresh_time = time.time()

    # Init arrays
    x = []
    y = []

    if len(points) > 0:
        # Append the x and y values to their respective arrays and convert to np arrays
        for i in points:
            x.append(i[0])
            y.append(i[1])

        x = np.array(x)
        y = np.array(y)

        # Suppress RankWarning
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=np.RankWarning)

            # Fit a polynomial of degree 2 (you can adjust the degree)
            coefficients = np.polyfit(x, y, 2)
            #print(coefficients)

        if coefficients[0] < 0:
            coefficients[0] = -coefficients[0]
            coefficients[1] = -coefficients[1]


        x, y = clean_data(x, y)
        #if len(points) > 5:
        #    points = list(zip(x_cleaned, y_cleaned))
        if len(points) > 0:

            # Generate points along the fitted curve for smooth drawing
            #x_fit = np.linspace(min(x_cleaned), max(x_cleaned), 1000)
            #y_fit = np.polyval(coefficients, x_fit)

            # Generate points of predicted path
            future_x = np.linspace(0, FRAME_HEIGHT, 100)
            future_y = np.polyval(coefficients, future_x)

            # Draw points for debugging
            #for point in zip(x, y):
            #    cv2.circle(frame, point, 5, (0, 0, 255), -1)

            #for i in range(len(future_x)):
            #    cv2.circle(frame, (int(future_x[i]), int(future_y[i])), 5, (0, 255, 0), -1)

            # Only show predicted path if the number of points exceeds some minimum (in this case, 5)
            if len(points) > 5 and points[len(points) - 1][1] > points[len(points) - 2][1]:

                for i in range(len(future_x) - 1):
                    pt1 = (int(future_x[i]), int(future_y[i]))
                    pt2 = (int(future_x[i + 1]), int(future_y[i + 1]))
                    latest_prediction = list(zip(future_x, future_y))
                    cv2.line(frame, pt1, pt2, (0, 0, 255), 2)
            else:
                for i in range(len(latest_prediction) - 1):
                    pt1 = (int(latest_prediction[i][0]), int(latest_prediction[i][1]))
                    pt2 = (int(latest_prediction[i + 1][0]), int(latest_prediction[i + 1][1]))
                    cv2.line(frame, pt1, pt2, (0, 0, 255), 2)
    if len(coefficients) == 3 and len(points) > 5:
        a, b, c = coefficients

        # Calculate the launch angle
        launch_angle_rad = math.atan(2 * a * points[0][0] + b)

        # Convert the angle from radians to degrees
        if launch_angle_rad > 0:
            launch_angle_deg = math.degrees(launch_angle_rad) - 180
        else:
            launch_angle_deg = math.degrees(launch_angle_rad)

        horizontal_range = abs(points[0][0] - points[-1][0])

        # Calculate initial velocity
        if launch_angle_rad != 0:
            initial_velocity = math.sqrt((horizontal_range * 9.8) / abs(math.sin(2 * launch_angle_rad)))

        if points[0][0] > points[-1][0]:
            line_length = max(points[0][0] - FRAME_WIDTH, points[0][0], points[0][1], points[0][1] - FRAME_HEIGHT)
            end_point_x = int(points[0][0] + line_length * math.cos(launch_angle_rad + math.pi))
            end_point_y = int(points[0][1] + line_length * math.sin(launch_angle_rad + math.pi))
        else:
            line_length = max(FRAME_WIDTH - points[0][0], points[0][0], points[0][1], FRAME_HEIGHT - points[0][1])
            end_point_x = int(points[0][0] + line_length * math.cos(launch_angle_rad))
            end_point_y = int(points[0][1] + line_length * math.sin(launch_angle_rad))

        # Draw the line on the image
        cv2.line(frame, points[0], (end_point_x, end_point_y), (255, 0, 0), 2)



    # cv2.polylines(frame, [points_array], isClosed=False, color=(255, 0, 0), thickness=2)

    # Display the frame with path overlayed and ball highlighted
    cv2.imshow('Ball Detected', frame)

    #coefficients.tolist()

    # Return grayscale frame
    return frame_gray


def main():
    global refresh_time
    global points
    global latest_prediction
    global coefficients
    global launch_angle_deg
    global initial_velocity

    # Set up video capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    # Get initial frame and process
    _ret, start_frame = cap.read()
    start_frame = imutils.resize(start_frame, width=500)
    start_frame = cv2.cvtColor(start_frame, cv2.COLOR_BGR2GRAY)
    start_frame = cv2.bilateralFilter(start_frame, 5, 175, 175)

    # Loop
    while (cap.isOpened()):
        # Get the  amount of time that has passed since the last function call, and if it exceeds a threshold reset arrays
        time_dif = time.time() - refresh_time
        #print(time_dif)
        if time_dif > 1:
            points = []
            latest_prediction = []
            refresh_time = time.time()
            coefficients = []
            if launch_angle_deg != 0 and initial_velocity != 0:
                print("Launch Angle (degrees):", -launch_angle_deg)
                print("Initial Velocity:", initial_velocity, "Pixels Per Second")
                launch_angle_deg = 0
                initial_velocity = 0

        start_frame = detect_ball(start_frame, cap)

        # print(points)
        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
