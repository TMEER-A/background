
import cv2
import numpy as np
import math

def count_fingers(contour, drawing):
    hull = cv2.convexHull(contour, returnPoints=False)
    if len(hull) > 3:
        defects = cv2.convexityDefects(contour, hull)
        if defects is None:
            return 0
        count = 0
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(contour[s][0])
            end = tuple(contour[e][0])
            far = tuple(contour[f][0])

            # Find length of all sides of triangle
            a = math.dist(start, end)
            b = math.dist(start, far)
            c = math.dist(end, far)
            angle = math.acos((b**2 + c**2 - a**2)/(2*b*c)) * 57

            if angle <= 90:
                count += 1
                cv2.circle(drawing, far, 8, [211, 84, 0], -1)
        return count
    return 0

cap = cv2.VideoCapture('pro video.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    roi = frame[100:400, 100:400]
    cv2.rectangle(frame, (100, 100), (400, 400), (0, 255, 0), 2)

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    mask = cv2.GaussianBlur(mask, (5, 5), 100)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        max_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(max_contour) > 1000:
            drawing = np.zeros(roi.shape, np.uint8)
            cv2.drawContours(drawing, [max_contour], -1, (0, 255, 0), 2)
            fingers = count_fingers(max_contour, drawing)
            cv2.putText(frame, f"Fingers: {fingers}", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()