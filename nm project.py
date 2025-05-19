import cv2
import numpy as np

# Load background image
background = cv2.imread('flower.jpg')
if background is None:
    raise FileNotFoundError("Error: 'flower.jpg' not found.")

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Error: Could not open webcam.")

# Read a frame to get dimensions
ret, frame = cap.read()
if not ret:
    cap.release()
    raise RuntimeError("Error: Could not read from webcam.")

# Resize background to match frame size
background = cv2.resize(background, (frame.shape[1], frame.shape[0]))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Mirror effect

    # Convert frame to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define range for green screen (you can adjust these values)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])

    # Create mask to detect green background
    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    # Invert mask to keep foreground (person)
    mask_inv = cv2.bitwise_not(green_mask)

    # Extract the foreground (person)
    foreground = cv2.bitwise_and(frame, frame, mask=mask_inv)

    # Extract background where green was detected
    background_part = cv2.bitwise_and(background, background, mask=green_mask)

    # Combine both parts
    final = cv2.add(foreground, background_part)

    # Display result
    cv2.imshow("Green Screen Background Replacement", final)

    # Exit with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
