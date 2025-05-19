import cv2
import numpy as np

# Load the user image
user_img = cv2.imread('user.jpg')

# Load the shirt image (with transparent background)
shirt_img = cv2.imread('shirt.png', cv2.IMREAD_UNCHANGED)  # Ensure it's RGBA

def overlay_image(bg, fg, x, y):
    """Overlay fg image on bg at (x, y), where fg is RGBA"""
    fg_h, fg_w = fg.shape[:2]

    # Clip if out of bounds
    if y + fg_h > bg.shape[0] or x + fg_w > bg.shape[1]:
        return bg

    overlay = fg[..., :3]  # RGB
    mask = fg[..., 3:] / 255.0  # Alpha channel

    roi = bg[y:y+fg_h, x:x+fg_w]

    for c in range(3):
        roi[..., c] = roi[..., c] * (1 - mask[..., 0]) + overlay[..., c] * mask[..., 0]

    bg[y:y+fg_h, x:x+fg_w] = roi
    return bg

# Resize shirt to fit the user manually (adjust based on your image)
shirt_width = 250
shirt_height = 250
resized_shirt = cv2.resize(shirt_img, (shirt_width, shirt_height))

# Choose a fixed position for overlay (adjust for your user image)
x_position = 150  # Horizontal position
y_position = 200  # Vertical position

# Overlay shirt on user image
output_img = overlay_image(user_img.copy(), resized_shirt, x_position, y_position)

# Show and save result
cv2.imshow('Virtual Try-On (Simple)', output_img)
cv2.imwrite('output_tryon.jpg', output_img)
cv2.waitKey(0)
cv2.destroyAllWindows()