
import cv2

# Read the image
image = cv2.imread('./output_07/image/left.png')

# Resize the image
resized_image = cv2.resize(image, (512, 512))

# Save the resized image
cv2.imwrite('./output_resize/resize_left.png', resized_image)

