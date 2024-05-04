import cv2
import numpy as np
import braille_capture_methods as bc
from pathlib import Path

path = "C:/Users/mattc/Documents/GitHub/Braille-Translator/Grade-2-Braille-Example.jpg"

# path = r"C:\Users\mattc\Documents\GitHub\Braille-Translator\Dorm_Braille.JPG"

# path = r"C:\Users\mattc\Documents\GitHub\Braille-Translator\Hello_World_Braille.png"
# dot color = 0 if black, 1 if white
dot_color = 0



img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
blur = cv2.GaussianBlur(img, (5,5), 1)
# Create Binary image
ret, thresh = cv2.threshold(blur, 200,255, cv2.THRESH_BINARY)
if dot_color == 1:
    thresh = cv2.bitwise_not(thresh)
bc.show_image(thresh, "img")

detector = bc.create_detector(thresh)

# Step 1. Identify dots

dots = detector.detect(img)

# draws detected dots


img_with_keypoints = cv2.drawKeypoints(img, dots, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
x,y,w,h = bc.find_bounds(dots)
cropped = bc.crop_to_braille(img_with_keypoints, (x, y, w, h))
bc.show_image(cropped, "fo show")



# Step 2. Create a bounding rectangle


# Identify size of circles.
try:
    dot_size = dots[0].size
except:
    raise Exception("No dots found...")
    
x,y,w,h = bc.find_bounds(dots)
# cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0),2)


cropped = bc.crop_to_braille(img, (x, y, w, h))

# sort dots by confidence
bc.generate_response(dots, img)
dots_confidence = dots + ()
dots_x = dots + ()
dots_y = dots + ()
# sorts dots based on confidence
dots_confidence = sorted(dots_confidence, key=lambda KeyPoint: KeyPoint.response, reverse=True)
# sorts dots based on x value
dots_x = sorted(dots_x, key=lambda KeyPoint: KeyPoint.pt[0])
# sorts dots based on y value
dots_y = sorted(dots_y, key=lambda KeyPoint: KeyPoint.pt[1])
grouped_dots = bc.group_dots(dots_x, dot_size)
# put into 2x3 matrix
bc.organize_cell(grouped_dots)

letters = bc.cell_to_English(grouped_dots)

print(letters)

