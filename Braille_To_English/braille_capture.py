import cv2
import numpy as np
import braille_capture_methods as bc
import math
from pathlib import Path

#path = "C:/Users/mattc/Documents/GitHub/Braille-Translator/Grade-2-Braille-Example.jpg"

#path = r"C:\Users\mattc\Documents\GitHub\Braille-Translator\Dorm_Braille.JPG"

path = r"C:\Users\mattc\Documents\GitHub\Braille-Translator\Hello_World_Braille.png"

path = r"C:\Python Projects\Braille Translator\Braille-Translator-2\testcase1.png"

#path = r"C:\Python Projects\Braille Translator\Braille-Translator-1\Dorm_Braille_noLetters.JPG"

path = r"C:\Python Projects\Braille Translator\Braille-Translator-2\image.png"
# dot color = 0 if black, 1 if white
dot_color = 0



img = cv2.imread(path)
# adjust size to increase resolution of bad photos
# 500,000 pixels is the ideal minium resolution
area = img.shape[0]*img.shape[1]
area_factor = 500000.0/area
if(area_factor>1):
    print("resize")
    img = cv2.resize(img, (int(img.shape[1]*math.sqrt(area_factor)), int(img.shape[0]*math.sqrt(area_factor))), interpolation= cv2.INTER_LINEAR)
    
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

blur = cv2.GaussianBlur(gray, (5,5), 1)
# Create Binary image

ret1, thresh = cv2.threshold(blur, 200,255, cv2.THRESH_BINARY)

if dot_color == 1:
    thresh = cv2.bitwise_not(thresh)

detector = bc.create_detector(gray, thresh)

# Step 1. Identify dots

dots = detector.detect(thresh)

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
bc.generate_response(dots, thresh)
dots_confidence = dots + ()
dots_x = dots + ()
dots_y = dots + ()
# sorts dots based on confidence
dots_confidence = sorted(dots_confidence, key=lambda KeyPoint: KeyPoint.response, reverse=True)
# sorts dots based on x value
dots_x = sorted(dots_x, key=lambda KeyPoint: KeyPoint.pt[0])
# sorts dots based on y value
dots_y = sorted(dots_y, key=lambda KeyPoint: KeyPoint.pt[1])
# group dots from the same cell
grouped_dots = bc.group_dots(dots_x, dot_size)
bc.add_spaces(grouped_dots)
# put into 2x3 matrix
cell_coords = bc.organize_cell(grouped_dots, x, y, w, h)

letters = bc.cell_to_English(cell_coords)

print(letters)

