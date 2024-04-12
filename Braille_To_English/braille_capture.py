import cv2
import numpy as np
import braille_capture_methods as bc

path = "C:/Users/mattc/Documents/GitHub/Braille-Translator/Grade-2-Braille-Example.jpg"

img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
detector = bc.create_detector()


# Step 1. Identify dots

dots = bc.get_dots(img, detector)

# draws detected dots
img_with_keypoints = cv2.drawKeypoints(img, dots, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)



cv2.imshow('Blob Detection', img_with_keypoints)
cv2.waitKey(0)
cv2.destroyAllWindows()





# Step 2. Create a bounding rectangle


# Identify size of circles.

dot_size = dots[0].size

x,y,w,h = cv2.boundingRect(dots)
cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0),2)

cv2.imshow('Blob Detection', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
# Step 3. Split rectangle into 6ths
    
    
# Step 4. Identify which of the six sections are "filled"

