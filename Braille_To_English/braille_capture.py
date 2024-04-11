import cv2
import numpy as np

path = "C:/Users/mattc/Documents/GitHub/Braille-Translator/Grade-2-Braille-Example.jpg"

image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)


params = cv2.SimpleBlobDetector_Params()



params.filterByColor = 1
# detect darker blobs : 0, detect lighter blobs : 256
params.blobColor = 0

# Change thresholds
params.minThreshold = 10
params.maxThreshold = 200
 
# Filter by Area.
params.filterByArea = True
params.minArea = 100
 
# Filter by Circularity
# 1 = perfect circle, 0.785 is a square
params.filterByCircularity = True
params.minCircularity = 0.1
 
# Filter by Convexity
params.filterByConvexity = False
params.minConvexity = 0
 
# Filter by Inertia
params.filterByInertia = False
params.minInertiaRatio = 0

detector = cv2.SimpleBlobDetector(params)
# Goal is to be able to identify the different braille positions.




# Step 1. Identify all braille dots

dots = detector.detect(image)
# draws detected dots
img_with_keypoints = cv2.drawKeypoints(image, dots, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imshow('Blob Detection', img_with_keypoints)
cv2.waitKey(0)
cv2.destroyAllWindows()
    # Step 2. Create a bounding rectangle
    
    
    
    #Step 3. Split rectangle into 6ths
    
    
    #Step 4. Identify which of the six sections are "filled"

