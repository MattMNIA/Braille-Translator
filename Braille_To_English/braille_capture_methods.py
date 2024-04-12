import cv2
import numpy as np


"""
Braille Measurements provided by up.codes/s/braille

Dot diameter: 0.059-0.063 inches
Distance between dots in the same cell: 0.090-0.100 inches
Distance corresponding dots in adjacent cells: 0.241-0.300 inches
Distance between corresponding dots from one cell directly below: 0.395-0.400 inches
"""
    
    
    
    
    
    
    
    
    
    
    
def create_detector():
    """
    Returns:
        blob detector object
    """
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
    params.minCircularity = 0.8
    
    # Filter by Convexity
    params.filterByConvexity = False
    params.minConvexity = .1
    
    # Filter by Inertia
    params.filterByInertia = False
    params.minInertiaRatio = .1

    detector = cv2.SimpleBlobDetector.create(params)
    return detector

def get_dots(self, img, detector):
    """
    Args:
        img (image): image of braille
        detector (SimpleBlobDetector): Detects blobs of certain size and roundness

    Returns:
        arr[] of KeyPoints: returns keypoints of dots in image
    """
    return detector.detect(img)

def show_image(img, image_name):
    """shows image in window

    Args:
        img (image): image to be shown
        image_name (String): name of window
    """
    cv2.imshow(image_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def find_corners(dots):
    """ given array of dots, find x, y, width, and height of the bounding
    rectangle of the braille characters
    Args:
        dots (array of KeyPoints): Array containing information about all braill dots
        
    Returns:
        x, y, w, h: coordinate of top left corner, width, height
    """
    
