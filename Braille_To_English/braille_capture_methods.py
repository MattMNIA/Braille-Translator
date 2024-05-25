from queue import Full
import cv2
import math
import numpy as np

Braille_to_Letters = {
    "000000":" ",
    "100000":"a",
    "110000":"b",
    "100100":"c",
    "100110":"d",
    "100010":"e",
    "110100":"f",
    "110110":"g",
    "110010":"h",
    "010100":"i",
    "010110":"j",
    "101000":"k",
    "111000":"l",
    "101100":"m",
    "101110":"n",
    "101010":"o",
    "111100":"p",
    "111110":"q",
    "111010":"r",
    "011100":"s",
    "011110":"t",
    "101001":"u",
    "111001":"v",
    "010111":"w",
    "101101":"x",
    "101111":"y",
    "101011":"z",
}

Braille_to_Numbers = {
    "100000":"1",
    "110000":"2",
    "100100":"3",
    "100110":"4",
    "100010":"5",
    "110100":"6",
    "110110":"7",
    "110010":"8",
    "010100":"9",
    "010110":"0",
}

upper_case_sign = "001000"
number_sign = "001111"
letter_sign = "000011"

"""
Braille Measurements provided by up.codes/s/braille

Dot diameter: 0.059-0.063 inches
Distance between center of dots in the same cell: 0.090-0.100 inches
Distance corresponding dots in adjacent cells: 0.241-0.300 inches
Distance between corresponding dots from one cell directly below: 0.395-0.400 inches
"""
    
def find_bounds(dots):
    """Finds the bounding rectangle for the braille characters

    Args:
        dots (arr[]): list of all detected dots

    Returns:
        ints: x, y, w, h: x and y coordinate of top left corner, width, and height
    """
    
    
    min_x, min_y = dots[0].pt[0], dots[0].pt[1]
    max_y = max_x = 0
    for dot in dots:
        # uses radius to adjust rectangle to surround edges
        r = dot.size/2
        min_x = min(dot.pt[0]-r, min_x)
        min_y = min(dot.pt[1]-r, min_y)
        max_x = max(dot.pt[0]+r, max_x)
        max_y = max(dot.pt[1]+r, max_y)
    # add 1 so when converting to int it rounds up
    return int(min_x), int(min_y), int(max_x-min_x+1), int(max_y-min_y+1)

        


    
    
def crop_to_braille(img, bounding_rect):
    """crops image to only include braille section

    Args:
        img (image): image containing braille
        bounding_rect (x, y, w, h): x and y coords of top left corner, and width and height
    """
    x, y, w, h = bounding_rect
    return img[y:y+h, x:x+w]
    
    
def create_generic_detector():
    """
    Returns:
        blob detector object
    """
    params = cv2.SimpleBlobDetector_Params()

    params.filterByColor = 0
    # detect darker blobs : 0, detect lighter blobs : 256
    params.blobColor = 0

    # Change thresholds
    params.minThreshold = 10
    params.maxThreshold = 200
    
    # Filter by Area.
    params.filterByArea = True
    params.minArea = 30
    
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

def create_detector(img):
    """summary: Uses the size of image to more reliably
    detect braille dots.
    
    
    Args:
        img (image): image of braille characters
    
    Returns:
        blob detector object
    """
    
    # creates detector
    params = cv2.SimpleBlobDetector_Params()

    params.filterByColor = 1
    # detect darker blobs : 0, detect lighter blobs : 256
    params.blobColor = 0

    # Change thresholds
    params.minThreshold = 50
    params.maxThreshold = 200
    
    # Filter by Area.
    params.filterByArea = True
    # Use "find_blob_size" method to determine
    # appropriate minimum area for blobs
    area = find_blob_size(img)
    params.minArea = area
    
    # Filter by Circularity
    # 1 = perfect circle, 0.785 is a square
    params.filterByCircularity = True
    params.minCircularity = 0.6
    
    # Filter by Convexity
    params.filterByConvexity = False
    params.minConvexity = 0.1
    
    # Filter by Inertia
    params.filterByInertia = False
    params.minInertiaRatio = 0.1

    detector = cv2.SimpleBlobDetector.create(params)
    return detector


def find_blob_size(img):
    """Fixes issue with detector detecting dots that are
    too small

    Args:
        img (image): image that is being scanned for dots

    Returns:
        factor: appropriate factor for minArea for a blob
        that prevents small and insignificant blobs from 
        being picked up
    """
    area = img.size//10
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = area
    params.maxArea = area
    ret, thresh = cv2.threshold(img, 200,255, cv2.THRESH_BINARY)
    detector = cv2.SimpleBlobDetector.create(params)
    dots = detector.detect(thresh)
    while len(dots)<2 and area>2:
        area =int(area*0.75)
        params.minArea = int(area)
        detector = cv2.SimpleBlobDetector.create(params)
        dots = detector.detect(thresh)
        # circle_image = cv2.circle(circle_image, (circle_image.shape[1]//2, circle_image.shape[0]//2), int(math.sqrt(area/3.14)), (255,255,0), 1, cv2.FONT_HERSHEY_COMPLEX)
        # show_image(circle_image, "circle")
    area =int(area*0.75)
    return area



def show_image(img, image_name):
    """shows image in window

    Args:
        img (image): image to be shown
        image_name (String): name of window
    """
    try:
        h, w, c = img.shape
    except:
        h, w = img.shape
    if w>h:
        factor = 1000/(w*1.0)
    else:
        factor = 1000/(h*1.0)
    cv2.namedWindow(image_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(image_name, int(factor*w), int(factor*h))
    cv2.imshow(image_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def KeyPoint_clone(dots):
    """Deep Clones dots arra

    Args:
        dots (arr[]): array of dots to be copied
        
    Return:
         (arr[]): deep copy of dots
    """
    return [cv2.KeyPoint(x = k.pt[0], y = k.pt[1], 
            size = k.size, angle = k.angle, 
            response = k.response, octave = k.octave, 
            class_id = k.class_id) for k in dots]
    
def generate_response(dots, thresh):
    """Accounts for bugged "response" variable in OpenCV's Keypoint class
    Approximates confidence level for each keypoint

    Args:
        dots (arr[]): array of dots
        img (image): images that dots were gathered from
    """
    # cv2.namedWindow("Thresh", cv2.WINDOW_NORMAL)
    # cv2.resizeWindow("Thresh", 500, 500)
    for dot in dots:
        # cv2.namedWindow("Thresh", cv2.WINDOW_NORMAL)
        # cv2.resizeWindow("Thresh", 500, 500)
        x, y = dot.pt
        r = dot.size//2
        area = 3.14*(r)**2
        thresh_cropped = thresh[int(y-r):int(y+r+1), int(x-r):int(x+r+1)]
        black_pix = (r*2)**2 - cv2.countNonZero(thresh_cropped)
        confidence = black_pix/area
        
        dot.response = confidence
        
def group_dots(dots, dot_size):
    """find and groups dots that coorespond to the same letter

    Args:
        dots (arr[]): array of dots sorted by x coordinates
        dot_size (int): diamerter of first dot
        
    Return:
        grouped_dots (List(List())): a list filled with lists of dots grouped by x coordinate
    """
    # gap of 1.5 between horizontally
    grouped_dots = []
    grouped_dots.append([dots[0]])
    idx = 0
    for i in range(1, len(dots)):
        if abs(dots[i].pt[0] - dots[i-1].pt[0])<(2*dot_size):
            grouped_dots[idx].append(dots[i])
        else:
            idx += 1
            grouped_dots.append([dots[i]])
    return grouped_dots
    
    
def find_row(dot, x, y, w, h):
    """_summary_

    Args:
        dot (keypoint): Keypoint of braille character
        x (int): left bound for braille characters
        y (int): top bound for braille characters
        w (int): width of braille section
        h (int): height of braille section
        
    Return:
        0 if in top row
        1 if in middle row
        2 if in bottom row
    """
    # dots 1 and 4 will be at any height < y+(1.5*dot_size)
    # dots 2 and 5 will be at any height y+(1.5*dot_size) <= and < y+(3.0*dotsize)
    # dots 3 and 6 will be at any height <= y+(3.0*dot_size)
    
    if dot.pt[1] < y+(h//3):
        return 0
    elif dot.pt[1] < y+(h//3*2):
        return 1
    else:
        return 2
def add_spaces(grouped_dots):
    """uses avg_x_dist method to find average distance between letters
    if the distance between two letters is greater than the average,
    we know that that must indicate a new word

    Args:
        grouped_dots (List(List())): List of groups of dots that belong to a letter

    """
    
    avg_dist = avg_x_dist(grouped_dots)
    i = 1
    num_spaces = 0
    while i < len(grouped_dots)-1:
        # first dot in group will always be in the left column
        dist = abs(grouped_dots[i][0].pt[0]-grouped_dots[i+1][0].pt[0])
        i+=1
        if dist>avg_dist*1.2:
            grouped_dots.insert(i, None)
            i+=1
            num_spaces+=1
    
def avg_x_dist(grouped_dots):
    """finds average distance between left row of each neigboring dots
    this will be used to determine if there is enough space between dots
    to indicate a new word
    Args:
        grouped_dots (List(List())): List of groups of dots that belong to a letter

        
    Return:
        avg_dist (int): the average distance between the first column of neighboring cells
    """
    sum_dist = 0
    for i in range(0, len(grouped_dots)-1):
        # first dot in group will always be in the left column
        dist = abs(grouped_dots[i][0].pt[0]-grouped_dots[i+1][0].pt[0])
        sum_dist += dist
    return int(sum_dist/(len(grouped_dots)-1.0)+0.5)
        
def organize_cell(grouped_dots, x, y, w, h):
    """Organizes every array of dots in grouped dots, into a 
    6 index long array in which each index cooresponds to a braille
    cells dot positions
    
    eg...
    
    1  4
    2  5
    3  6
    
    Args:
        grouped_dots (List(List())): List of groups of dots that belong to a letter
        x (int): left bound for braille characters
        y (int): top bound for braille characters
        w (int): width of braille section
        h (int): height of braille section
        
    Return:
        organized_cell (List(List())): 6xN list of 1s and 0s
    """
    # split 1, 2, and 3 from 4, 5, and 6
    cell_positions = np.zeros((len(grouped_dots), 6), dtype=int)
    idx = 0
    for dots in grouped_dots:
        if dots is not None:
            base = dots[0]
            left = []
            right = []
            for dot in dots:
                # if in the same column as the leftmost dot
                if dot.pt[0] - base.pt[0] < 0.5 * base.size:
                    left.append(dot)
                else:
                    right.append(dot)
            for dot in left:
                cell_positions[idx][find_row(dot, x, y, w, h)] = 1
            for dot in right:
                cell_positions[idx][3+find_row(dot, x, y, w, h)] = 1
        idx += 1
    return cell_positions
    

def get_cell_coords(organized_cell):
    """Converts the 6 index array into a 6 character long string

    Args:
        organized_cell (List(List())): 6xN list of 1s and 0s
        
    Return:
        cell_coords (List(List())): the same list as organized cell but as strings
    """    
    cell_coords = []
    for dots in organized_cell:
        str1 = ""
        for b in dots:
            str1 += str(b)
        cell_coords.append(str1)
    return cell_coords

def cell_to_English(organized_cell):
    """Converts 6 index array of 1s and 0s to the alphabet representation

    Args:
        organized_cell (List(List())): 6xN list of 1s and 0s
        
    Return:
        letters (List): List of characters
    """
    letters = list()
    capitalize = False
    isNumber = False
    cell_coords = get_cell_coords(organized_cell)
    c = ''
    for cell in cell_coords:
        if cell == number_sign:
            isNumber = True
        elif cell == letter_sign:
            isNumber = False
        elif cell == upper_case_sign:
            capitalize = True
        else:
            if isNumber:
                if cell in Braille_to_Numbers.keys():
                    c = Braille_to_Numbers[cell]
                else: 
                    c = " "
            else:
                if cell in Braille_to_Letters.keys():
                    c = Braille_to_Letters[cell]
                    if(capitalize):
                        c = c.upper()
                    capitalize = False
                else: 
                    c = " "
            letters.append(c)
    return letters
            
                