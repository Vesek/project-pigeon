from time import sleep, time
import cv2 as cv
import numpy as np
from scipy import stats
import math

def detect_keypoints(obj, scene):
    images = [obj, scene]
    '''
    Detect the keypoints using SURF Detector, compute the descriptors
    '''
    keypoints = []
    detector = cv.SIFT_create()
    for image in images:
        keypoints.append(detector.detectAndCompute(image, None)) # Returns keypoints and descriptors

    '''
    Matching descriptor vectors with a FLANN based matcher
    Since SURF is a floating-pdoint descriptor NORM_L2 is used
    '''
    matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_FLANNBASED) 
    knn_matches = matcher.knnMatch(keypoints[0][1], keypoints[1][1], 2)

    # Filter matches using the Lowe's ratio test
    ratio_thresh = 0.75
    good_matches = []
    for m,n in knn_matches:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)
    return good_matches, keypoints
            
def draw(obj, scene, good_matches, keypoints):
    '''
    Draw matches
    '''
    img_matches = np.empty((max(obj.shape[0], scene.shape[0]), obj.shape[1]+scene.shape[1], 3), dtype=np.uint8)
    cv.drawMatches(obj, keypoints[0][0], scene, keypoints[1][0], good_matches, img_matches, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    return img_matches
    
def localize(good_matches, keypoints):
    '''
    Localize the object
    '''
    obj = np.empty((len(good_matches),2), dtype=np.float32)
    scene = np.empty((len(good_matches),2), dtype=np.float32)
    
    for i in range(len(good_matches)):
        #Get the keypoints from the good matches
        obj[i,0] = keypoints[0][0][good_matches[i].queryIdx].pt[0]
        obj[i,1] = keypoints[0][0][good_matches[i].queryIdx].pt[1]
        scene[i,0] = keypoints[1][0][good_matches[i].trainIdx].pt[0]
        scene[i,1] = keypoints[1][0][good_matches[i].trainIdx].pt[1]
        
        H, _ =  cv.findHomography(obj, scene, cv.RANSAC)

        return H

def find_matching_coordinates(keypoints_1, keypoints_2, matches):
    coordinates_1 = []
    coordinates_2 = []
    for match in matches:
        image_object_idx = match.queryIdx
        image_scene_idx = match.trainIdx
        (x1,y1) = keypoints_1[image_object_idx].pt
        (x2,y2) = keypoints_2[image_scene_idx].pt
        coordinates_1.append((x1,y1))
        coordinates_2.append((x2,y2))
    return coordinates_1, coordinates_2

def calculate_mean_distance(coordinates_1, coordinates_2):
    all_distances = []
    merged_coordinates = list(zip(coordinates_1, coordinates_2))
    for coordinate in merged_coordinates:
        x_difference = coordinate[0][0] - coordinate[1][0]
        y_difference = coordinate[0][1] - coordinate[1][1]
        distance = math.hypot(x_difference, y_difference)
        all_distances.append(distance)
        median_value = np.median(all_distances)
        if isinstance(median_value, np.ndarray) and len(median_value) == 2:
            median_value = np.median(median_value)
        
        first_mode = stats.mode(all_distances)
        modes = first_mode.mode
        mode_value = np.median(modes)
        
    return (median_value + mode_value)/2

def calculate_speed_in_kmps(feature_distance, GSD, time_difference):
    distance = feature_distance * GSD / 100000
    speed = distance / time_difference
    return speed

if __name__ == "__main__":
    from picamera import PiCamera
    
    #test:
    cam = PiCamera()    
    cam.resolution = (500, 500)
    timestamp1 = time()
    print("timestamp1 is: ", timestamp1)
    cam.capture("image1.jpg")
    cam.capture("image2.jpg")
    time_difference = time() - timestamp1
    print("timedifference is: ", time_difference)
    #time_difference = 7
    img_object = cv.imread("image1.jpg")
    img_scene = cv.imread("image2.jpg")
    if img_object is None or img_scene is None:
        print('Could not open or find the images!')
        exit(0)
    
    good_matches, keypoints = detect_keypoints(img_object, img_scene)
    h = localize(good_matches, keypoints)
    img = draw(img_object, img_scene, good_matches, keypoints)
    
    #-- Show detected matches
    resized_picture = cv.resize(img, (1280, 720))    
        
    coordinates_1, coordinates_2 = find_matching_coordinates(keypoints[0][0], keypoints[1][0], good_matches)
    average_feature_distance = calculate_mean_distance(coordinates_1, coordinates_2)
    
    speed = calculate_speed_in_kmps(average_feature_distance, 12648, time_difference)
    print("The speed of ISS is ", speed, "(km/s)")
    cv.waitKey(0)