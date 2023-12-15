if __name__ == "__main__":

    from picamera import PiCamera

    from time import sleep

    import cv2 as cv

    import numpy as np
    import math
    

    cam = PiCamera()

    

    cam.resolution = (500, 500)

    cam.capture("image1.jpg")
    
    cam.capture("image2.jpg")

    img_object = cv.imread("image1.jpg")

    img_scene = cv.imread("image2.jpg")

    

    

    if img_object is None or img_scene is None:

        print('Could not open or find the images!')

        exit(0)

    #-- Step 1: Detect the keypoints using SURF Detector, compute the descriptors

    minHessian = 400

    detector = cv.SIFT_create()

    keypoints_obj, descriptors_obj = detector.detectAndCompute(img_object, None)

    keypoints_scene, descriptors_scene = detector.detectAndCompute(img_scene, None)

    #-- Step 2: Matching descriptor vectors with a FLANN based matcher

    # Since SURF is a floating-pdoint descriptor NORM_L2 is used

    matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_FLANNBASED)

    knn_matches = matcher.knnMatch(descriptors_obj, descriptors_scene, 2)

    #-- Filter matches using the Lowe's ratio test

    ratio_thresh = 0.75

    good_matches = []

    for m,n in knn_matches:

        if m.distance < ratio_thresh * n.distance:

            good_matches.append(m)

    #-- Draw matches

    img_matches = np.empty((max(img_object.shape[0], img_scene.shape[0]), img_object.shape[1]+img_scene.shape[1], 3), dtype=np.uint8)

    cv.drawMatches(img_object, keypoints_obj, img_scene, keypoints_scene, good_matches, img_matches, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    #-- Localize the object

    obj = np.empty((len(good_matches),2), dtype=np.float32)

    scene = np.empty((len(good_matches),2), dtype=np.float32)

    for i in range(len(good_matches)):

        #-- Get the keypoints from the good matches

        obj[i,0] = keypoints_obj[good_matches[i].queryIdx].pt[0]

        obj[i,1] = keypoints_obj[good_matches[i].queryIdx].pt[1]

        scene[i,0] = keypoints_scene[good_matches[i].trainIdx].pt[0]

        scene[i,1] = keypoints_scene[good_matches[i].trainIdx].pt[1]

    H, _ =  cv.findHomography(obj, scene, cv.RANSAC)

    #-- Get the corners from the image_1 ( the object to be "detected" )

    obj_corners = np.empty((4,1,2), dtype=np.float32)

    obj_corners[0,0,0] = 0

    obj_corners[0,0,1] = 0

    obj_corners[1,0,0] = img_object.shape[1]

    obj_corners[1,0,1] = 0

    obj_corners[2,0,0] = img_object.shape[1]

    obj_corners[2,0,1] = img_object.shape[0]

    obj_corners[3,0,0] = 0

    obj_corners[3,0,1] = img_object.shape[0]

    scene_corners = cv.perspectiveTransform(obj_corners, H)

    #-- Draw lines between the corners (the mapped object in the scene - image_2 )

    cv.line(img_matches, (int(scene_corners[0,0,0] + img_object.shape[1]), int(scene_corners[0,0,1])),\

        (int(scene_corners[1,0,0] + img_object.shape[1]), int(scene_corners[1,0,1])), (0,255,0), 4)

    cv.line(img_matches, (int(scene_corners[1,0,0] + img_object.shape[1]), int(scene_corners[1,0,1])),\

        (int(scene_corners[2,0,0] + img_object.shape[1]), int(scene_corners[2,0,1])), (0,255,0), 4)

    cv.line(img_matches, (int(scene_corners[2,0,0] + img_object.shape[1]), int(scene_corners[2,0,1])),\

        (int(scene_corners[3,0,0] + img_object.shape[1]), int(scene_corners[3,0,1])), (0,255,0), 4)

    cv.line(img_matches, (int(scene_corners[3,0,0] + img_object.shape[1]), int(scene_corners[3,0,1])),\

        (int(scene_corners[0,0,0] + img_object.shape[1]), int(scene_corners[0,0,1])), (0,255,0), 4)

    #-- Show detected matches
    resized_picture = cv.resize(img_matches, (1280, 720))    
    
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
        all_distances = 0
        merged_coordinates = list(zip(coordinates_1, coordinates_2))
        for coordinate in merged_coordinates:
            x_difference = coordinate[0][0] - coordinate[1][0]
            y_difference = coordinate[0][1] - coordinate[1][1]
            distance = math.hypot(x_difference, y_difference)
            all_distances = all_distances + distance
        return all_distances / len(merged_coordinates)
    coordinates_1, coordinates_2 = find_matching_coordinates(keypoints_obj, keypoints_scene, good_matches)
    average_feature_distance = calculate_mean_distance(coordinates_1, coordinates_2)
    def calculate_speed_in_kmps(feature_distance, GSD, time_difference):
        distance = feature_distance * GSD / 100000
        speed = distance / time_difference
        return speed
    #set time_difference just for a test
    time_difference = 5
    speed = calculate_speed_in_kmps(average_feature_distance, 12648, time_difference)
    print(speed)
    #cv.imshow('Good Matches & Object detection', resized_picture)
    cv.waitKey(0)