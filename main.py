'''
Much of the here used code comes from the OpenCV and Rapberry Pi Foundation documentation and we give them a big thanks for that.
'''

import cv2
import numpy as np
from datetime import datetime, timedelta
import time
import exif
import collections
import processing
import os.path
import glob
from statistics import mean

# Get the parent directory
parent_dir = os.path.dirname(__file__)

class PigeonISS():
    def __init__(self, iss):
        self.iss = iss
        self.img_counter = 0
        self.d = collections.deque(maxlen=2)

    # Method to capture images
    def capture(self, camera):
        # Capture an image using the provided camera
        image = np.empty(cam.resolution[::-1] + (3,), dtype=np.uint8)
        cam.capture(image, 'bgr')
        
        # Record timestamp and store the image in a deque
        # TODO: Actually use the dequeness of the deque
        timestamp = datetime.now()
        self.d.append([image, timestamp])
        
        # Extract ISS coordinates and create image metadata
        # This code wouldn't be possible witohut the Python EXIF module documentation
        self.coords = self.iss.coordinates()
        status, image_jpg_coded = cv2.imencode('.jpg', image)
        image_jpg_coded_bytes = image_jpg_coded.tobytes()
        exif_jpg = exif.Image(image_jpg_coded_bytes)
        longitude = self.coords.longitude.signed_dms()
        latitude = self.coords.latitude.signed_dms()
        exif_jpg.gps_latitude = (latitude[1], latitude[2], latitude[3])
        exif_jpg.gps_latitude_ref = "S" if latitude[0] else "N"
        exif_jpg.gps_longitude = (longitude[1], longitude[2], longitude[3])
        exif_jpg.gps_longitude_ref = "W" if longitude[0] else "E"
        exif_jpg.gps_altitude = round(self.coords.elevation.m, 2)
        exif_jpg.gps_altitude_ref = exif.GpsAltitudeRef.ABOVE_SEA_LEVEL
        exif_jpg.datetime_original = timestamp.strftime("%a %d %b %Y, %H:%M:%S:%f")
        
        # Save the image with updated metadata
        with open(os.path.join(parent_dir,f'image{self.img_counter}.jpg'), 'wb') as new_file:
            new_file.write(exif_jpg.get_file())
        self.img_counter += 1

    # Method to test using images loaded from a directory
    def test_capture(self, first, second):
        # Extract timestamps from two images
        with open(first, 'rb') as image_file:
            img = exif.Image(image_file)
            time_str = img.get("datetime_original")
            time_a = datetime.strptime(time_str, '%Y:%m:%d %H:%M:%S')

        with open(second, 'rb') as image_file:
            img = exif.Image(image_file)
            time_str = img.get("datetime_original")
            time_b = datetime.strptime(time_str, '%Y:%m:%d %H:%M:%S')

        # Read and resize images, then store them in the deque
        a = cv2.imread(first)
        a = cv2.resize(a, (4056, 3040))
        b = cv2.imread(second)
        b = cv2.resize(b, (4056, 3040))
        self.d.append([a, 0])
        self.d.append([b, 0])
        
        # Return the time difference in seconds
        return (time_b - time_a).seconds

# Main execution block
if __name__ == "__main__":
    test_camera = False

    # Check if testing with pre-captured images
    if test_camera:
        dir_path = os.path.join(parent_dir, "test_imgs")
        file_paths = sorted(filter(os.path.isfile, glob.glob(os.path.join(dir_path, '*'))))
        print(dir_path, file_paths)
        
        # Raise an exception if there are not enough files in the test_imgs directory
        if len(file_paths) < 2:
            raise Exception("Not enough files in the test_imgs directory")

        pigeon = PigeonISS(None)
    else:
        # Import necessary modules for ISS tracking and image capture
        from orbit import ISS
        from picamera import PiCamera
        cam = PiCamera()
        cam.resolution = (4056, 3040)
        pigeon = PigeonISS(ISS())

    # Record the start time
    start_time = datetime.now()

    # Fill the first slot in the deque
    if not test_camera:
        pigeon.capture(cam)

    # List to store calculated speeds
    speed_list = []

    # Run the loop for 9 minutes
    while (datetime.now() < start_time + timedelta(minutes=9)):
        
        if test_camera:
            # Break if all images have been processed
            if pigeon.img_counter + 1 == len(file_paths):
                print("Out of images")
                break

            # Calculate the time difference for testing images
            totaltime = pigeon.test_capture(file_paths[pigeon.img_counter], file_paths[pigeon.img_counter + 1])
            pigeon.img_counter += 1

        # Capture an image if not testing with pre-captured images
        if not test_camera:
            pigeon.capture(cam)

        # Calculate the time difference if not testing
        if not test_camera:
            totaltime = (pigeon.d[1][1] - pigeon.d[0][1]).seconds
        print("Time difference:", totaltime)

        # Perform image processing to detect keypoints and calculate speed
        good_matches, keypoints = processing.detect_keypoints(pigeon.d[1][0], pigeon.d[0][0])
        coordinates_1, coordinates_2 = processing.find_matching_coordinates(keypoints[0][0], keypoints[1][0], good_matches)
        average_feature_distance = processing.calculate_mean_distance(coordinates_1, coordinates_2)
        speed = processing.calculate_speed_in_kmps(average_feature_distance, 14000, totaltime)
        speed_list.append(speed)
        print(mean(speed_list))
    
    # Close the camera
    cam.close()

    # Print the average speed
    print(f"ISS is travelling at: {mean(speed_list)} km/s")

    # Format the estimate_kmps to have a precision of 5 significant figures
    estimate_kmps_formatted = "{:.4f}".format(mean(speed_list))

    # Write the resulting speed to a file
    result_file = os.path.join(parent_dir, "result.txt")
    with open(result_file, 'w') as file:
        file.write(estimate_kmps_formatted)

    print("Data written to", result_file)
