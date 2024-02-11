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
from statistics import mean
import logging

# Get the parent directory
parent_dir = os.path.dirname(__file__)

# Setup the log file
logging.basicConfig(filename=os.path.join(parent_dir,"pigeon.log"), encoding='utf-8', level=logging.INFO)

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
        # Naah, if it works, it works
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
        if self.img_counter < 40:
            with open(os.path.join(parent_dir,f'image{self.img_counter}.jpg'), 'wb') as new_file:
                new_file.write(exif_jpg.get_file())
        self.img_counter += 1

# Main execution block
if __name__ == "__main__":
    from orbit import ISS
    from picamera import PiCamera
    cam = PiCamera()
    cam.resolution = (4056, 3040)
    pigeon = PigeonISS(ISS())

    # Record the start time
    start_time = datetime.now()

    # Fill the first slot in the deque
    pigeon.capture(cam)
    time.sleep(10)

    # List to store calculated speeds
    speed_list = []

    # Run the loop for 9 minutes
    while (datetime.now() < start_time + timedelta(minutes=9)):
        # Capture an image
        pigeon.capture(cam)

        # Calculate the time difference
        totaltime = (pigeon.d[1][1] - pigeon.d[0][1]).seconds

        # Perform image processing to detect keypoints and calculate speed
        good_matches, keypoints = processing.detect_keypoints(pigeon.d[1][0], pigeon.d[0][0])
        coordinates_1, coordinates_2 = processing.find_matching_coordinates(keypoints[0][0], keypoints[1][0], good_matches)
        average_feature_distance = processing.calculate_mean_distance(coordinates_1, coordinates_2)
        speed = processing.calculate_speed_in_kmps(average_feature_distance, 14000, totaltime)
        speed_list.append(speed)
        logging.info(f"Current speed:{str(mean(speed_list))}")
        
        time.sleep(10)
    
    # Close the camera
    cam.close()

    # Log the average speed
    logging.info(f"ISS is travelling at: {mean(speed_list)} km/s")

    # Format the estimate_kmps to have a precision of 5 significant figures
    estimate_kmps_formatted = "{:.4f}".format(mean(speed_list))

    # Write the resulting speed to a file
    result_file = os.path.join(parent_dir, "result.txt")
    with open(result_file, 'w') as file:
        file.write(estimate_kmps_formatted)

    logging.info(f"Data written to {result_file}")
    logging.info(f"Total runtime: {str(datetime.now()-start_time)}")
