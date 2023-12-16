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

class PigeonISS():
    def __init__(self, iss):
        self.iss = iss
        self.img_counter = 0
        self.d = collections.deque(maxlen=2)
        
    def capture(self, camera):
        image = np.empty(cam.resolution[::-1] + (3,), dtype=np.uint8)
        cam.capture(image, 'bgr')
        timestamp = datetime.now()
        self.d.append([image,timestamp])
        self.image = image
        self.coords = self.iss.coordinates()
        status, image_jpg_coded = cv2.imencode('.jpg', image)
        image_jpg_coded_bytes = image_jpg_coded.tobytes()
        exif_jpg = exif.Image(image_jpg_coded_bytes)
        longitude = self.coords.longitude.signed_dms()
        latitude = self.coords.latitude.signed_dms()
        exif_jpg.gps_latitude = (latitude[1],latitude[2],latitude[3])
        exif_jpg.gps_latitude_ref = "S" if latitude[0] else "N"
        exif_jpg.gps_longitude = (longitude[1],longitude[2],longitude[3])
        exif_jpg.gps_longitude_ref = "W" if longitude[0] else "E"
        exif_jpg.gps_altitude = round(self.coords.elevation.m,2)
        exif_jpg.gps_altitude_ref = exif.GpsAltitudeRef.ABOVE_SEA_LEVEL
        exif_jpg.datetime_original = timestamp.strftime("%a %d %b %Y, %H:%M:%S:%f")
        with open(f'image{self.img_counter}.jpg', 'wb') as new_file:
            new_file.write(exif_jpg.get_file())
        self.img_counter += 1
        
    def test_capture(self, first, second):
        with open(first, 'rb') as image_file:
            img = exif.Image(image_file)
            time_str = img.get("datetime_original")
            time_a = datetime.strptime(time_str, '%Y:%m:%d %H:%M:%S')
        
        with open(second, 'rb') as image_file:
            img = exif.Image(image_file)
            time_str = img.get("datetime_original")
            time_b = datetime.strptime(time_str, '%Y:%m:%d %H:%M:%S')
         
        a = cv2.imread(first)
        a = cv2.resize(a,(4056,3040))
        b = cv2.imread(second)
        b = cv2.resize(b,(4056,3040))
        self.d.append([a,0])
        self.d.append([b,0])
        return (time_b - time_a).seconds
    
if __name__ == "__main__":
    parent_dir = os.path.dirname(__file__)
    test_camera = True    
    if test_camera:
        dir_path = os.path.join(parent_dir,"test_imgs")
        file_paths = sorted(filter(os.path.isfile,glob.glob(os.path.join(dir_path, '*'))))
        print(dir_path, file_paths)
        if len(file_paths) < 2:
            raise Exception("Not enough files in the test_imgs directory")
        pigeon = PigeonISS(None)
    else:        
        from orbit import ISS
        from picamera import PiCamera
        cam = PiCamera()
        cam.resolution = (4056,3040)
        pigeon = PigeonISS(ISS())
    start_time = datetime.now()
    if not test_camera: pigeon.capture(cam)
    speed_list = []
    while (datetime.now() < start_time + timedelta(minutes=9)):
        if test_camera:
            if pigeon.img_counter+1 == len(file_paths):
                print("out of images")
                break
            totaltime = pigeon.test_capture(file_paths[pigeon.img_counter], file_paths[pigeon.img_counter+1])
            pigeon.img_counter += 1
        if not test_camera: pigeon.capture(cam)
        if not test_camera: totaltime = (pigeon.d[1][1] - pigeon.d[0][1]).seconds
        print("Time difference:", totaltime)
        good_matches, keypoints = processing.detect_keypoints(pigeon.d[1][0], pigeon.d[0][0])
        # h = processing.localize(good_matches, keypoints)
        coordinates_1, coordinates_2 = processing.find_matching_coordinates(keypoints[0][0], keypoints[1][0], good_matches)
        average_feature_distance = processing.calculate_mean_distance(coordinates_1, coordinates_2)
        
        speed = processing.calculate_speed_in_kmps(average_feature_distance, 14000, totaltime)
        speed_list.append(speed)
        print(mean(speed_list))
    print(f"ISS is travelling at: {mean(speed_list)}km/s")
    # Format the estimate_kmps to have a precision of 5 significant figures
    estimate_kmps_formatted = "{:.5f}".format(mean(speed_list))

    # Write to the file
    result_file = os.path.join(parent_dir,"result.txt") # Replace with your desired file path
    with open(result_file, 'w') as file:
        file.write(estimate_kmps_formatted)

    print("Data written to", result_file)
