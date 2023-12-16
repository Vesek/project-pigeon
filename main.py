import cv2
import numpy as np
from datetime import datetime, timedelta
import time
import exif
import collections
import processing
import os.path
import glob

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
         
        aaa= cv2.imread(first)
        be = cv2.imread(second)
        self.d.append([aaa,0])
        self.d.append([be,0])
        return (time_b - time_a).seconds
    
if __name__ == "__main__":
    test_camera = True    
    if test_camera:
        dir_path = os.path.join(os.path.dirname(__file__),"test_imgs")
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
    while (datetime.now() < start_time + timedelta(minutes=1)):
        if test_camera:
            totaltime = pigeon.test_capture(file_paths[pigeon.img_counter], file_paths[pigeon.img_counter+1])
            pigeon.img_counter += 1
        if not test_camera: pigeon.capture(cam)
        print(len(pigeon.d))
        if not test_camera: totaltime = (pigeon.d[1][1] - pigeon.d[0][1]).seconds
        print(totaltime)
        good_matches, keypoints = processing.detect_keypoints(pigeon.d[1][0], pigeon.d[0][0])
        # h = processing.localize(good_matches, keypoints)
        coordinates_1, coordinates_2 = processing.find_matching_coordinates(keypoints[0][0], keypoints[1][0], good_matches)
        average_feature_distance = processing.calculate_mean_distance(coordinates_1, coordinates_2)
        
        speed = processing.calculate_speed_in_kmps(average_feature_distance, 13666, totaltime)
        print(speed)
        time.sleep(1)
