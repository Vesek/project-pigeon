from orbit import ISS
from picamera import PiCamera
import cv2
import numpy as np
from datetime import datetime, timedelta
import time
import exif
import collections

cam = PiCamera()
cam.resolution = (4056,3040)
cam.framerate = 24

class PigeonISS():
    def __init__(self):
        self.iss = ISS()
        self.img_counter = 0
        self.d = collections.deque(maxlen=5)
        
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
        exif_jpg.datetime_original = timestamp.strftime("%a %d %b %Y, %H:%M")
        with open(f'image{self.img_counter}.jpg', 'wb') as new_file:
            new_file.write(exif_jpg.get_file())
        self.img_counter += 1
        
    
if __name__ == "__main__":
    pigeon = PigeonISS()
    start_time = datetime.now()
    pigeon.capture(cam)
    while (datetime.now() < start_time + timedelta(minutes=3)):
        pigeon.capture(cam)
        print(len(pigeon.d))
        time.sleep(1)    
