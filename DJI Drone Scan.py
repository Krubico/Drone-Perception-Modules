from djitellopy import Tello
from AdaBins import models
import cv2

tello = Tello()
tello.connect()
tello.takeoff()

tello.takeoff()

tello.streamon()

tello.rotate_counter_clockwise(90)
first_wall = tello.get_frame_read()
np.array(first_wall)

tello.rotate_counter_clockwise(90)
second_wall = tello.get_frame_read()
np.array(second_wall)

# Depth inference here

cv2.imwrite('C:/Users/Krubics JH/DJI Drone Scan/DJI Drone Scan.py/Positional Scan Images', first_wall)
cv2.imwrite('C:/Users/Krubics JH/DJI Drone Scan/DJI Drone Scan.py/Positional Scan Images', second_wall)

tello.land()

