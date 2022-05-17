from jetracer.nvidia_racecar import NvidiaRacecar
from time import sleep

print("Imports")
car = NvidiaRacecar()

print("!")
car.throttle = -0.5
car.steering = -1
sleep(0.5)
car.throttle = 0.0
car.steering = 0.0
