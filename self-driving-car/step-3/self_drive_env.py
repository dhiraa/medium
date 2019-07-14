# Self Driving Car

# Importing the libraries
import numpy as np
from random import random, randint
import matplotlib.pyplot as plt
import time, random

# Importing the Kivy packages
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.graphics import Color, Ellipse, Line
from kivy.config import Config
from kivy.properties import NumericProperty, ReferenceListProperty, ObjectProperty
from kivy.vector import Vector
from kivy.clock import Clock
from kivy.core.window import Window

rotation_angles = [0, 20, -20]
SENSOR_PROXIMITY = 30 #How far the sensor should be placed from the car/rectangle front side


class Car(Widget):
    """
    Car widget that handles the orientation of the car and sensor.
    It also takes care of how to steer/rotate the car along with its sensor
    """

    angle = NumericProperty(0) # initializing the angle of the car (angle between the x-axis of the map and the axis of the car)
    rotation = NumericProperty(0) # initializing the last rotation of the car (after playing the action, the car does a rotation of 0, 20 or -20 degrees)

    velocity_x = NumericProperty(0) # initializing the x-coordinate of the velocity vector
    velocity_y = NumericProperty(0) # initializing the y-coordinate of the velocity vector
    velocity = ReferenceListProperty(velocity_x, velocity_y) # velocity vector

    sensor1_x = NumericProperty(0) # initializing the x-coordinate of the first sensor (the one that looks forward)
    sensor1_y = NumericProperty(0) # initializing the y-coordinate of the first sensor (the one that looks forward)
    sensor1 = ReferenceListProperty(sensor1_x, sensor1_y) # first sensor vector

    sensor2_x = NumericProperty(0) # initializing the x-coordinate of the second sensor (the one that looks 30 degrees to the left)
    sensor2_y = NumericProperty(0) # initializing the y-coordinate of the second sensor (the one that looks 30 degrees to the left)
    sensor2 = ReferenceListProperty(sensor2_x, sensor2_y) # second sensor vector

    sensor3_x = NumericProperty(0) # initializing the x-coordinate of the third sensor (the one that looks 30 degrees to the right)
    sensor3_y = NumericProperty(0) # initializing the y-coordinate of the third sensor (the one that looks 30 degrees to the right)
    sensor3 = ReferenceListProperty(sensor3_x, sensor3_y) # third sensor vector

    signal1 = NumericProperty(0) # initializing the signal received by sensor 1
    signal2 = NumericProperty(0) # initializing the signal received by sensor 2
    signal3 = NumericProperty(0) # initializing the signal received by sensor 3

    def move(self, rotation):
        # updating the position of the car according to its last position and velocity
        self.pos = Vector(*self.velocity) + self.pos

        # getting the rotation of the car
        self.rotation = rotation
        # updating the angle
        self.angle = self.angle + self.rotation

        # SENSOR_PROXIMITY is how far we wanted to place the sensor infront of the car
        # So basically we are taking a vector which places the sensor with SENSOR_PROXIMITY as distance
        # a head of the car
        # Next we are placing three sensors such that one sensor at the center and other two with +/- 30 deg angle,
        #covering the entire front area
        # 360% is to make the angle with in the numeric range of angles
        self.sensor1 = Vector(SENSOR_PROXIMITY, 0).rotate((self.angle+30)%360) + self.pos # updating the position of sensor 2
        self.sensor2 = Vector(SENSOR_PROXIMITY, 0).rotate(self.angle) + self.pos # updating the position of sensor 1
        self.sensor3 = Vector(SENSOR_PROXIMITY, 0).rotate((self.angle-30)%360) + self.pos # updating the position of sensor 3

class Sensor1(Widget):
    pass
class Sensor2(Widget):
    pass
class Sensor3(Widget):
    pass

class SelfDriveEnv(Widget):

    car = ObjectProperty(None)
    sensor1 = ObjectProperty(None)
    sensor2 = ObjectProperty(None)
    sensor3 = ObjectProperty(None)

    def __init__(self, **kwargs):
        super(SelfDriveEnv, self).__init__(**kwargs)
        # Add callback to get keyboard close event
        self._keyboard = Window.request_keyboard(self._keyboard_closed, self)
        # Add callback to bind the keybaord events with environment object
        self._keyboard.bind(on_key_down=self._on_keyboard_down)

        self.ANIMATE = False

    def _keyboard_closed(self):
        self._keyboard.unbind(on_key_down=self._on_keyboard_down)
        self._keyboard = None

    def _on_keyboard_down(self, keyboard, keycode, text, modifiers):

        if keycode[1] == 'up':
            self.car.center_y += 20
            self.sensor1.center_y += 20
            self.sensor2.center_y += 20
            self.sensor3.center_y += 20
        elif keycode[1] == 'down':
            self.car.center_y -= 20
            self.sensor1.center_y -= 20
            self.sensor2.center_y -= 20
            self.sensor3.center_y -= 20
        elif keycode[1] == 'left':
            self.car.center_x -= 20
            self.sensor1.center_x -= 20
            self.sensor2.center_x -= 20
            self.sensor3.center_x -= 20
        elif keycode[1] == 'right':
            self.car.center_x += 20
            self.sensor1.center_x += 20
            self.sensor2.center_x += 20
            self.sensor3.center_x += 20
        elif keycode[1] == 'q':
            self.update(mannual=True)
        elif keycode[1] == 'w':
            self.update(mannual=True)
        elif keycode[1] == 's':
            self.ANIMATE = not self.ANIMATE
            print("Animation stopped...now you are in control. Be safe!")
        elif keycode[1] == 'escape':
            exit()
        return True

    def serve_car(self):  # starting the car when we launch the application
        self.car.center = self.center  # the car will start at the center of the map
        self.car.velocity = Vector(6, 0)  # the car will start to go horizontally to the right with a speed of 6

    def update(self, mannual=False): # the big update function that updates everything that needs to be updated at each discrete time t when reaching a new state (getting new signals from the sensors)

        if self.ANIMATE == True or mannual == True:
            self.car.move(rotation_angles[random.randint(0,2)])
            self.car.velocity = Vector(6, 0).rotate(self.car.angle)  # it goes to a normal speed (speed = 6)

            self.sensor1.pos = self.car.sensor1 # updating the position of the first sensor (ball1) right after the car moved
            self.sensor2.pos = self.car.sensor2 # updating the position of the second sensor (ball2) right after the car moved
            self.sensor3.pos = self.car.sensor3 # updating the position of the third sensor (ball3) right after the car moved

            if self.car.x < 10: # if the car is in the left edge of the frame
                self.car.x = 10 # it is not slowed down
            if self.car.x > self.width-10: # if the car is in the right edge of the frame
                self.car.x = self.width-10 # it is not slowed down
            if self.car.y < 10: # if the car is in the bottom edge of the frame
                self.car.y = 10 # it is not slowed down
            if self.car.y > self.height-10: # if the car is in the upper edge of the frame
                self.car.y = self.height-10 # it is not slowed down



class SelfDriveApp(App):

    def build(self):
        self.title = 'Self Driving Car'
        parent = SelfDriveEnv()
        parent.serve_car()
        Clock.schedule_interval(parent.update, 10.0 / 60.0)
        return  parent



if __name__ == '__main__':
    SelfDriveApp().run()
