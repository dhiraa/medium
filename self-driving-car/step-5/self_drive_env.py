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

from torch_ai import DeepQLearningNetwork
class SelfDriveEnvData():
    """
    A utility class that encapsulates the car environment related variables that
    needs to be shared and updated across the widgets
    """
    def __init__(self):
        self.one_time_flag = True
        self.action2rotation = [0, 20,-20]  # action = 0 => no rotation, action = 1 => rotate 20 degres, action = 2 => rotate -20 degres

        # How far the sensor should be placed from the car/rectangle front side
        self.SENSOR_PROXIMITY = 30
        self.screen_width = -1
        self.screen_height = -1
        self.sand_pixel_matrix = None

        # Introducing last_x and last_y, used to keep the last point in memory when we draw the sand on the map
        self.last_mouse_pos_x = 0
        self.last_mouse_pos_y = 0
        self.n_points = 0  # the total number of points in the last drawing
        self.length = 0  # the length of the last drawing
        self.last_reward = 0.  # initializing the last reward
        self.scores = []  # initializing the mean score curve (sliding window of the rewards) with respect to time
        # Initializing the last distance
        self.last_distance = 0.

    def one_time_updates(self):
        if self.one_time_flag:
            self.sand_pixel_matrix = np.zeros((self.screen_width, self.screen_height))
            self.goal_x = 20  # the goal to reach is at the upper left of the map (the x-coordinate is 20 and not 0 because the car gets bad reward if it touches the wall)
            self.goal_y = self.screen_height - 20  # the goal to reach is at the upper left of the map (y-coordinate)

            self.one_time_flag = False


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

    def set_env_data(self, env_data):
        self.env_data = env_data

    def get_sand_density(self, x, y):#, matrix=None):
        SENSOR_SENSING_PROXITY = 10
        matrix = self.env_data.sand_pixel_matrix
        x = int(x)
        y = int(y)
        # Lets select the area around the sensor x,y with a pre-defined proximity i.e numpy matrix range selection
        # on rows and cols
        sensor_data =  matrix[x - SENSOR_SENSING_PROXITY: x + SENSOR_SENSING_PROXITY,
               y - SENSOR_SENSING_PROXITY: y + SENSOR_SENSING_PROXITY]
        sensor_data = np.sum(sensor_data)
        sensor_data = int(sensor_data) / 400. #20*20 : 10+10 on x-axis and 10+10 on y-axis
        # if sensor_data  >  0:
        #     print("Sensor data: ", sensor_data)

        return int(sensor_data)

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
        self.sensor1 = Vector(self.env_data.SENSOR_PROXIMITY, 0).rotate(self.angle) + self.pos # updating the position of sensor 1
        self.sensor2 = Vector(self.env_data.SENSOR_PROXIMITY, 0).rotate((self.angle+30)%360) + self.pos # updating the position of sensor 2
        self.sensor3 = Vector(self.env_data.SENSOR_PROXIMITY, 0).rotate((self.angle-30)%360) + self.pos # updating the position of sensor 3

        # getting the signal received by sensor 1 (density of sand around sensor 1)
        self.signal1 = self.get_sand_density(self.sensor1_x, self.sensor1_y)
        # getting the signal received by sensor 2 (density of sand around sensor 2)
        self.signal2 = self.get_sand_density(self.sensor2_x, self.sensor2_y)
        # getting the signal received by sensor 3 (density of sand around sensor 3)
        self.signal3 = self.get_sand_density(self.sensor3_x, self.sensor3_y)

        if self.sensor1_x > self.env_data.screen_width-10 or \
                self.sensor1_x < 10 or \
                self.sensor1_y > self.env_data.screen_height-10 or \
                self.sensor1_y < 10: # if sensor 1 is out of the map (the car is facing one edge of the map)
            self.signal1 = 1. # sensor 1 detects full sand
        if self.sensor2_x > self.env_data.screen_width-10 or \
                self.sensor2_x < 10 or \
                self.sensor2_y > self.env_data.screen_height-10 or \
                self.sensor2_y < 10: # if sensor 2 is out of the map (the car is facing one edge of the map)
            self.signal2 = 1. # sensor 2 detects full sand
        if self.sensor3_x > self.env_data.screen_width-10 or \
                self.sensor3_x < 10 or \
                self.sensor3_y > self.env_data.screen_height-10 or \
                self.sensor3_y < 10: # if sensor 3 is out of the map (the car is facing one edge of the map)
            self.signal3 = 1. # sensor 3 detects full sand

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

    def __init__(self, env_data, **kwargs):
        super(SelfDriveEnv, self).__init__(**kwargs)
        # Add callback to get keyboard close event
        self._keyboard = Window.request_keyboard(self._keyboard_closed, self)
        # Add callback to bind the keybaord events with environment object
        self._keyboard.bind(on_key_down=self._on_keyboard_down)

        self.AUTO_MODE = True
        self.env_data: SelfDriveEnvData = env_data
        self.car.set_env_data(self.env_data)

        self.brain = DeepQLearningNetwork(5,3,0.9)

    def set_env_data(self, env_data):
        self.env_data = env_data

    def _keyboard_closed(self):
        self._keyboard.unbind(on_key_down=self._on_keyboard_down)
        self._keyboard = None

    def _on_keyboard_down(self, keyboard, keycode, text, modifiers):

        if keycode[1] == 'up':
            if not self.AUTO_MODE:
                self.car.center_y += 20
                self.sensor1.center_y += 20
                self.sensor2.center_y += 20
                self.sensor3.center_y += 20
        elif keycode[1] == 'down':
            if not self.AUTO_MODE:
                self.car.center_y -= 20
                self.sensor1.center_y -= 20
                self.sensor2.center_y -= 20
                self.sensor3.center_y -= 20
        elif keycode[1] == 'left':
            if not self.AUTO_MODE:
                self.car.center_x -= 20
                self.sensor1.center_x -= 20
                self.sensor2.center_x -= 20
                self.sensor3.center_x -= 20
        elif keycode[1] == 'right':
            if not self.AUTO_MODE:
                self.car.center_x += 20
                self.sensor1.center_x += 20
                self.sensor2.center_x += 20
                self.sensor3.center_x += 20
        elif keycode[1] == 'q':
            if not self.AUTO_MODE:
                self.update(mannual=True)
        elif keycode[1] == 'w':
            if not self.AUTO_MODE:
                self.update(mannual=True)
        elif keycode[1] == 's':
            self.AUTO_MODE = not self.AUTO_MODE
            if self.AUTO_MODE:
                print("Our AI will learn to drive while you are having the break!")
            else:
                print("Animation stopped...now you are in control. Be safe!")
        elif keycode[1] == 'escape':
            exit()
        return True

    def serve_car(self):  # starting the car when we launch the application
        self.car.center = self.center  # the car will start at the center of the map
        self.car.velocity = Vector(6, 0)  # the car will start to go horizontally to the right with a speed of 6

    def update(self, mannual=False): # the big update function that updates everything that needs to be updated at each discrete time t when reaching a new state (getting new signals from the sensors)

        self.env_data.screen_width = self.width # width of the map (horizontal edge)
        self.env_data.screen_height = self.height # height of the map (vertical edge)
        self.env_data.one_time_updates()

        if self.AUTO_MODE == True or mannual == True:

            # difference of x-coordinates between the goal and the car
            xx = self.env_data.goal_x - self.car.x
            # difference of y-coordinates between the goal and the car
            yy = self.env_data.goal_y - self.car.y
            # direction of the car with respect to the goal (if the car is heading perfectly
            # towards the goal, then orientation = 0)
            orientation = Vector(*self.car.velocity).angle((xx,yy)) / 180.
            last_signal = [self.car.signal1, self.car.signal2, self.car.signal3, orientation, -orientation]

            # our input state vector, composed of the three signals received by the three sensors, plus the orientation and -orientation
            action = self.brain.update(self.env_data.last_reward, last_signal)
            # playing the action from our ai (the object brain of the dqn class)
            self.env_data.scores.append(self.brain.score())  # appending the score (mean of the last 100 rewards to the reward window)

            # print("action -----> " + str(action))

            rotation = self.env_data.action2rotation[action]

            # print("rotation -----> " + str(rotation))
            self.car.move(rotation=rotation)

            self.sensor1.pos = self.car.sensor1 # updating the position of the first sensor (sensor1) right after the car moved
            self.sensor2.pos = self.car.sensor2 # updating the position of the second sensor (sensor2) right after the car moved
            self.sensor3.pos = self.car.sensor3 # updating the position of the third sensor (sensor3) right after the car moved

            # getting the new distance between the car and the goal right after the car moved
            distance = np.sqrt((self.car.x - self.env_data.goal_x) ** 2 + (self.car.y - self.env_data.goal_y) ** 2)  

            if self.env_data.sand_pixel_matrix[int(self.car.x), int(self.car.y)] > 0:  # if the car is on the sand
                self.car.velocity = Vector(1, 0).rotate(self.car.angle)  # it is slowed down (speed = 1)
                self.env_data.last_reward = -1  # and reward = -1
            else:  # otherwise
                self.car.velocity = Vector(6, 0).rotate(self.car.angle)  # it goes to a normal speed (speed = 6)
                self.env_data.last_reward = -0.2  # and it gets bad reward (-0.2)
                if distance < self.env_data.last_distance:  # however if it getting close to the goal
                    self.env_data.last_reward = 0.1  # it still gets slightly positive reward 0.1

            if self.car.x < 10:  # if the car is in the left edge of the frame
                self.car.x = 10  # it is not slowed down
                self.env_data.last_reward = -1  # but it gets bad reward -1
            if self.car.x > self.width - 10:  # if the car is in the right edge of the frame
                self.car.x = self.width - 10  # it is not slowed down
                self.env_data.last_reward = -1  # but it gets bad reward -1
            if self.car.y < 10:  # if the car is in the bottom edge of the frame
                self.car.y = 10  # it is not slowed down
                self.env_data.last_reward = -1  # but it gets bad reward -1
            if self.car.y > self.height - 10:  # if the car is in the upper edge of the frame
                self.car.y = self.height - 10  # it is not slowed down
                self.env_data.last_reward = -1  # but it gets bad reward -1

            if distance < 100:  # when the car reaches its goal
                self.env_data.goal_x = self.width - self.env_data.goal_x  # the goal becomes the bottom right corner of the map (the downtown), and vice versa (updating of the x-coordinate of the goal)
                self.env_data.goal_y = self.height - self.env_data.goal_y  # the goal becomes the bottom right corner of the map (the downtown), and vice versa (updating of the y-coordinate of the goal)

            # Updating the last distance from the car to the goal
            self.env_data.last_distance = distance


# Painting for graphic interface (see kivy tutorials: https://kivy.org/docs/tutorials/firstwidget.html)

class SandWidget(Widget):
    # Adding this line if we don't want the right click to put a red point
    Config.set('input', 'mouse', 'mouse,multitouch_on_demand')

    def set_env_data(self, env_data):
        self.env_data = env_data

    def on_touch_down(self, touch): # putting some sand when we do a left click
        with self.canvas:
            Color(0.8,0.7,0)
            touch.ud['line'] = Line(points = (touch.x, touch.y), width = 10)
            self.env_data.last_mouse_pos_x = int(touch.x)
            self.env_data.last_mouse_pos_y = int(touch.y)
            self.env_data.n_points = 0
            self.env_data.length = 0
            self.env_data.sand_pixel_matrix[int(touch.x),int(touch.y)] = 1

    def on_touch_move(self, touch): # putting some sand when we move the mouse while pressing left
        # global length,n_points,last_x,last_y
        if touch.button=='left':
            touch.ud['line'].points += [touch.x, touch.y]
            x = int(touch.x)
            y = int(touch.y)
            self.env_data.length += np.sqrt(max((x - self.env_data.last_mouse_pos_x)**2 +
                                                (y - self.env_data.last_mouse_pos_y)**2, 2))
            self.env_data.n_points += 1.
            density = self.env_data.n_points/(self.env_data.length)
            touch.ud['line'].width = int(20*density + 1)
            self.env_data.sand_pixel_matrix[int(touch.x) - 10 : int(touch.x) + 10, int(touch.y) - 10 : int(touch.y) + 10] = 1
            self.env_data.last_mouse_pos_x = x
            self.env_data.last_mouse_pos_y = y

class SelfDriveApp(App):

    def build(self):
        self.title = 'Self Driving Car'
        self.env_data = SelfDriveEnvData()

        parent = SelfDriveEnv(env_data=self.env_data)

        self.painter = SandWidget()
        self.painter.set_env_data(env_data=self.env_data)

        parent.serve_car()
        Clock.schedule_interval(parent.update, 1.0 / 60.0)

        clearbtn = Button(text='clear')
        savebtn = Button(text='save/plot', pos=(parent.width, 0))
        loadbtn = Button(text='load', pos=(2 * parent.width, 0))
        clearbtn.bind(on_release=self.clear_canvas)
        savebtn.bind(on_release=self.save)
        # loadbtn.bind(on_release=self.load)
        parent.add_widget(self.painter)
        parent.add_widget(clearbtn)
        parent.add_widget(savebtn)
        parent.add_widget(loadbtn)

        return  parent

    def clear_canvas(self, obj): # clear button
        self.painter.canvas.clear()
        self.env_data.sand_pixel_matrix = np.zeros((self.env_data.screen_width,self.env_data.screen_height))

    def save(self, obj): # save button
        plt.plot(self.env_data.scores)
        plt.show()
        plt.pause(0.001)


if __name__ == '__main__':
    SelfDriveApp().run()
