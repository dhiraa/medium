# Importing the Kivy packages
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.properties import NumericProperty, ReferenceListProperty, ObjectProperty
from kivy.vector import Vector
from kivy.core.window import Window



class Car(Widget):
    # initializing the angle of the car (angle between the x-axis of the map and the axis of the car)
    angle = NumericProperty(0)
    # initializing the last rotation of the car
    # (after playing the action, the car does a rotation of 0, 20 or -20 degrees)
    rotation = NumericProperty(0)

    sensor2_x = NumericProperty(0)  # initializing the x-coordinate of the second sensor (the one that looks 30 degrees to the left)
    sensor2_y = NumericProperty(0)  # initializing the y-coordinate of the second sensor (the one that looks 30 degrees to the left)
    sensor2_pos = ReferenceListProperty(sensor2_x, sensor2_y)  # second sensor vector

    sensor3_x = NumericProperty(0)  # initializing the x-coordinate of the third sensor (the one that looks 30 degrees to the right)
    sensor3_y = NumericProperty(0)  # initializing the y-coordinate of the third sensor (the one that looks 30 degrees to the right)
    sensor3_pos = ReferenceListProperty(sensor3_x, sensor3_y)  # third sensor vector


    def rotate(self, rotation):
        print("Car: ")
        print(self.angle)
        print(self.pos)
        self.rotation = rotation # getting the rotation of the car
        self.angle = self.angle + self.rotation # updating the angle



class Sensor1(Widget):

    angle = NumericProperty(0)  # initializing the angle of the car (angle between the x-axis of the map and the axis of the car)
    rotation = NumericProperty(0)  # initializing the last rotation of the car (after playing the action, the car does a rotation of 0, 20 or -20 degrees)

    sensor1_x = NumericProperty(0)  # initializing the x-coordinate of the first sensor (the one that looks forward)
    sensor1_y = NumericProperty(0)  # initializing the y-coordinate of the first sensor (the one that looks forward)
    sensor1 = ReferenceListProperty(sensor1_x, sensor1_y)  # first sensor vector


    # TODO: this doesn't work? reason being that the sensor is not in sync with the car/rectangle
    # TODO: it moves/rotates with respect to the environment, where it is placed
    def move(self, rotation):
        self.rotation = rotation  # getting the rotation of the car
        self.angle = self.angle + self.rotation  # updating the angle

        print("sensor1")
        print(self.pos)
        self.pos = Vector(*self.pos).rotate(self.angle)  # updating the position of sensor 1
        print(self.pos)
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

    def _keyboard_closed(self):
        self._keyboard.unbind(on_key_down=self._on_keyboard_down)
        self._keyboard = None

    def _on_keyboard_down(self, keyboard, keycode, text, modifiers):

        if keycode[1] == 'up':
            self.car.center_y += 5
            self.sensor1.center_y += 5
            self.sensor2.center_y += 5
            self.sensor3.center_y += 5
        elif keycode[1] == 'down':
            self.car.center_y -= 5
            self.sensor1.center_y -= 5
            self.sensor2.center_y -= 5
            self.sensor3.center_y -= 5
        elif keycode[1] == 'left':
            self.car.center_x -= 5
            self.sensor1.center_x -= 5
            self.sensor2.center_x -= 5
            self.sensor3.center_x -= 5
        elif keycode[1] == 'right':
            self.car.center_x += 5
            self.sensor1.center_x += 5
            self.sensor2.center_x += 5
            self.sensor3.center_x += 5
        elif keycode[1] == 'q':
            self.car.rotate(5)
            self.sensor1.move(5)
        elif keycode[1] == 'w':
            self.car.rotate(-5)
            self.sensor1.move(-5)
        elif keycode[1] == 'escape':
            exit()
        return True

class SelfDriveApp(App):

    def build(self):
        self.title = 'Self Driving Car'
        parent = SelfDriveEnv()
        return  parent



if __name__ == '__main__':
    SelfDriveApp().run()
