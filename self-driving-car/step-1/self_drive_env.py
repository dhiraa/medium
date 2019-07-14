# Importing the Kivy packages
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.properties import ObjectProperty


class Car(Widget):
    pass

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

class SelfDriveApp(App):

    def build(self):
        self.title = 'Self Driving Car'
        parent = SelfDriveEnv()
        return  parent

if __name__ == '__main__':
    SelfDriveApp().run()
