# Self Driving Car

# Importing the libraries
import numpy as np
from random import random, randint
import matplotlib.pyplot as plt
import time

# Importing the Kivy packages
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.graphics import Color, Ellipse, Line, Rectangle
from kivy.config import Config
from kivy.properties import NumericProperty, ReferenceListProperty, ObjectProperty
from kivy.vector import Vector
from kivy.clock import Clock
from datetime import datetime
from datetime import timedelta
from kivy.uix.label import Label

# Importing the Dqn object from our AI in ia.py
from rohaan_ai import Dqn

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# Adding this line if we don't want the right click to put a red point
Config.set('input', 'mouse', 'mouse, multitouch_on_demand')
Config.set('graphics', 'width', '1000')
Config.set('graphics', 'height', '800')

# Performance Evaluation


# Introducing last_x and last_y, used to keep the last point in memory when we draw the sand on the map
last_x = 0
last_y = 0
n_points = 0 # the total number of points in the last drawing
length = 0 # the length of the last drawing

# Getting our AI, which we call "brain", and that contains our neural network that represents our Q-function
brain = Dqn(5,3,0.9) # 5 sensors, 3 actions, gama = 0.9
brain2 = Dqn(5,3,0.9) # 5 sensors, 3 actions, gama = 0.9
brain3 = Dqn(5,3,0.9) # 5 sensors, 3 actions, gama = 0.9
action2rotation = [0,20,-20] # action = 0 => no rotation, action = 1 => rotate 20 degres, action = 2 => rotate -20 degres
last_reward = 0 # initializing the last reward
last_reward2 = 0 # initializing the last reward
last_reward3 = 0 # initializing the last reward
scores = [] # initializing the mean score curve (sliding window of the rewards) with respect to time
scores2 = []
scores3 = []

# Rewards
LIVING_PENALTY = -0.2 # Penalty for not achieving the goal
GETTING_CLOSER_BONUS = 0.1 # Reward for getting closer to goal
COLLISION_PENALTY = 15 # Penalty for colling with the other car
SAND_PENALTY = -1.5 # Penalty for going over sand-trap
PROXIMITY_TO_GOAL = 50 # How close to the goal is good enough


# Initializing the map
first_update = True # using this trick to initialize the map only once
def init():
    global sand # sand is an array that has as many cells as our graphic interface has pixels. Each cell has a one if there is sand, 0 otherwise.
    global goal_x # x-coordinate of the goal (where the car has to go, that is the airport or the downtown)
    global goal_y # y-coordinate of the goal (where the car has to go, that is the airport or the downtown)
    global goal_x2 # x-coordinate of the goal (where the car has to go, that is the airport or the downtown)
    global goal_y2 # y-coordinate of the goal (where the car has to go, that is the airport or the downtown)
    global goal_x3 # x-coordinate of the goal (where the car has to go, that is the airport or the downtown)
    global goal_y3 # y-coordinate of the goal (where the car has to go, that is the airport or the downtown)
    global first_update
    sand = np.zeros((longueur,largeur)) # initializing the sand array with only zeros
    # goal_x = 20 # the goal to reach is at the upper left of the map (the x-coordinate is 20 and not 0 because the car gets bad reward if it touches the wall)
    # goal_y = largeur - 20 # the goal to reach is at the upper left of the map (y-coordinate)
    # goal_x2 = 40 # the goal to reach is at the upper left of the map (the x-coordinate is 20 and not 0 because the car gets bad reward if it touches the wall)
    # goal_y2 = largeur - 40 # the goal to reach is at the upper left of the map (y-coordinate)
    first_update = False # trick to initialize the map only once

# Initializing the last distance
last_distance = 0
last_distance2 = 0
last_distance3 = 0

# Creating the car class (to understand "NumericProperty" and "ReferenceListProperty", see kivy tutorials: https://kivy.org/docs/tutorials/pong.html)

class Car(Widget):

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
        self.pos = Vector(*self.velocity) + self.pos # updating the position of the car according to its last position and velocity
        self.rotation = rotation # getting the rotation of the car
        self.angle = self.angle + self.rotation # updating the angle
        self.sensor1 = Vector(30, 0).rotate(self.angle) + self.pos # updating the position of sensor 1
        self.sensor2 = Vector(30, 0).rotate((self.angle+30)%360) + self.pos # updating the position of sensor 2
        self.sensor3 = Vector(30, 0).rotate((self.angle-30)%360) + self.pos # updating the position of sensor 3
        self.signal1 = int(np.sum(sand[int(self.sensor1_x)-10:int(self.sensor1_x)+10, int(self.sensor1_y)-10:int(self.sensor1_y)+10]))/400. # getting the signal received by sensor 1 (density of sand around sensor 1)
        self.signal2 = int(np.sum(sand[int(self.sensor2_x)-10:int(self.sensor2_x)+10, int(self.sensor2_y)-10:int(self.sensor2_y)+10]))/400. # getting the signal received by sensor 2 (density of sand around sensor 2)
        self.signal3 = int(np.sum(sand[int(self.sensor3_x)-10:int(self.sensor3_x)+10, int(self.sensor3_y)-10:int(self.sensor3_y)+10]))/400. # getting the signal received by sensor 3 (density of sand around sensor 3)
        if self.sensor1_x > longueur-10 or self.sensor1_x<10 or self.sensor1_y>largeur-10 or self.sensor1_y<10: # if sensor 1 is out of the map (the car is facing one edge of the map)
            self.signal1 = 1. # sensor 1 detects full sand
        if self.sensor2_x > longueur-10 or self.sensor2_x<10 or self.sensor2_y>largeur-10 or self.sensor2_y<10: # if sensor 2 is out of the map (the car is facing one edge of the map)
            self.signal2 = 1. # sensor 2 detects full sand
        if self.sensor3_x > longueur-10 or self.sensor3_x<10 or self.sensor3_y>largeur-10 or self.sensor3_y<10: # if sensor 3 is out of the map (the car is facing one edge of the map)
            self.signal3 = 1. # sensor 3 detects full sand

class Ball1(Widget): # sensor 1 (see kivy tutorials: kivy https://kivy.org/docs/tutorials/pong.html)
    pass
class Ball2(Widget): # sensor 2 (see kivy tutorials: kivy https://kivy.org/docs/tutorials/pong.html)
    pass
class Ball3(Widget): # sensor 3 (see kivy tutorials: kivy https://kivy.org/docs/tutorials/pong.html)
    pass

class Car2(Widget):

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
        self.pos = Vector(*self.velocity) + self.pos # updating the position of the car according to its last position and velocity
        self.rotation = rotation # getting the rotation of the car
        self.angle = self.angle + self.rotation # updating the angle
        self.sensor1 = Vector(30, 0).rotate(self.angle) + self.pos # updating the position of sensor 1
        self.sensor2 = Vector(30, 0).rotate((self.angle+30)%360) + self.pos # updating the position of sensor 2
        self.sensor3 = Vector(30, 0).rotate((self.angle-30)%360) + self.pos # updating the position of sensor 3
        self.signal1 = int(np.sum(sand[int(self.sensor1_x)-10:int(self.sensor1_x)+10, int(self.sensor1_y)-10:int(self.sensor1_y)+10]))/400. # getting the signal received by sensor 1 (density of sand around sensor 1)
        self.signal2 = int(np.sum(sand[int(self.sensor2_x)-10:int(self.sensor2_x)+10, int(self.sensor2_y)-10:int(self.sensor2_y)+10]))/400. # getting the signal received by sensor 2 (density of sand around sensor 2)
        self.signal3 = int(np.sum(sand[int(self.sensor3_x)-10:int(self.sensor3_x)+10, int(self.sensor3_y)-10:int(self.sensor3_y)+10]))/400. # getting the signal received by sensor 3 (density of sand around sensor 3)
        if self.sensor1_x > longueur-10 or self.sensor1_x<10 or self.sensor1_y>largeur-10 or self.sensor1_y<10: # if sensor 1 is out of the map (the car is facing one edge of the map)
            self.signal1 = 1. # sensor 1 detects full sand
        if self.sensor2_x > longueur-10 or self.sensor2_x<10 or self.sensor2_y>largeur-10 or self.sensor2_y<10: # if sensor 2 is out of the map (the car is facing one edge of the map)
            self.signal2 = 1. # sensor 2 detects full sand
        if self.sensor3_x > longueur-10 or self.sensor3_x<10 or self.sensor3_y>largeur-10 or self.sensor3_y<10: # if sensor 3 is out of the map (the car is facing one edge of the map)
            self.signal3 = 1. # sensor 3 detects full sand
# Creating the game class (to understand "ObjectProperty", see kivy tutorials: kivy https://kivy.org/docs/tutorials/pong.html)

class Ball21(Widget): # sensor 1 (see kivy tutorials: kivy https://kivy.org/docs/tutorials/pong.html)
    pass
class Ball22(Widget): # sensor 2 (see kivy tutorials: kivy https://kivy.org/docs/tutorials/pong.html)
    pass
class Ball23(Widget): # sensor 3 (see kivy tutorials: kivy https://kivy.org/docs/tutorials/pong.html)
    pass

class Car3(Widget):

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
        self.pos = Vector(*self.velocity) + self.pos # updating the position of the car according to its last position and velocity
        self.rotation = rotation # getting the rotation of the car
        self.angle = self.angle + self.rotation # updating the angle
        self.sensor1 = Vector(30, 0).rotate(self.angle) + self.pos # updating the position of sensor 1
        self.sensor2 = Vector(30, 0).rotate((self.angle+30)%360) + self.pos # updating the position of sensor 2
        self.sensor3 = Vector(30, 0).rotate((self.angle-30)%360) + self.pos # updating the position of sensor 3
        self.signal1 = int(np.sum(sand[int(self.sensor1_x)-10:int(self.sensor1_x)+10, int(self.sensor1_y)-10:int(self.sensor1_y)+10]))/400. # getting the signal received by sensor 1 (density of sand around sensor 1)
        self.signal2 = int(np.sum(sand[int(self.sensor2_x)-10:int(self.sensor2_x)+10, int(self.sensor2_y)-10:int(self.sensor2_y)+10]))/400. # getting the signal received by sensor 2 (density of sand around sensor 2)
        self.signal3 = int(np.sum(sand[int(self.sensor3_x)-10:int(self.sensor3_x)+10, int(self.sensor3_y)-10:int(self.sensor3_y)+10]))/400. # getting the signal received by sensor 3 (density of sand around sensor 3)
        if self.sensor1_x > longueur-10 or self.sensor1_x<10 or self.sensor1_y>largeur-10 or self.sensor1_y<10: # if sensor 1 is out of the map (the car is facing one edge of the map)
            self.signal1 = 1. # sensor 1 detects full sand
        if self.sensor2_x > longueur-10 or self.sensor2_x<10 or self.sensor2_y>largeur-10 or self.sensor2_y<10: # if sensor 2 is out of the map (the car is facing one edge of the map)
            self.signal2 = 1. # sensor 2 detects full sand
        if self.sensor3_x > longueur-10 or self.sensor3_x<10 or self.sensor3_y>largeur-10 or self.sensor3_y<10: # if sensor 3 is out of the map (the car is facing one edge of the map)
            self.signal3 = 1. # sensor 3 detects full sand
# Creating the game class (to understand "ObjectProperty", see kivy tutorials: kivy https://kivy.org/docs/tutorials/pong.html)

class Ball31(Widget): # sensor 1 (see kivy tutorials: kivy https://kivy.org/docs/tutorials/pong.html)
    pass
class Ball32(Widget): # sensor 2 (see kivy tutorials: kivy https://kivy.org/docs/tutorials/pong.html)
    pass
class Ball33(Widget): # sensor 3 (see kivy tutorials: kivy https://kivy.org/docs/tutorials/pong.html)
    pass

class Game(Widget):
    
    car = ObjectProperty(None) # getting the car object from our kivy file
    ball1 = ObjectProperty(None) # getting the sensor 1 object from our kivy file
    ball2 = ObjectProperty(None) # getting the sensor 2 object from our kivy file
    ball3 = ObjectProperty(None) # getting the sensor 3 object from our kivy file
    car2 = ObjectProperty(None) # getting the car object from our kivy file
    ball21 = ObjectProperty(None) # getting the sensor 1 object from our kivy file
    ball22 = ObjectProperty(None) # getting the sensor 2 object from our kivy file
    ball23 = ObjectProperty(None) # getting the sensor 3 object from our kivy file
    car3 = ObjectProperty(None) # getting the car object from our kivy file
    ball31 = ObjectProperty(None) # getting the sensor 1 object from our kivy file
    ball32 = ObjectProperty(None) # getting the sensor 2 object from our kivy file
    ball33 = ObjectProperty(None) # getting the sensor 3 object from our kivy file
    
    def serve_car(self): # starting the car when we launch the application
        self.car.center = self.center # the car will start at the center of the map
        self.car.velocity = Vector(6, 0) # the car will start to go horizontally to the right with a speed of 6
        self.car.center[1] = self.car.center[1] - 20
        self.car2.center = self.center # the car will start at the center of the map
        self.car2.center[1] = self.car2.center[1] + 20
        self.car2.velocity = Vector(-6, 0) # the car will start to go horizontally to the left with a speed of 6
        self.car3.center = self.center # the car will start at the center of the map
        self.car3.center[1] = self.car3.center[1]
        self.car3.velocity = Vector(0, 6) # the car will start to go horizontally to the left with a speed of 6

    def update(self, dt): # the big update function that updates everything that needs to be updated at each discrete time t when reaching a new state (getting new signals from the sensors)
        
        # Shared Globals
        global timestep
        global longueur # specifying the global variables (width of the map)
        global largeur # specifying the global variables (height of the map)
        
        # Car 1 Globals
        global brain # specifying the global variables (the brain of the car, that is our AI)
        global last_reward # specifying the global variables (the last reward)
        global scores # specifying the global variables (the means of the rewards)
        global last_distance # specifying the global variables (the last distance from the car to the goal)
        global goal_x # specifying the global variables (x-coordinate of the goal)
        global goal_y # specifying the global variables (y-coordinate of the goal)
        global last_goal
        global duration
        
        # Car 2 Globals
        global brain2 # specifying the global variables (the brain of the car2, that is our AI)
        global last_reward2 # specifying the global variables (the last reward)
        global scores2 # specifying the global variables (the means of the rewards)
        global last_distance2 # specifying the global variables (the last distance from the car to the goal)
        global goal_x2 # specifying the global variables (x-coordinate of the goal)
        global goal_y2 # specifying the global variables (y-coordinate of the goal)
        global last_goal2
        global duration2
        
        # Car 2 Globals
        global brain3 # specifying the global variables (the brain of the car2, that is our AI)
        global last_reward3 # specifying the global variables (the last reward)
        global scores3 # specifying the global variables (the means of the rewards)
        global last_distance3 # specifying the global variables (the last distance from the car to the goal)
        global goal_x3 # specifying the global variables (x-coordinate of the goal)
        global goal_y3 # specifying the global variables (y-coordinate of the goal)
        global last_goal3
        global duration3
        
        global total_goals_achieved
        global total_collisions
        global total_collisions_prev
        global score_t
        global collisions_t
        
        longueur = self.width # width of the map (horizontal edge)
        largeur = self.height # height of the map (vertical edge)
        
        goal_1_reached = False
        goal_2_reached = False
        goal_3_reached = False
        
        if first_update: # trick to initialize the map only once
            goal_x = randint(10,self.width - 10)
            goal_y = randint(10,self.height - 10)
            goal_x2 = randint(10,self.width - 10)
            goal_y2 = randint(10,self.height - 10)
            goal_x3 = randint(10,self.width - 10)
            goal_y3 = randint(10,self.height - 10)
            
            total_goals_achieved = 0
            total_collisions = 0
            total_collisions_prev = 0
            init()

        # Car 1 Update
        
        with self.canvas:
                Color(0,0,1, mode="rgb")
                self.rect = Ellipse(pos=(goal_x,goal_y), size=(5,5))
                
        xx = goal_x - self.car.x # difference of x-coordinates between the goal and the car
        yy = goal_y - self.car.y # difference of y-coordinates between the goal and the car
        orientation = Vector(*self.car.velocity).angle((xx,yy))/180. # direction of the car with respect to the goal (if the car is heading perfectly towards the goal, then orientation = 0)
        last_signal = [self.car.signal1, self.car.signal2, self.car.signal3, orientation, -orientation] # our input state vector, composed of the three signals received by the three sensors, plus the orientation and -orientation
        action = brain.update(last_reward, last_signal) # playing the action from our ai (the object brain of the dqn class)
        timestep += 1
        scores.append(brain.score()) # appending the score (mean of the last 100 rewards to the reward window)
        rotation = action2rotation[action] # converting the action played (0, 1 or 2) into the rotation angle (0°, 20° or -20°)
        self.car.move(rotation) # moving the car according to this last rotation angle
        distance = np.sqrt((self.car.x - goal_x)**2 + (self.car.y - goal_y)**2) # getting the new distance between the car and the goal right after the car moved
        self.ball1.pos = self.car.sensor1 # updating the position of the first sensor (ball1) right after the car moved
        self.ball2.pos = self.car.sensor2 # updating the position of the second sensor (ball2) right after the car moved
        self.ball3.pos = self.car.sensor3 # updating the position of the third sensor (ball3) right after the car moved

        if sand[int(self.car.x),int(self.car.y)] > 0: # if the car is on the sand
            self.car.velocity = Vector(1, 0).rotate(self.car.angle) # it is slowed down (speed = 1)
            last_reward = SAND_PENALTY # and reward = -1
        else: # otherwise
            self.car.velocity = Vector(6, 0).rotate(self.car.angle) # it goes to a normal speed (speed = 6)
            last_reward = LIVING_PENALTY # and it gets bad reward (-0.2)
            if distance < last_distance: # however if it getting close to the goal
                last_reward += GETTING_CLOSER_BONUS # it still gets slightly positive reward 0.1

        if self.car.x < 10: # if the car is in the left edge of the frame
            self.car.x = 10 # it is not slowed down
            last_reward = -1 # but it gets bad reward -1
        if self.car.x > self.width-10: # if the car is in the right edge of the frame
            self.car.x = self.width-10 # it is not slowed down
            last_reward = -1 # but it gets bad reward -1
        if self.car.y < 10: # if the car is in the bottom edge of the frame
            self.car.y = 10 # it is not slowed down
            last_reward = -1 # but it gets bad reward -1
        if self.car.y > self.height-10: # if the car is in the upper edge of the frame
            self.car.y = self.height-10 # it is not slowed down
            last_reward = -1 # but it gets bad reward -1

        if distance < PROXIMITY_TO_GOAL: # when the car reaches its goal
            
            with self.canvas:
                Color(0,0,0, mode="rgb")
                self.rect = Ellipse(pos=(goal_x,goal_y), size=(5,5))
            goal_x = randint(10,self.width - 10)
            goal_y = randint(10,self.height - 10)
            print('Car 1 Reached its goal at Timestep: ' + str(timestep) + ' after: ' + str(timestep-last_goal))
            print('Recent Score: ' + str(scores[-1]))
            goal_1_reached = True
            if last_goal != 0:
                duration.append(timestep-last_goal)
            if len(duration) > 20:
                del duration[0]
            last_goal = timestep
            last_reward = 1

        # Updating the last distance from the car to the goal
        last_distance = distance
        
        # Car 2 Update
        
        with self.canvas:
                Color(1,0,0, mode="rgb")
                self.rect = Ellipse(pos=(goal_x2,goal_y2), size=(5,5))
                
        xx2 = goal_x2 - self.car2.x # difference of x-coordinates between the goal and the car
        yy2 = goal_y2 - self.car2.y # difference of y-coordinates between the goal and the car
        orientation2 = Vector(*self.car2.velocity).angle((xx2,yy2))/180. # direction of the car with respect to the goal (if the car is heading perfectly towards the goal, then orientation = 0)
        last_signal2 = [self.car2.signal1, self.car2.signal2, self.car2.signal3, orientation2, -orientation2] # our input state vector, composed of the three signals received by the three sensors, plus the orientation and -orientation
        action2 = brain2.update(last_reward2, last_signal2) # playing the action from our ai (the object brain of the dqn class)
        scores2.append(brain2.score()) # appending the score (mean of the last 100 rewards to the reward window)
        rotation2 = action2rotation[action2] # converting the action played (0, 1 or 2) into the rotation angle (0°, 20° or -20°)
        self.car2.move(rotation2) # moving the car according to this last rotation angle
        distance2 = np.sqrt((self.car2.x - goal_x2)**2 + (self.car2.y - goal_y2)**2) # getting the new distance between the car and the goal right after the car moved
        self.ball21.pos = self.car2.sensor1 # updating the position of the first sensor (ball1) right after the car moved
        self.ball22.pos = self.car2.sensor2 # updating the position of the second sensor (ball2) right after the car moved
        self.ball23.pos = self.car2.sensor3 # updating the position of the third sensor (ball3) right after the car moved

        if sand[int(self.car2.x),int(self.car2.y)] > 0: # if the car is on the sand
            self.car2.velocity = Vector(1, 0).rotate(self.car2.angle) # it is slowed down (speed = 1)
            last_reward2 = SAND_PENALTY # and reward = -1
        else: # otherwise
            self.car2.velocity = Vector(6, 0).rotate(self.car2.angle) # it goes to a normal speed (speed = 6)
            last_reward2 = LIVING_PENALTY # and it gets bad reward (-0.2)
            if distance2 < last_distance2: # however if it getting close to the goal
                last_reward2 += GETTING_CLOSER_BONUS # it still gets slightly positive reward 0.1

        if self.car2.x < 10: # if the car is in the left edge of the frame
            self.car2.x = 10 # it is not slowed down
            last_reward2 = -1 # but it gets bad reward -1
        if self.car2.x > self.width-10: # if the car is in the right edge of the frame
            self.car2.x = self.width-10 # it is not slowed down
            last_reward2 = -1 # but it gets bad reward -1
        if self.car2.y < 10: # if the car is in the bottom edge of the frame
            self.car2.y = 10 # it is not slowed down
            last_reward2 = -1 # but it gets bad reward -1
        if self.car2.y > self.height-10: # if the car is in the upper edge of the frame
            self.car2.y = self.height-10 # it is not slowed down
            last_reward2 = -1 # but it gets bad reward -1

        if distance2 < PROXIMITY_TO_GOAL: # when the car2 reaches its goal
            with self.canvas:
                Color(0,0,0, mode="rgb")
                self.rect = Ellipse(pos=(goal_x2,goal_y2), size=(5,5))
                
            # goal_x2 = self.width - goal_x2 # the goal becomes the bottom right corner of the map (the downtown), and vice versa (updating of the x-coordinate of the goal)
            # goal_y2 = self.height - goal_y2 # the goal becomes the bottom right corner of the map (the downtown), and vice versa (updating of the y-coordinate of the goal)
            goal_x2 = randint(10,self.width - 10)
            goal_y2 = randint(10,self.height - 10)
            # with self.canvas:
            #     Color(0,0,1, mode="rgb")
            #     self.rect = Ellipse(pos=(goal_x2,goal_y2), size=(5,5))
            print('Car 2 Reached its goal at Timestep' + str(timestep) + ' after: ' + str(timestep-last_goal2))
            print('Recent Score: ' + str(scores2[-1]))
            goal_1_reached = True
            if last_goal2 != 0:
                duration2.append(timestep-last_goal2)
            if len(duration2) > 20:
                del duration2[0]
            last_goal2 = timestep
            last_reward2 = 1

        # Updating the last distance from the car to the goal
        last_distance2 = distance2
        
        # Car 3 Update
        
        with self.canvas:
                Color(0,1,0, mode="rgb")
                self.rect = Ellipse(pos=(goal_x3,goal_y3), size=(5,5))
                
        xx3 = goal_x3 - self.car3.x # difference of x-coordinates between the goal and the car
        yy3 = goal_y3 - self.car3.y # difference of y-coordinates between the goal and the car
        orientation3 = Vector(*self.car3.velocity).angle((xx3,yy3))/180. # direction of the car with respect to the goal (if the car is heading perfectly towards the goal, then orientation = 0)
        last_signal3 = [self.car3.signal1, self.car3.signal2, self.car3.signal3, orientation3, -orientation3] # our input state vector, composed of the three signals received by the three sensors, plus the orientation and -orientation
        action3 = brain3.update(last_reward3, last_signal3) # playing the action from our ai (the object brain of the dqn class)
        scores3.append(brain3.score()) # appending the score (mean of the last 100 rewards to the reward window)
        rotation3 = action2rotation[action3] # converting the action played (0, 1 or 2) into the rotation angle (0°, 20° or -20°)
        self.car3.move(rotation3) # moving the car according to this last rotation angle
        distance3 = np.sqrt((self.car3.x - goal_x3)**2 + (self.car3.y - goal_y3)**2) # getting the new distance between the car and the goal right after the car moved
        self.ball31.pos = self.car3.sensor1 # updating the position of the first sensor (ball1) right after the car moved
        self.ball32.pos = self.car3.sensor2 # updating the position of the second sensor (ball2) right after the car moved
        self.ball33.pos = self.car3.sensor3 # updating the position of the third sensor (ball3) right after the car moved

        if sand[int(self.car3.x),int(self.car3.y)] > 0: # if the car is on the sand
            self.car3.velocity = Vector(1, 0).rotate(self.car3.angle) # it is slowed down (speed = 1)
            last_reward3 = SAND_PENALTY # and reward = -1
        else: # otherwise
            self.car3.velocity = Vector(6, 0).rotate(self.car3.angle) # it goes to a normal speed (speed = 6)
            last_reward3 = LIVING_PENALTY # and it gets bad reward (-0.2)
            if distance3 < last_distance3: # however if it getting close to the goal
                last_reward3 += GETTING_CLOSER_BONUS # it still gets slightly positive reward 0.1

        if self.car3.x < 10: # if the car is in the left edge of the frame
            self.car3.x = 10 # it is not slowed down
            last_reward3 = -1 # but it gets bad reward -1
        if self.car3.x > self.width-10: # if the car is in the right edge of the frame
            self.car3.x = self.width-10 # it is not slowed down
            last_reward3 = -1 # but it gets bad reward -1
        if self.car3.y < 10: # if the car is in the bottom edge of the frame
            self.car3.y = 10 # it is not slowed down
            last_reward3 = -1 # but it gets bad reward -1
        if self.car3.y > self.height-10: # if the car is in the upper edge of the frame
            self.car3.y = self.height-10 # it is not slowed down
            last_reward3 = -1 # but it gets bad reward -1

        if distance3 < PROXIMITY_TO_GOAL: # when the car2 reaches its goal
            with self.canvas:
                Color(0,0,0, mode="rgb")
                self.rect = Ellipse(pos=(goal_x3,goal_y3), size=(5,5))
                
            # goal_x2 = self.width - goal_x2 # the goal becomes the bottom right corner of the map (the downtown), and vice versa (updating of the x-coordinate of the goal)
            # goal_y2 = self.height - goal_y2 # the goal becomes the bottom right corner of the map (the downtown), and vice versa (updating of the y-coordinate of the goal)
            goal_x3 = randint(10,self.width - 10)
            goal_y3 = randint(10,self.height - 10)
            # with self.canvas:
            #     Color(0,0,1, mode="rgb")
            #     self.rect = Ellipse(pos=(goal_x2,goal_y2), size=(5,5))
            print('Car 3 Reached its goal at Timestep' + str(timestep) + ' after: ' + str(timestep-last_goal3))
            print('Recent Score: ' + str(scores3[-1]))
            goal_1_reached = True
            if last_goal3 != 0:
                duration3.append(timestep-last_goal3)
            if len(duration3) > 20:
                del duration3[0]
            last_goal3 = timestep
            last_reward3 = 1

        # Updating the last distance from the car to the goal
        last_distance3 = distance3
        
        # Car collision
        
        if abs(self.car.x - self.car2.x) < 10 and abs(self.car.y - self.car2.y) < 10: # if the cars collide
            self.car.x = 1 # slow down
            self.car2.x = 1 
            print('COLLISION!')
            last_reward = COLLISION_PENALTY # large penalty
            last_reward2 = COLLISION_PENALTY 
            total_collisions += 1
        if abs(self.car2.x - self.car3.x) < 10 and abs(self.car2.y - self.car3.y) < 10: # if the cars collide
            self.car2.x = 1 # slow down
            self.car3.x = 1 
            print('COLLISION!')
            last_reward2 = COLLISION_PENALTY # large penalty
            last_reward3 = COLLISION_PENALTY 
            total_collisions += 1
        if abs(self.car.x - self.car3.x) < 10 and abs(self.car.y - self.car3.y) < 10: # if the cars collide
            self.car.x = 1 # slow down
            self.car3.x = 1 
            print('COLLISION!')
            last_reward = COLLISION_PENALTY # large penalty
            last_reward3 = COLLISION_PENALTY 
            total_collisions += 1
        
        if goal_1_reached == True or goal_2_reached == True or goal_3_reached == True:
            goal_1_reached = False
            goal_2_reached = False
            goal_3_reached = False
            total_goals_achieved += 1
            
        self.display_text.text = 'Timestep: ' + str(timestep) + '\n' + \
            'Mean Score (last 100 timesteps):\n' +\
            'Rover 1: ' + str(round(brain.score(),2)) + '\n' + \
            'Rover 2: ' + str(round(brain2.score(),2)) + '\n' + \
            'Rover 3: ' + str(round(brain3.score(),2)) + '\n' + \
            'Total Goals Achieved: ' + str(total_goals_achieved)
        
        if (timestep % 10) == 0:
            score_t.append(total_goals_achieved/timestep)
            collisions_t.append(total_collisions/timestep)
            # collisions_t.append( (total_collisions - total_collisions_prev) / 10)
            # total_collisions_prev = total_collisions
            if (timestep % 100) == 0:
                plt.figure(11)
                plt.plot(score_t)
                plt.ylabel('Mean Goals Achieved')
                plt.xlabel('Timesteps')
                plt.savefig('plots/scores_plot.png')
                plt.close()
                
                plt.figure(22)
                plt.plot(collisions_t)
                plt.ylabel('Mean Collisions')
                plt.xlabel('Timesteps')
                plt.savefig('plots/collisions_plot.png')
                plt.close()
        
# Painting for graphic interface (see kivy tutorials: https://kivy.org/docs/tutorials/firstwidget.html)

class MyPaintWidget(Widget):
            
    # def __init__(self, **kwargs):
    #     super(MyPaintWidget, self).__init__(**kwargs)
    #     self.now = datetime.now()
    #     self.my_label = Label(text= self.now.strftime('%H:%M:%S'))
    #     self.add_widget(self.my_label)
        
    def on_touch_down(self, touch): # putting some sand when we do a left click
        global length,n_points,last_x,last_y
        with self.canvas:
            if touch.button=='left':
                Color(0.8,0.7,0)
                d=10.
                touch.ud['line'] = Line(points = (touch.x, touch.y), width = 10)
                last_x = int(touch.x)
                last_y = int(touch.y)
                n_points = 0
                length = 0
                sand[int(touch.x),int(touch.y)] = 1
            if touch.button=='right':
                Color(0,0,0,1)
                d=10.
                touch.ud['line'] = Line(points = (touch.x, touch.y), width = 10)
                last_x = int(touch.x)
                last_y = int(touch.y)
                n_points = 0
                length = 0
                sand[int(touch.x),int(touch.y)] = 0

    def on_touch_move(self, touch): # putting some sand when we move the mouse while pressing left
        global length,n_points,last_x,last_y
        if touch.button=='left':
            touch.ud['line'].points += [touch.x, touch.y]
            x = int(touch.x)
            y = int(touch.y)
            length += np.sqrt(max((x - last_x)**2 + (y - last_y)**2, 2))
            n_points += 1.
            density = n_points/(length)
            touch.ud['line'].width = int(20*density + 1)
            sand[int(touch.x) - 10 : int(touch.x) + 10, int(touch.y) - 10 : int(touch.y) + 10] = 1
            last_x = x
            last_y = y
        if touch.button=='right':
            touch.ud['line'].points += [touch.x, touch.y]
            x = int(touch.x)
            y = int(touch.y)
            length += np.sqrt(max((x - last_x)**2 + (y - last_y)**2, 2))
            n_points += 1.
            density = n_points/(length)
            touch.ud['line'].width = int(20*density + 1)
            sand[int(touch.x) - 10 : int(touch.x) + 10, int(touch.y) - 10 : int(touch.y) + 10] = 0
            last_x = x
            last_y = y

    # def update_clock(self, *args):
    #     # Called once a second using the kivy.clock module
    #     # Add one second to the current time and display it on the label
    #     self.now = self.now + timedelta(seconds = 1)
    #     self.my_label.text = self.now.strftime('%H:%M:%S')
        
    # def update(self):
    #     # self.main_text.text = str(random.randint(0,200))
    #     self.my_label = Label(text= self.now.strftime('%H:%M:%S'))

# API and switches interface (see kivy tutorials: https://kivy.org/docs/tutorials/pong.html)

class CarApp(App):

    def build(self): # building the app
        parent = Game()
        parent.serve_car()
        Clock.schedule_interval(parent.update, 1.0 / 60.0)
        self.painter = MyPaintWidget()
        clearbtn = Button(text='clear')
        # savebtn = Button(text='save',pos=(parent.width,0))
        # loadbtn = Button(text='load',pos=(2*parent.width,0))
        plotbtn = Button(text='plot',pos=(parent.width,0))
        clearbtn.bind(on_release=self.clear_canvas)
        # savebtn.bind(on_release=self.save)
        # loadbtn.bind(on_release=self.load)
        plotbtn.bind(on_release=self.plots)
        parent.add_widget(self.painter)
        parent.add_widget(clearbtn)
        # parent.add_widget(savebtn)
        # parent.add_widget(loadbtn)
        parent.add_widget(plotbtn)

        return parent

    def clear_canvas(self, obj): # clear button
        global sand
        self.painter.canvas.clear()
        sand = np.zeros((longueur,largeur))

    def save(self, obj): # save button
        print("saving brain...")
        brain.save()
        plt.figure(0)
        plt.plot(scores)
        plt.ylabel('Score')
        plt.figure(1)
        plt.plot(duration)
        plt.xlabel('Goals Reached')
        plt.ylabel('Duration (in timesteps)')
        plt.show(1)
    
    def plots(self, obj): # plot button
        print("plotting...")
        # plt.figure(1)
        # plt.plot(score_t)
        # plt.ylabel('Average Goals Reached')
        # plt.xlabel('Timesteps')
        # plt.show(1)
        
        # plt.figure(2)
        # plt.plot(collisions_t)
        # plt.ylabel('Average Collisions')
        # plt.xlabel('Timesteps')
        # plt.show(2)
        
        plt.figure(1)
        plt.plot(score_t)
        plt.ylabel('Mean Goals Achieved')
        plt.xlabel('Timesteps')
        plt.savefig('plots/scores_plot.png')
        plt.show(1)
        plt.close()
               
        plt.figure(22)
        plt.plot(collisions_t)
        plt.ylabel('Mean Collisions')
        plt.xlabel('Timesteps')
        plt.savefig('plots/collisions_plot.png')
        plt.close()
        
    def load(self, obj): # load button
        print("loading last saved brain...")
        brain.load()
        
    def on_pause(self):
        return True
        
# def reset():
#     import kivy.core.window as window
#     from kivy.base import EventLoop
#     if not EventLoop.event_listeners:
#         from kivy.cache import Cache
#         window.Window = window.core_select_lib('window', window.window_impl, True)
#         Cache.print_usage()
#         for cat in Cache._categories:
#             Cache._objects[cat] = {}
        
# # Running the app
# if __name__ == '__main__':
#     # reset()
timestep = 0
last_goal = 0
last_goal2 = 0
last_goal3 = 0
duration = []
duration2 = []
duration3 = []
score_t = []
collisions_t = []
total_collisions_prev = 0
CarApp().run()
