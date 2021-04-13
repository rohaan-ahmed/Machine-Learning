# Self Driving Car

# Importing the libraries
import numpy as np
from random import random, randint, uniform
import matplotlib.pyplot as plt
import time
from datetime import datetime
from datetime import timedelta

# Importing the Kivy packages
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.graphics import Color, Ellipse, Line, Rectangle
from kivy.config import Config
from kivy.properties import NumericProperty, ReferenceListProperty, ObjectProperty
from kivy.vector import Vector
from kivy.clock import Clock
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label

# Importing the Dqn object from our AI in ia.py
from rohaan_ai import DQN_car
from rohaan_ai import DQN_car_cluster

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# Adding this line if we don't want the right click to put a red point
Config.set('input', 'mouse', 'mouse, multitouch_on_demand')
Config.set('graphics', 'width', '1000')
Config.set('graphics', 'height', '800')

# Introducing last_x and last_y, used to keep the last point in memory when we draw the sand on the map
last_x = 0
last_y = 0
n_points = 0 # the total number of points in the last drawing
length = 0 # the length of the last drawing

# Getting our AI, which we call "car1_ai", and that contains our neural network that represents our Q-function
car1_ai = DQN_car(5,3,0.9) # 5 sensors, 3 actions, gama = 0.9
car2_ai = DQN_car(5,3,0.9) # 5 sensors, 3 actions, gama = 0.9
cluster_ai = DQN_car_cluster(14, 10, 0.9) # State: car1 1, car1 y, car1 velocity x, car1 velocity y, car1 orientation, x2, y2, vx2, vy2, o2, x_goal1, y_goal1, x_goal2, y_goal2, 
                                         # Actions: Rewards for living penalty, getting closer bonus, collision penalty, sand penalty, goal reached reward for both cars
action2rotation = [0,20,-20] # action = 0 => no rotation, action = 1 => rotate 20 degres, action = 2 => rotate -20 degres
last_reward = 0 # initializing the last reward
last_reward2 = 0 # initializing the last reward
last_cluster_reward = 0
scores = [] # initializing the mean score curve (sliding window of the rewards) with respect to time
scores2 = []

# Rewards
# LIVING_PENALTY = -0.2 # Penalty for not achieving the goal
# GETTING_CLOSER_BONUS = 0.1 # Reward for getting closer to goal
# COLLISION_PENALTY = 15 # Penalty for colling with the other car
# SAND_PENALTY = -1.5 # Penalty for going over sand-trap
PROXIMITY_TO_GOAL = 50 # How close to the goal is good enough
GOAL_ACHIEVED_REWARD = 50
# global living_penalty1
# global getting_closer_bonus1
# global collision_penalty1
# global sand_penalty1
# global goal_reached1
# global living_penalty2
# global getting_closer_bonus2
# global collision_penalty2
# global sand_penalty2
# global goal_reached2

living_penalty1 = uniform(0,1) * -1
getting_closer_bonus1 = uniform(0,1)
collision_penalty1 = uniform(0,100) * -1
sand_penalty1 = uniform(0,50) * -1
goal_reached1 = uniform(0,100)
living_penalty2 = uniform(0,1) * -1
getting_closer_bonus2 = uniform(0,1)
collision_penalty2 = uniform(0,100) * -1
sand_penalty2 = uniform(0,50) * -1
goal_reached2 = uniform(0,100)
            
# Initializing the map
first_update = True # using this trick to initialize the map only once
def init():
    global sand # sand is an array that has as many cells as our graphic interface has pixels. Each cell has a one if there is sand, 0 otherwise.
    global goal_x # x-coordinate of the goal (where the car has to go, that is the airport or the downtown)
    global goal_y # y-coordinate of the goal (where the car has to go, that is the airport or the downtown)
    global goal_x2 # x-coordinate of the goal (where the car has to go, that is the airport or the downtown)
    global goal_y2 # y-coordinate of the goal (where the car has to go, that is the airport or the downtown)
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

class Game(Widget):
    
    car = ObjectProperty(None) # getting the car object from our kivy file
    ball1 = ObjectProperty(None) # getting the sensor 1 object from our kivy file
    ball2 = ObjectProperty(None) # getting the sensor 2 object from our kivy file
    ball3 = ObjectProperty(None) # getting the sensor 3 object from our kivy file
    car2 = ObjectProperty(None) # getting the car object from our kivy file
    ball21 = ObjectProperty(None) # getting the sensor 1 object from our kivy file
    ball22 = ObjectProperty(None) # getting the sensor 2 object from our kivy file
    ball23 = ObjectProperty(None) # getting the sensor 3 object from our kivy file
    timestep_label = ObjectProperty(None)
    
    def serve_car(self): # starting the car when we launch the application
        self.car.center = self.center # the car will start at the center of the map
        self.car.velocity = Vector(6, 0) # the car will start to go horizontally to the right with a speed of 6
        self.car.center[1] = self.car.center[1] - 10
        self.car2.center = self.center # the car will start at the center of the map
        self.car2.center[1] = self.car2.center[1] + 10
        self.car2.velocity = Vector(-6, 0) # the car will start to go horizontally to the left with a speed of 6
        print(self.car.center)

    def update(self, dt): # the big update function that updates everything that needs to be updated at each discrete time t when reaching a new state (getting new signals from the sensors)
        
        # Shared Globals
        global timestep
        global longueur # specifying the global variables (width of the map)
        global largeur # specifying the global variables (height of the map)
        
        # Car 1 Globals
        global car1_ai # specifying the global variables (the brain of the car, that is our AI)
        global last_reward # specifying the global variables (the last reward)
        global scores # specifying the global variables (the means of the rewards)
        global last_distance # specifying the global variables (the last distance from the car to the goal)
        global goal_x # specifying the global variables (x-coordinate of the goal)
        global goal_y # specifying the global variables (y-coordinate of the goal)
        global last_goal
        global duration
        
        # Car 2 Globals
        global car2_ai # specifying the global variables (the brain of the car2, that is our AI)
        global last_reward2 # specifying the global variables (the last reward)
        global scores2 # specifying the global variables (the means of the rewards)
        global last_distance2 # specifying the global variables (the last distance from the car to the goal)
        global goal_x2 # specifying the global variables (x-coordinate of the goal)
        global goal_y2 # specifying the global variables (y-coordinate of the goal)
        global last_goal2
        global duration2
        
        global living_penalty1
        global getting_closer_bonus1
        global collision_penalty1
        global sand_penalty1
        global goal_reached1
        global living_penalty2
        global getting_closer_bonus2
        global collision_penalty2
        global sand_penalty2
        global goal_reached2
    
        global last_cluster_reward
        
        global total_goals_achieved
        global goals_achieved_car1
        global goals_achieved_car2
        global total_collisions
        global total_collisions_prev
        global score_t
        global score_car1_t
        global score_car2_t
        global collisions_t
        
        
        longueur = self.width # width of the map (horizontal edge)
        largeur = self.height # height of the map (vertical edge)

        goal_1_reached = False
        goal_2_reached = False
        
        if first_update: # trick to initialize the map only once
            goal_x = randint(10,self.width - 10)
            goal_y = randint(10,self.height - 10)
            goal_x2 = randint(10,self.width - 10)
            goal_y2 = randint(10,self.height - 10)
            
            living_penalty1 = uniform(0,1) * -1
            getting_closer_bonus1 = uniform(0,1)
            collision_penalty1 = uniform(0,100) * -1
            sand_penalty1 = uniform(0,50) * -1
            goal_reached1 = uniform(0,100)
            goal_reached1 = GOAL_ACHIEVED_REWARD
            living_penalty2 = uniform(0,1) * -1
            getting_closer_bonus2 = uniform(0,1)
            collision_penalty2 = uniform(0,100) * -1
            sand_penalty2 = uniform(0,50) * -1
            goal_reached2 = uniform(0,100)
            goal_reached2 = GOAL_ACHIEVED_REWARD
            
            total_goals_achieved = 0
            goals_achieved_car1 = 0
            goals_achieved_car2 = 0
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
        action = car1_ai.update(last_reward, last_signal) # playing the action from our ai (the object car1_ai of the dqn class)
        timestep += 1
        scores.append(car1_ai.score()) # appending the score (mean of the last 100 rewards to the reward window)
        rotation = action2rotation[action] # converting the action played (0, 1 or 2) into the rotation angle (0°, 20° or -20°)
        self.car.move(rotation) # moving the car according to this last rotation angle
        distance = np.sqrt((self.car.x - goal_x)**2 + (self.car.y - goal_y)**2) # getting the new distance between the car and the goal right after the car moved
        self.ball1.pos = self.car.sensor1 # updating the position of the first sensor (ball1) right after the car moved
        self.ball2.pos = self.car.sensor2 # updating the position of the second sensor (ball2) right after the car moved
        self.ball3.pos = self.car.sensor3 # updating the position of the third sensor (ball3) right after the car moved

        if sand[int(self.car.x),int(self.car.y)] > 0: # if the car is on the sand
            self.car.velocity = Vector(1, 0).rotate(self.car.angle) # it is slowed down (speed = 1)
            last_reward = sand_penalty1 # and reward = -1
        else: # otherwise
            self.car.velocity = Vector(6, 0).rotate(self.car.angle) # it goes to a normal speed (speed = 6)
            last_reward = living_penalty1 # and it gets bad reward (-0.2)
            if distance < last_distance: # however if it getting close to the goal
                last_reward += getting_closer_bonus1 # it still gets slightly positive reward 0.1

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
            goals_achieved_car1 += 1
            if last_goal != 0:
                duration.append(timestep-last_goal)
            if len(duration) > 20:
                del duration[0]
            last_goal = timestep
            last_reward = goal_reached1

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
        action2 = car2_ai.update(last_reward2, last_signal2) # playing the action from our ai (the object brain of the dqn class)
        scores2.append(car2_ai.score()) # appending the score (mean of the last 100 rewards to the reward window)
        rotation2 = action2rotation[action2] # converting the action played (0, 1 or 2) into the rotation angle (0°, 20° or -20°)
        self.car2.move(rotation2) # moving the car according to this last rotation angle
        distance2 = np.sqrt((self.car2.x - goal_x2)**2 + (self.car2.y - goal_y2)**2) # getting the new distance between the car and the goal right after the car moved
        self.ball21.pos = self.car2.sensor1 # updating the position of the first sensor (ball1) right after the car moved
        self.ball22.pos = self.car2.sensor2 # updating the position of the second sensor (ball2) right after the car moved
        self.ball23.pos = self.car2.sensor3 # updating the position of the third sensor (ball3) right after the car moved

        if sand[int(self.car2.x),int(self.car2.y)] > 0: # if the car is on the sand
            self.car2.velocity = Vector(1, 0).rotate(self.car2.angle) # it is slowed down (speed = 1)
            last_reward2 = sand_penalty1 # and reward = -1
        else: # otherwise
            self.car2.velocity = Vector(6, 0).rotate(self.car2.angle) # it goes to a normal speed (speed = 6)
            last_reward2 = living_penalty1 # and it gets bad reward (-0.2)
            if distance2 < last_distance2: # however if it getting close to the goal
                # print('getting_closer_bonus2')
                # print(getting_closer_bonus2)
                last_reward2 += getting_closer_bonus1 # it still gets slightly positive reward 0.1

        if self.car2.x < 10: # if the car is in the left edge of the frame
            self.car2.x = 10 # it is not slowed down
            last_reward2 = -100 # but it gets bad reward -1
        if self.car2.x > self.width-10: # if the car is in the right edge of the frame
            self.car2.x = self.width-10 # it is not slowed down
            last_reward2 = -100 # but it gets bad reward -1
        if self.car2.y < 10: # if the car is in the bottom edge of the frame
            self.car2.y = 10 # it is not slowed down
            last_reward2 = -100 # but it gets bad reward -1
        if self.car2.y > self.height-10: # if the car is in the upper edge of the frame
            self.car2.y = self.height-10 # it is not slowed down
            last_reward2 = -100 # but it gets bad reward -1

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
            goal_2_reached = True
            goals_achieved_car2 += 1
            if last_goal2 != 0:
                duration2.append(timestep-last_goal2)
            if len(duration2) > 20:
                del duration2[0]
            last_goal2 = timestep
            last_reward2 = goal_reached1
            

        # Updating the last distance from the car to the goal
        last_distance2 = distance2
        
        # Car collision
        
        if abs(self.car.x - self.car2.x) < 15 and abs(self.car.y - self.car2.y) < 15: # if the cars collide
            self.car.x = 1 # slow down
            self.car2.x = 1 
            print('COLLISION!')
            last_reward = collision_penalty1 # large penalty
            last_reward2 = collision_penalty1
            total_collisions += 1
        
        # Cluster 
        cluster_state = [self.car.x, self.car.y, self.car.velocity[0], self.car.velocity[1], orientation,
                         self.car2.x, self.car2.y, self.car2.velocity[0], self.car2.velocity[1], orientation2,
                         goal_x, goal_y, goal_x2, goal_y2]
        # print(cluster_state)
        cluster_actions = cluster_ai.update(last_cluster_reward, cluster_state) # playing the action from our ai (the object car1_ai of the dqn class)
        # print('---cluster_actions---')
        # print(cluster_actions)
        
        # living_penalty1 = cluster_actions[0][0].item()
        # getting_closer_bonus1 =cluster_actions[0][1].item()
        # collision_penalty1 = cluster_actions[0][2].item()
        # sand_penalty1 = cluster_actions[0][3].item()
        # goal_reached1 = cluster_actions[0][4].item()
        # living_penalty2 = cluster_actions[0][5].item()
        # getting_closer_bonus2 = cluster_actions[0][6].item()
        # collision_penalty2 = cluster_actions[0][7].item()
        # sand_penalty2 = cluster_actions[0][8].item()
        # goal_reached2 = cluster_actions[0][9].item()
        
        living_penalty1 = cluster_actions[0].item()
        getting_closer_bonus1 =cluster_actions[1].item()
        collision_penalty1 = cluster_actions[2].item()
        sand_penalty1 = cluster_actions[3].item()
        goal_reached1 = cluster_actions[4].item()
        goal_reached1 = GOAL_ACHIEVED_REWARD
        living_penalty2 = cluster_actions[5].item()
        getting_closer_bonus2 = cluster_actions[6].item()
        collision_penalty2 = cluster_actions[7].item()
        sand_penalty2 = cluster_actions[8].item()
        goal_reached2 = cluster_actions[9].item()
        goal_reached2 = GOAL_ACHIEVED_REWARD
        
        # print('REWARDS', end='\r')
        print('-- REWARDS -- \n' \
            'living_penalty1 {0} \n'  \
            'getting_closer_bonus1 {1} \n'  \
            'collision_penalty1 {2} \n' \
            'sand_penalty1 {3} \n' \
            'goal_reached1 {4} \n' \
            'living_penalty2 {5} \n' \
            'getting_closer_bonus2 {6} \n' \
            'collision_penalty2 {7} \n' \
            'sand_penalty2 {8} \n' \
            'goal_reached2 {9} \n'.format( \
            living_penalty1,\
            getting_closer_bonus1,\
            collision_penalty1,\
            sand_penalty1,\
            goal_reached1,\
            living_penalty2,\
            getting_closer_bonus2,\
            collision_penalty2,\
            sand_penalty2,\
            goal_reached2), end='\r')
            
        # print('-- REWARDS: -- ' \
        #     'living_penalty1 {0} '  \
        #     'getting_closer_bonus1 {1} '  \
        #     'collision_penalty1 {2} ' \
        #     'sand_penalty1 {3} ' \
        #     'goal_reached1 {4} ' \
        #     'living_penalty2 {5} ' \
        #     'getting_closer_bonus2 {6} ' \
        #     'collision_penalty2 {7} ' \
        #     'sand_penalty2 {8} ' \
        #     'goal_reached2 {9} \r'.format( \
        #     living_penalty1,\
        #     getting_closer_bonus1,\
        #     collision_penalty1,\
        #     sand_penalty1,\
        #     goal_reached1,\
        #     living_penalty2,\
        #     getting_closer_bonus2,\
        #     collision_penalty2,\
        #     sand_penalty2,\
        #     goal_reached2))
        
        # last_cluster_reward = last_reward + last_reward2
        if goal_1_reached == True or goal_2_reached == True:
            last_cluster_reward = 50
            goal_1_reached = False
            goal_2_reached = False
            total_goals_achieved += 1
        else:
            last_cluster_reward = -10
            if distance2 < last_distance2 or distance2 < last_distance2: # however if it getting close to the goal
                last_cluster_reward += 5 
        
        self.display_text.text = 'Timestep: ' + str(timestep) + '\n' + \
            'Mean Score (50 timesteps):\n' +\
            'Rover 1: ' + str(round(score_car1_t[-1])) + '\n' + \
            'Rover 2: ' + str(round(score_car2_t[-1])) + '\n' + \
            'Total Goals Achieved: ' + str(total_goals_achieved)
        
        if (timestep % 50) == 0:
            score_t.append(total_goals_achieved/timestep)
            collisions_t.append(total_collisions/timestep)
            score_car1_t.append(goals_achieved_car1/timestep)
            score_car2_t.append(goals_achieved_car2/timestep)
            # goals_achieved_car1 = 0
            # goals_achieved_car2 = 0
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
                
                plt.figure(33)
                plt.plot(score_car1_t)
                plt.plot(score_car2_t)
                plt.ylabel('Goals Achieved per 50 Timesteps')
                plt.xlabel('Timesteps')
                plt.legend('Rover 1', 'Rover 2')
                plt.savefig('plots/rover_scores_plot.png')
                plt.close()
                
                if (timestep % 500) == 0:
                    cluster_ai.save_load_best_model(score=score_t[-1])
                    car1_ai.save_load_best_model(score=score_car1_t[-1])
                    car2_ai.save_load_best_model(score=score_car2_t[-1])
                
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

# API and switches interface (see kivy tutorials: https://kivy.org/docs/tutorials/pong.html)

class CarApp(App):

    def build(self): # building the app
        parent = Game()
        parent.serve_car()
        Clock.schedule_interval(parent.update, 1.0 / 60.0)
        self.painter = MyPaintWidget()
        clearbtn = Button(text='clear')
        saveloadbtn = Button(text='save/load best',pos=(parent.width,0))
        # loadbtn = Button(text='load best',pos=(2*parent.width,0))
        plotbtn = Button(text='plot',pos=(2*parent.width,0))
        clearbtn.bind(on_release=self.clear_canvas)
        saveloadbtn.bind(on_release=self.save_load)
        # loadbtn.bind(on_release=self.load)
        plotbtn.bind(on_release=self.plots)
        parent.add_widget(self.painter)
        parent.add_widget(clearbtn)
        parent.add_widget(saveloadbtn)
        # parent.add_widget(loadbtn)
        parent.add_widget(plotbtn)

        return parent

    def clear_canvas(self, obj): # clear button
        global sand
        self.painter.canvas.clear()
        sand = np.zeros((longueur,largeur))

    def save_load(self, obj): # save button
        print("saving or loading best models")
        cluster_ai.save_load_best_model(score_t[-1])
        car1_ai.save_load_best_model(score=score_car1_t[-1])
        car2_ai.save_load_best_model(score=score_car2_t[-1])
                    
    def plots(self, obj): # plot button
        print("plotting...")
        
        plt.figure(1)
        plt.plot(score_t)
        plt.ylabel('Mean Goals Achieved')
        plt.xlabel('Timesteps')
        plt.savefig('plots/scores_plot.png')
        plt.show(1)
        plt.close()
               
        plt.figure(2)
        plt.plot(collisions_t)
        plt.ylabel('Mean Collisions')
        plt.xlabel('Timesteps')
        plt.savefig('plots/collisions_plot.png')
        plt.show(2)
        plt.close()
        
        plt.figure(3)
        plt.plot(score_car1_t)
        plt.plot(score_car2_t)
        plt.ylabel('Mean Score')
        plt.xlabel('Timesteps')
        plt.legend('Rover 1', 'Rover 2')
        plt.savefig('plots/rover_scores_plot.png')
        plt.show(3)
        plt.close()
                
    def load(self, obj): # load button
        print("loading last saved brain...")
        car1_ai.load()
        car2_ai.load()
        
    def on_pause(self):
        return True
        
# Running the app
if __name__ == '__main__':
    # reset()
    timestep = 0
    last_goal = 0
    last_goal2 = 0
    duration = []
    duration2 = []
    score_t = []
    score_car1_t = [0]
    score_car2_t = [0]
    collisions_t = []
    total_collisions_prev = 0
    CarApp().run()
