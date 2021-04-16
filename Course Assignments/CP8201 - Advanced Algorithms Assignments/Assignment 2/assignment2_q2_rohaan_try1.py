

import numpy as np


### ALGORITHM ###

### TEST CASES ###

L1_intersects =   [[0, 0, 1, 1, 0],
                  [0, 0, 1, 1, 0],
                  [1, 1, 0, 1, 0],
                  [1, 1, 1, 0, 0],
                  [0, 0, 0, 0, 0]]
L1 = [1,2,3,4,5]

## Approach 1: Assuming we know whether 2 lines intersect beforehand
L = np.array(L1) - 1

def intersects(row, col):
    '''
    O(1) function returns True if L1 and L2 intersect
    '''
    return L1_intersects[row][col] == 1

 
max_L = []

for i in range (len(L)):
    L_i = L[i]
    for j in range (len(L)):
        L_j = L[j]
        if (intersects(i, j)):
            if L_i not in max_L:
                max_L.append(L_i)
            if L_j not in max_L:
                max_L.append(L_j)

max_L = np.array(max_L) + 1

## Approach 2: Assuming we must calculate if two lines intersect
'''
https://www.geeksforgeeks.org/check-if-two-given-line-segments-intersect/
Two segments (p1,q1) and (p2,q2) intersect if and only if one of the following two conditions is verified
1. General Case:
– (p1, q1, p2) and (p1, q1, q2) have different orientations and
– (p2, q2, p1) and (p2, q2, q1) have different orientations.
2. Special Case
– (p1, q1, p2), (p1, q1, q2), (p2, q2, p1), and (p2, q2, q1) are all collinear and
– the x-projections of (p1, q1) and (p2, q2) intersect
– the y-projections of (p1, q1) and (p2, q2) intersect
'''

class Point: 
    def __init__(self, x, y): 
        self.x = x 
        self.y = y 
        
# Given three colinear points p, q, r, the function checks if  
# point q lies on line segment 'pr'  
def onSegment(p, q, r): 
    if ( (q.x <= max(p.x, r.x)) and (q.x >= min(p.x, r.x)) and 
           (q.y <= max(p.y, r.y)) and (q.y >= min(p.y, r.y))): 
        return True
    return False

def orientation(p, q, r): 
    # to find the orientation of an ordered triplet (p,q,r) 
    # function returns the following values: 
    # 0 : Colinear points 
    # 1 : Clockwise points 
    # 2 : Counterclockwise 
      
    # See https://www.geeksforgeeks.org/orientation-3-ordered-points/amp/  
    # for details of below formula.  
      
    val = (float(q.y - p.y) * (r.x - q.x)) - (float(q.x - p.x) * (r.y - q.y)) 
    if (val > 0): 
          
        # Clockwise orientation 
        return 1
    elif (val < 0): 
          
        # Counterclockwise orientation 
        return 2
    else: 
          
        # Colinear orientation 
        return 0
  
# The main function that returns true if  
# the line segment 'p1q1' and 'p2q2' intersect. 
def doIntersect(p1,q1,p2,q2): 
      
    # Find the 4 orientations required for  
    # the general and special cases 
    o1 = orientation(p1, q1, p2) 
    o2 = orientation(p1, q1, q2) 
    o3 = orientation(p2, q2, p1) 
    o4 = orientation(p2, q2, q1) 
  
    # General case 
    if ((o1 != o2) and (o3 != o4)): 
        return True
  
    # Special Cases 
  
    # p1 , q1 and p2 are colinear and p2 lies on segment p1q1 
    if ((o1 == 0) and onSegment(p1, p2, q1)): 
        return True
  
    # p1 , q1 and q2 are colinear and q2 lies on segment p1q1 
    if ((o2 == 0) and onSegment(p1, q2, q1)): 
        return True
  
    # p2 , q2 and p1 are colinear and p1 lies on segment p2q2 
    if ((o3 == 0) and onSegment(p2, p1, q2)): 
        return True
  
    # p2 , q2 and q1 are colinear and q1 lies on segment p2q2 
    if ((o4 == 0) and onSegment(p2, q1, q2)): 
        return True
  
    # If none of the cases 
    return False

# Driver program to test above functions: 
p1 = Point(1, 1) 
q1 = Point(10, 1) 
p2 = Point(1, 2) 
q2 = Point(10, 2) 
  
if doIntersect(p1, q1, p2, q2): 
    print("Yes") 
else: 
    print("No") 
  
p1 = Point(10, 0) 
q1 = Point(0, 10) 
p2 = Point(0, 0) 
q2 = Point(10,10) 
  
if doIntersect(p1, q1, p2, q2): 
    print("Yes") 
else: 
    print("No") 
  
p1 = Point(-5,-5) 
q1 = Point(0, 0) 
p2 = Point(1, 1) 
q2 = Point(10, 10) 
  
if doIntersect(p1, q1, p2, q2): 
    print("Yes") 
else: 
    print("No") 

p1 = Point(0, 1) 
q1 = Point(0, -1) 
p2 = Point(1, 0) 
q2 = Point(-1, 0) 
  
if doIntersect(p1, q1, p2, q2): 
    print("Yes") 
else: 
    print("No") 