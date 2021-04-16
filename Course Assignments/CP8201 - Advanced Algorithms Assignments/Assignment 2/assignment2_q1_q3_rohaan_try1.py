

import numpy as np

### TEST CASES ###

P1 = np.array([1, 2, 3, 4, 5, 6, 7, 8]) # n points

L1= np.array([[1, 8],
             [1, 5],
             [2, 5],
             [3, 6],
             [4, 7]]) # m lines

P2 = np.array([1, 2, 3, 4, 5, 6, 7, 8]) # n points

L2= np.array([[1, 8],
             [1, 5],
             [2, 5],
             [3, 6],
             [4, 7],
             [1, 6]]) # m lines

L3= np.array([[1, 2],
             [1, 3],
             [1, 4],
             [2, 3]]) # m lines

P3 = np.array([1, 2, 3, 4, 5, 6])
'''
S=[[0, pi], [0, pi/5], [pi/4, pi/2], [3pi/5, pi]]
S=[[0, pi], [0, pi/4], [pi/2, pi]]
S= [(3.008881694911577, 4.53153946133636),
 (1.9056812185241827, 3.008881694911577),
 (0.5661165304252112, 1.9056812185241827),
 (0.12451499204540153, 0.5661165304252112),
 (0.12451499204540153, 4.53153946133636),
 (0.12451499204540153, 1.9056812185241827)]
S=[(2.1977663205556324, 3.3246267218719012),
 (1.6901451468038171, 3.3246267218719012),
 (1.6901451468038171, 2.1977663205556324),
 (1.6901451468038171, 6.261962151869915),
 (1.6901451468038171, 3.031880785667277)]
S = [[0, pi], [pi/6, 2pi], [pi/5, 6pi/7], [5pi/6, ], [3pi/4]]
'''
### ALGORITHM ###

L = L3
P = P3

def maximal_L(P, L):
    def find_joints(P, L):
        '''
        O(m) function which loops through each line segment and finds which 
        points has multiple lines
        Returns:
            p_joint = Points with jointed line segments
            j_lines = Joint line segments
        '''
        points = np.zeros(len(P))
    
    
        for i in range (len(L)):
            points[L[i,0]-1] += 1
            points[L[i,1]-1] += 1
            
        p_joint = np.argwhere(points > 1) + 1
    
        j_lines = []
        for i in range (len(L)):
            if L[i,0] in p_joint or L[i,1] in p_joint:
                j_lines.append([L[i,0], L[i,1]])
        
        return p_joint, np.array(j_lines)
    
    p_joint, j_lines = find_joints(P, L)
    
    min_j_lines = j_lines
    max_L = L
    
    while(len(min_j_lines) != 0):
        for jl in j_lines:
            L_new = np.delete(max_L, np.where( np.bitwise_and( (max_L[:,0]==jl[0]), (max_L[:,1]==jl[1]) ) )[0], 0)
            p_joint_new, j_lines_new = find_joints(P, L_new)
            if len(j_lines_new) < len(min_j_lines):
                min_j_lines = j_lines_new
                L_new_max = L_new
        max_L = L_new_max
    
    return max_L

T = maximal_L(P, L)
