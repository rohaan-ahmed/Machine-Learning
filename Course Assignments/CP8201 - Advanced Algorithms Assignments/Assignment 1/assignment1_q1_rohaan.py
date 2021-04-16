"""
@author: Rohaan Ahmed
    
CP8201 Assignment 1
    
Q1
(40 marks) An open interval, or simply an interval, of the real line R is denoted by (a, b) 
where a, b ∈ R and a < b. A covering set for an interval (a, b) ⊂ R is a set of intervals 
C = {(a1, a1), . . . , (an, bn)} such that (a, b) ⊆ (a1, b1) ∪ · · · ∪ (an, bn). The integer 
n is called the size of the covering set. A subset of intervals B ⊂ C that is a covering for 
X = (a, b) is a called a subcover of C. Given any covering set C for X, we can always ﬁnd a 
subcover of C of minimal size. For example, let X = (17, 21.3) and 
C = {(14, 17.1), (15, 18), (12.3, 20), (16.2, 20.7), (19, 22), (18, 21)}. Then C is a cover 
for X. A minimal subcover of C is B = {(12.3, 20), (19, 22)}.
"""
import numpy as np

X = np.array([17, 21.3])
C = np.array([[14, 17.1], [15, 18], [12.3, 20], [16.2, 20.7], [19, 22], [18, 21]])

# X = np.array([0, 10])
# C = np.array([[1, 9], [0, 7], [5, 10]])

a = X[0]
b = X[1]

R = []
depth = np.reshape(C[:,1]-X[0],(C.shape[0],1))

C2 = np.append(C, depth, axis=1)
C3 = C2[C2[:,2].argsort()]
C4 = C3[:,0:2]
C5 = C3[:,0:2]

current_start_point = a

while len(C5) > 0 and current_start_point < b:
    current_depth = 0
    greedy_sub_interval = 0
    for i in range (0,len(C5)):
        if current_start_point > C5[i,0] and current_start_point < C5[i,1]:
            if (C5[i,1] - current_start_point) > current_depth:
                current_depth = C5[i,1] - current_start_point
                greedy_sub_interval = i
    
    current_start_point = C5[greedy_sub_interval,1]
    R.append(C5[greedy_sub_interval,:])
    C5 = np.delete(C5,greedy_sub_interval, axis = 0)

R = np.array(R)
                
                
            