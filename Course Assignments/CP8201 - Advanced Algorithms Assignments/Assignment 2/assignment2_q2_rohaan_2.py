

import numpy as np

L1 = np.array([[0, 25],
     [23, 338],
     [45, 225],
     [68, 293],
     [135, 315],
     [180, 270],
     [240, 360]])

L2 = np.array([[0, 10],
     [20, 30],
     [40, 50]])

L3 = np.array([[10, 350],
     [20, 300],
     [40, 250]])

def relation(R, L):
    L1 = R
    L2 = L
    if (L1[0] < L2[0] and L1[1] < L2[1] and L2[0] < L1[1]):
        relation = 1 # cross
    elif (L1[0] < L2[0] and L1[1] > L2[1]):
        relation = 2 # L1 contains L2
    else:
        relation = 3 # L1 and L2 don't overlap
    return relation
    
def interval_scheduling (L, parent_interval_end = 360):
    R = L[0].reshape(-1,2)
    Rstart = L[0,0]
    Rend = L[0,1]
    maxR = np.array([])
    loop_count = 0
    
    for i in range (1, len(L)):
        loop_count += 1
        if L[i,1] < parent_interval_end:
            if (relation(R.reshape(-1,2)[-1], L[i]) == 2):
                Rtemp, lc = interval_scheduling(L[i:], Rend)
                Rtemp.reshape(-1,2)
                loop_count += lc
                if len(Rtemp) > len(R):
                    R = Rtemp.reshape(-1,2)
            elif (relation(R.reshape(-1,2)[-1], L[i]) == 1):
                for j in range (0, len(R)):
                    if (relation(R.reshape(-1,2)[j], L[i]) != 1):
                        R = R[1:].reshape(-1,2)
                R = np.append(R, L[i]).reshape(-1,2)
                Rend = R[-1,-1]
            elif ( relation(R.reshape(-1,2)[-1], L[i]) == 3 ):
                R = L[i].reshape(-1,2)
        if len(R) > len(maxR):
            maxR = R.reshape(-1,2)
    
    return maxR.reshape(-1,2), loop_count

maxR, loop_counter = interval_scheduling (L1, parent_interval_end = 360)

maxR, loop_counter = interval_scheduling (L2, parent_interval_end = 360)

maxR, loop_counter = interval_scheduling (L3, parent_interval_end = 360)