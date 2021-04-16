"""
Created on Sun Oct  4 21:39:30 2020

@author: Rohaan Ahmed
    
    CP8201 Assignment 1
    
2. (60 marks) Let w be a string of brackets f and g. Then w is called balanced if it satis
es the following recursive de
nition:
- w is the empty string, or
- w is of the form f x g where x is a balanced string, or
- w is of the form xy where x and y are balanced strings.
For example, the strings w = fgfffggfgg and w = ffgfgg are balanced while the strings
w = ffgggfgg and w = ffgffgg are not.
The complexity of both of your algorithms should be linear in the length of the input
string.
"""

"""
# (i) Write an algorithm that determines whether a given string of parentheses is balanced.
# Analyze the complexity of your algorithm.

# In this and the subsequent sections, it is assumed that the order in which the brackets appear matters for the purposes of balanced vs unbalanced. 
# For example: w = {{}} is balanced, and w = }}{{ is unbalanced.
# """

# # f = { and g =}

# ww = ['fgfffggfgg', 
#     'ffgfgg',
#     'ffgggfgg',
#     'ffgffgg',
#     'ffffgffggg',
#     'ffgfggggg',
#     'ggff',
#     'ffgggfg']

# def is_balanced (w):
#     disbalance = 0
#     # This loop is O(n)
#     # iterate through the string, add 1 for open bracket, subtract 1 for closed bracked
#     for i in range (0, len(w)):
#         if w[i] == 'f':
#             disbalance += 1
#         elif w[i] == 'g':
#             disbalance -= 1
            
#         if disbalance < 0:
#             break
            
#     # Return disbalance amount, i.e., the number of brackets do not have a corresponding bracket
#     return abs(disbalance)

# for j in range (0,len(ww)):
    
#     w = ww[j]
#     disbalance = is_balanced (w)
    
#     if disbalance == 0:
#         print('w' + str(j) + ' is Balanced')
#     else:
#         print('w' + str(j) + ' is Unbalanced')

"""
(ii) Write a greedy algorithm to compute the length of the largest balanced substring
of a string of parentheses. Prove that your algorithm always outputs the correct
answer.

"""
# f = { and g =}
ww = ['fgfffggfgg', 
    'ffgfgg',
    'ffgggfgg',
    'ffgffgg',
    'ffffgffggg',
    'ffgfggggg',
    'ggfffg',
    'ffgggfg',
    'gffggfffgggf']

def len_longest_balanced_substring(w):
    
    n = len(w) 
    
# Create a stack and push -1 as initial index to it. 
    stack = [] 
    stack.append(-1) 
  
    # Initialize max_len 
    max_len = 0
    max_w = w
  
    # Traverse all characters of given string 
    for i in range (n): 
      
        # If opening bracket, push index of it 
        if w[i] == 'f': 
            stack.append(i) 
  
        else:    # If closing bracket, i.e., str[i] = ')' 
      
            # Pop the previous opening bracket's index 
            stack.pop() 
      
            # Check if this length formed with base of 
            # current valid substring is more than max  
            # so far 
            if len(stack) != 0: 
                substring_first_index = stack[len(stack)-1]
                substring_last_index = i
                if (substring_last_index - substring_first_index) > max_len:
                    max_w = w[substring_first_index+1:substring_last_index+1]
                    max_len = substring_last_index - substring_first_index
  
            # If stack is empty. push current index as  
            # base for next valid substring (if any) 
            else: 
                stack.append(i) 

    return max_len, max_w


for j in range (0,len(ww)):
    
    w = ww[j]
    max_len, max_w = len_longest_balanced_substring(w)
    print('length of largest balanced substring of w' + str(j) + ' is ' + str(max_len))
    print('largest balanced substring of w' + str(j) + ' is ' + max_w)
 

# """
# Finding largest substring

# In this section, it is assumed that the order in which the brackets appear does not matter for the purposes of balanced vs unbalanced. 
# For example: w = {{}} and w = }}{{ are both balanced.
# """

# for j in range (0,len(ww)):
#     print('___')
#     w = ww[j]
    
#     # Check if the initial string is balanced - O(n) operation
#     disbalance = is_balanced(w)
    
#     # If the string is balanced, print and exit
#     if disbalance == 0:
#         print('w' + str(j) + ': ' + w + ' is Balanced')
#     # If the string is unbalanced, print the disbalance number:
#     else:
#         print('w' + str(j) + ': ' + w + ' is Unbalanced')
#         print('there are ' + str(disbalance) + ' unbalanced brackets')
        
#         # We will use a sliding window of length (string_length - disbalance) and update
#         # the disbalance counter as we go along
        
#         # Select the first substring of length (string_length - disbalance)
#         new_substring = w[0:len(w)-disbalance]
#         # Check to see if substring is balanced - O(n) operation
#         new_disbalance = is_balanced(new_substring)
        
#         if new_disbalance != 0:
#             # Shift down by 1 by deleting the first element and adding 1 element to the end
#             # Executing this loop is an O(n) operation
#             for k in range (1,disbalance+1):
                
#                 # Subtract 1 to disbalance if deleting an open bracket, Add 1 if deleting close bracket
#                 if new_substring[0] == 'f':
#                     new_disbalance -= 1
#                 elif new_substring[0] == 'g':
#                     new_disbalance += 1
                
#                 # New string after shifting
#                 new_substring = w[k:len(w)-disbalance+k]
                
#                 # Add 1 to disbalance if adding an open bracket, Subtract 1 if adding close bracket
#                 if new_substring[len(new_substring)-1] == 'f':
#                     new_disbalance += 1
#                 elif new_substring[len(new_substring)-1] == 'g':
#                     new_disbalance -= 1
                
#                 if new_disbalance == 0:
#                     print(new_substring + ' is the largest Balanced substring of w' + str(j))
#                     break
#         else:
#             print(new_substring + ' is the largest Balanced substring of w' + str(j))
 