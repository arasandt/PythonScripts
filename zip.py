# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 18:38:08 2018

@author: 128537
"""

matrix3x4 = [
        [9,5,7,8],
        [8,4,4,3],
        [8,1,9,5]
        ]

matrix3x3 = [
        [4,2,1],
        [10,3,1],
        [10,10,10]
        ]

matrix_mul = [
        [0,0,0,0],
        [0,0,0,0],
        [0,0,0,0]
        ]

#ist(zip(*matrix3x4)))
matrix_mul = [[sum(x * y  for x ,y in zip(rows,cols)) for cols in zip(*matrix3x4)] for rows in matrix3x3]
print(matrix_mul)

#for rows in matrix3x3:
#    #print(rows)
#    for cols in zip(*matrix3x4):
#        #print(cols)
#        print(list(zip(rows,cols)))
        

