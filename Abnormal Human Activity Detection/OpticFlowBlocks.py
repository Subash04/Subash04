import numpy as np
import math

def calcOptFlowOfBlocks(mag,angle,grayImg):
    rows = grayImg.shape[0]   #rows that stores gray scale images
    cols = grayImg.shape[1]   #columns that stores gray scale images
    noOfRowInBlock = 20   #initialize noOfRowInBlock as 20
    noOfColInBlock = 20   #initialize noOfColInBlock as 20
    xBlockSize = rows // noOfRowInBlock   #Size of X block
    yBlockSize = cols // noOfColInBlock   #Size of Y block
    opFlowOfBlocks = np.zeros((xBlockSize,yBlockSize,2))   #both X & Y block as optical flow of blocks
    
    for index,value in np.ndenumerate(mag):
        opFlowOfBlocks[index[0]//noOfRowInBlock][index[1]//noOfColInBlock][0] += mag[index[0]][index[1]]
        opFlowOfBlocks[index[0]//noOfRowInBlock][index[1]//noOfColInBlock][1] += angle[index[0]][index[1]]

    centreOfBlocks = np.zeros((xBlockSize,yBlockSize,2))
    for index,value in np.ndenumerate(opFlowOfBlocks):
        opFlowOfBlocks[index[0]][index[1]][index[2]] = float(value)/(noOfRowInBlock*noOfColInBlock)
        val = opFlowOfBlocks[index[0]][index[1]][index[2]]   #index of optic flow blocks are stored in val variable

        if(index[2] == 1):
            angInDeg = math.degrees(val)
            if(angInDeg > 337.5):   #if angel is grater than 337.5 initilize k to zero
                k = 0
            else:   #else q,a1,q1,a2,q2 are changed
                q = angInDeg//22.5
                a1 = q*22.5
                q1 = angInDeg - a1
                a2 = (q+2)*22.5
                q2 =  a2 - angInDeg
                if(q1 < q2):   #checking q1 is lesser than q2
                    k = int(round(a1/45))
                else:
                    k = int(round(a2/45))        
            opFlowOfBlocks[index[0]][index[1]][index[2]] = k
            theta = val   #assigning val to theta

        if(index[2] == 0):
            r = val   #assigning val to r
            x = ((index[0] + 1)*noOfRowInBlock)-(noOfRowInBlock/2)   #center of block of X
            y = ((index[1] + 1)*noOfColInBlock)-(noOfColInBlock/2)   #center of block of Y
            centreOfBlocks[index[0]][index[1]][0] = x
            centreOfBlocks[index[0]][index[1]][1] = y
    return opFlowOfBlocks,noOfRowInBlock,noOfColInBlock,noOfRowInBlock*noOfColInBlock,centreOfBlocks,xBlockSize,yBlockSize
    #returns all values of optic flow block program
