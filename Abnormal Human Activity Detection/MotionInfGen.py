import cv2
import numpy as np
import math
import OpticFlowBlocks as ofb

def angleBtw2Blocks(ang1,ang2):
    if(ang1-ang2 < 0):
        ang1InDeg = math.degrees(ang1)
        ang2InDeg = math.degrees(ang2)
        return math.radians(360 - (ang1InDeg-ang2InDeg))
    return ang1 - ang2

def calcEuclideanDist(t1,t2):   #function calculates Euclidean Distance
    (x1, y1) = t1   # storing x1,y1 in t1 to pass in function
    (x2, y2) = t2   # storing x2,y2 in t2 to pass in function
    dist = float(((x2-x1)**2 + (y2-y1)**2)**0.5)   #distance is calculated form x1,x2,y1,y2
    return dist   #returns distance as dist

def getCentreOfBlock(blck1Indx,blck2Indx,centreOfBlocks):   #function returns center of blocks in created mega block
    x1 = centreOfBlocks[blck1Indx[0]][blck1Indx[1]][0]   #x1 calculated form index value
    y1 = centreOfBlocks[blck1Indx[0]][blck1Indx[1]][1]   #y1 calculated form index value
    x2 = centreOfBlocks[blck2Indx[0]][blck2Indx[1]][0]   #x2 calculated form index value
    y2 = centreOfBlocks[blck2Indx[0]][blck2Indx[1]][1]   #y2 calculated form index value
    slope = float(y2-y1)/(x2-x1) if (x1 != x2) else float("inf")   #slope was calculated
    return (x1,y1),(x2,y2),slope

def motionInMapGenerator(opFlowOfBlocks,blockSize,centreOfBlocks,xBlockSize,yBlockSize):
    global frameNo
    motionInfVal = np.zeros((xBlockSize,yBlockSize,8))   #value of motion influence
    for index,value in np.ndenumerate(opFlowOfBlocks[...,0]):
        Td = getThresholdDistance(opFlowOfBlocks[index[0]][index[1]][0],blockSize)   #threshold distance value
        k = opFlowOfBlocks[index[0]][index[1]][1]   #index value of optic flow of blocks
        posFi, negFi = getThresholdAngle(math.radians(45*(k)))
        
        for ind,val in np.ndenumerate(opFlowOfBlocks[...,0]):
            if(index != ind):
                (x1,y1),(x2,y2), slope = getCentreOfBlock(index,ind,centreOfBlocks)   
                euclideanDist = calcEuclideanDist((x1,y1),(x2,y2))
        
                if(euclideanDist < Td):
                    angWithXAxis = math.atan(slope)
                    angBtwTwoBlocks = angleBtw2Blocks(math.radians(45*(k)),angWithXAxis)
        
                    if(negFi < angBtwTwoBlocks and angBtwTwoBlocks < posFi):
                        motionInfVal[ind[0]][ind[1]][int(opFlowOfBlocks[index[0]][index[1]][1])] += math.exp(-1*(float(euclideanDist)/opFlowOfBlocks[index[0]][index[1]][0]))
    frameNo += 1
    return motionInfVal

def getThresholdAngle(ang):
    tAngle = float(math.pi)/2
    return ang+tAngle,ang-tAngle

def getThresholdDistance(mag,blockSize):
    return mag*blockSize

def getMotionInfuenceMap(vid):
    global frameNo
    frameNo = 0
    cap = cv2.VideoCapture(vid)
    ret, frame1 = cap.read()
    rows, cols = frame1.shape[0], frame1.shape[1]
    print(rows,cols)
    prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    motionInfOfFrames = []
    count = 0
    while 1:
        print("Frame ",count," running")
        ret, frame2 = cap.read()
        if (ret == False):
            break
        next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        prvs = next
        opFlowOfBlocks,noOfRowInBlock,noOfColInBlock,blockSize,centreOfBlocks,xBlockSize,yBlockSize = ofb.calcOptFlowOfBlocks(mag,ang,next)
        motionInfVal = motionInMapGenerator(opFlowOfBlocks,blockSize,centreOfBlocks,xBlockSize,yBlockSize)
        motionInfOfFrames.append(motionInfVal)
        count += 1

    return motionInfOfFrames, xBlockSize,yBlockSize