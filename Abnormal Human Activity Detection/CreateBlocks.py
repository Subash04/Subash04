import cv2
import numpy as np

def kmeans(megaBlockMotInfVal):   #Kmeans Algorithm
    cluster_n = 5
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS
    codewords = np.zeros((len(megaBlockMotInfVal),len(megaBlockMotInfVal[0]),cluster_n,8))   #Codewords
    for row in range(len(megaBlockMotInfVal)):
        for col in range(len(megaBlockMotInfVal[row])):
            ret, labels, cw = cv2.kmeans(np.float32(megaBlockMotInfVal[row][col]), cluster_n, None, criteria,10,flags)
            codewords[row][col] = cw
    return(codewords)   #Return Codewords

def createMegaBlocks(motionInfoOfFrames,noOfRows,noOfCols):   #Function used to create mega blocks
    n = 2   #initilize n as 2
    megaBlockMotInfVal = np.zeros(((noOfRows//n),(noOfCols//n),len(motionInfoOfFrames),8))
    frameCounter = 0   #initilize frameCounter as 0
    for frame in motionInfoOfFrames:
        for index,val in np.ndenumerate(frame[...,0]):
            temp = [list(megaBlockMotInfVal[index[0]//n][index[1]//n][frameCounter]),list(frame[index[0]][index[1]])]
            megaBlockMotInfVal[index[0]//n][index[1]//n][frameCounter] = np.array(list(map(sum, zip(*temp))))
        frameCounter += 1
    print(((noOfRows/n),(noOfCols/n),len(motionInfoOfFrames)))
    return megaBlockMotInfVal   #Return megaBlockMotInfVal
