import numpy as np
import cv2
import MotionInfGen as mig
import CreateBlocks as cb

def showUnusualActivities(unusual, vid, noOfRows, noOfCols, n):   #function used to detect unusual activity
    unusualFrames = sorted(unusual.keys())    #Unusual frames
    print(unusualFrames)   #unusual frames will be printed
    cap = cv2.VideoCapture(vid)   #capture the video form the source
    ret, frame = cap.read()
    rows, cols = frame.shape[0], frame.shape[1]
    rowLength = rows/(noOfRows/n)
    colLength = cols/(noOfCols/n)
    print("Mega Block Size ",(rowLength,colLength))   #row and column length of mega block
    count = 0
    screen_res = 980, 520   #resource of the screen
    scale_width = screen_res[0] / 320   #scale width
    scale_height = screen_res[1] / 240   #scale height
    scale = min(scale_width, scale_height)   #scale used to diplay
    window_width = int(320 * scale)   #Display Window Width
    window_height = int(240 * scale)   #Display Window Height
    cv2.namedWindow('Unusual Frame',cv2.WINDOW_NORMAL)   #name of the display window
    cv2.resizeWindow('Unusual Frame',window_width, window_height)   #size of the display window
    while 1:
        print(count)
        ret, uFrame = cap.read()
        if(count in unusualFrames):
            if (ret == False):
                break
            for blockNum in unusual[count]:
                print(blockNum)
                x1 = blockNum[1] * rowLength   #X1 Co-ordinates
                y1 = blockNum[0] * colLength   #Y1 Co-ordinates
                x2 = (blockNum[1]+1) * rowLength   #X2 Co-ordinates
                y2 = (blockNum[0]+1) * colLength   #Y2 Co-ordinates
                cv2.rectangle(uFrame,(int(x1),int(y1)),(int(x2),int(y2)),(0,0,255),1)
            print("Unusual frame number ",str(count))   #prints unusual number of frames
        cv2.imshow('Unusual Frame',uFrame)    #this shows a video output of unusual frames
        cv2.waitKey(40)   #speed of output
        count += 1

def constructMinDistMatrix(megaBlockMotInfVal,codewords, noOfRows, noOfCols, vid):
    threshold = 5.83682407063e-05   #threshold value for accuracy
    n = 2
    minDistMatrix = np.zeros((len(megaBlockMotInfVal[0][0]),(noOfRows//n),(noOfCols//n)))   #minimum distance matrix value
    for index,val in np.ndenumerate(megaBlockMotInfVal[...,0]):
        eucledianDist = []
        for codeword in codewords[index[0]][index[1]]:
            temp = [list(megaBlockMotInfVal[index[0]][index[1]][index[2]]),list(codeword)]   #index of codwords are validated
            dist = np.linalg.norm(megaBlockMotInfVal[index[0]][index[1]][index[2]]-codeword)    #eucledian distance of indexes
            eucDist = (sum(map(square,map(diff,zip(*temp)))))**0.5
            eucledianDist.append(eucDist)   #appending eucDist to eucledianDist
        minDistMatrix[index[2]][index[0]][index[1]] = min(eucledianDist)
    unusual = {}   #tuple of unusual
    for i in range(len(minDistMatrix)):
        if(np.amax(minDistMatrix[i]) > threshold):
            unusual[i] = []
            for index,val in np.ndenumerate(minDistMatrix[i]):
                if(val > threshold):
                        unusual[i].append((index[0],index[1]))
    print(unusual)
    showUnusualActivities(unusual, vid, noOfRows, noOfCols, n)

def test_video(vid):
    print("Test video ", vid)
    MotionInfOfFrames, rows, cols = mig.getMotionInfuenceMap(vid)   #sending motion influence frames to MotionInfGen program
    megaBlockMotInfVal = cb.createMegaBlocks(MotionInfOfFrames, rows, cols)
    np.save(r"C:\project\Dataset\videos\scene1\megaBlockMotInfVal_video_test1_20-20_k5.npy",megaBlockMotInfVal)   #saves mega block data in respective file location
    codewords = np.load(r"C:\project\Dataset\videos\scene1\codewords_video_train1_40-40_k5.npy")   # calls codeword from respective file location
    print("codewords",codewords)
    listOfUnusualFrames = constructMinDistMatrix(megaBlockMotInfVal,codewords,rows, cols, vid)   #number of unusual are listed and stored
    return

def square(a):   #square function
    return (a**2)

def diff(l):   #difference function
    return (l[0] - l[1])

if __name__ == '__main__':
    testSet = [r"C:\project\Dataset\videos\scene1\test1.avi"]    #file location of test video
    for video in testSet:
        test_video(video)   #source of test video
    print("Done")
