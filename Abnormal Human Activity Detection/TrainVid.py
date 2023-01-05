import numpy as np
import MotionInfGen as mig
import CreateBlocks as cb

def train_from_video(vid):
    print("Training Dataset From ", vid)  #get video file from the source
    MotionInfOfFrames, rows, cols = mig.getMotionInfuenceMap(vid)
    print("Motion Inf Map", len(MotionInfOfFrames))  #get the outcome of Motion Infuence Generator
    megaBlockMotInfVal = cb.createMegaBlocks(MotionInfOfFrames, rows, cols)  #compare the frames of created blocks and motion blocks
    np.save(r"C:\project\Dataset\videos\scene1\megaBlockMotInfVal_video_train1_40-40_k5.npy",megaBlockMotInfVal)   #save Mega Block Motion Influence Values as numpy file
    print(np.amax(megaBlockMotInfVal))
    print(np.amax(reject_outliers(megaBlockMotInfVal)))
    codewords = cb.kmeans(megaBlockMotInfVal)   #train algorithm using kmeans
    np.save(r"C:\project\Dataset\videos\scene1\codewords_video_train1_40-40_k5.npy",codewords)   #save codewords as numpy file
    print(codewords)  #to get the accuracy of trained dataset
    return

def reject_outliers(data, m=2):  #function to find reject outerlayers
    return data[abs(data - np.mean(data)) < m * np.std(data)]   #returns data

if __name__ == '__main__':  #call function
    trainingSet = [r"C:\project\Dataset\videos\scene1\train1.avi"]  #training video source file
    for video in trainingSet:
        train_from_video(video)  #pass the video from the source to function
    print("Done")  #completion of training is indicated
