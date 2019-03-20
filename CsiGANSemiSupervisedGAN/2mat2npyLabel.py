import numpy as np
import scipy.io as sio
import time



def saveNpyData(fileName):
    path= dataPath +fileName + ".mat"
    mat = sio.loadmat(path)
    data=mat[fileName]
    print(data.shape)
    label=np.zeros(data.shape[0])
    for i in range(label.shape[0]):
    #   #print(data[i])
        label[i]=data[i]-1
    #data= np.transpose(data,axes=[3,0,1,2])
    #label = mat['lb4']
    print(data.shape)
    np.save(dataPath + fileName + ".npy",label)
    #np.save('lb_4.npy',label)
    #for i in range(label.shape[0]):
        #print(label[i,0]
    
    print("done!")

#dataPath = "data200_60\\"
dataPath = "data\\"

fileName = "labelTest"
saveNpyData(fileName)

fileName = "labelTrain"
saveNpyData(fileName)

#dataPath = "data2\\"
fileName = "cycleLabelTrain_all"
saveNpyData(fileName)

#fileName = "labelleaveOne"
#saveNpyData(fileName)


time.sleep(2)