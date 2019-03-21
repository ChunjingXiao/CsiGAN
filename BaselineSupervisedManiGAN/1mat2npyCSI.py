import numpy as np
import scipy.io as sio
import time


def saveNpyData(fileName):
    path= dataPath + fileName + ".mat"
    mat = sio.loadmat(path)
    data=mat[fileName]
    print(data.shape)
    #label=np.zeros((data.shape[0],125))
    #for i in range(label.shape[0]):
        #print(data[i])
    #    label[i][data[i]-1]=1
    data= np.transpose(data,axes=[3,0,1,2])
    #label = mat['lb4']
    print(data.shape)
    np.save(dataPath + fileName + ".npy",data)
    #np.save('lb_4.npy',label)
    #for i in range(label.shape[0]):
        #print(label[i,0]
    
    print("done!")

    
#dataPath = "data200_60\\"
dataPath = "data\\"

fileName = "csi_tensorTest"
saveNpyData(fileName)


fileName = "csi_tensorTrain"
saveNpyData(fileName)

fileName = "csi_tensorUnlabel"
saveNpyData(fileName)


#dataPath = "data2\\"

fileName = "cycleCsi_tensorTrain"
saveNpyData(fileName)


fileName = "csi_tensorUnlabel"
saveNpyData(fileName)

fileName = "cycleCsi_tensorTrain_all"
saveNpyData(fileName)

fileName = "cycleCsi_tensorUnlabel"
saveNpyData(fileName)

#fileName = "csi_tensorleaveOne"
#saveNpyData(fileName)

time.sleep(2)
