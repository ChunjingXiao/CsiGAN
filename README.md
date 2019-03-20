# CsiGAN

This model aims to deal with the performance degradation problem of leave-one-subject-out validation for CSI-based activity recognition, and can efficiently improve the recognition accuracy. 


All the experiments are the eave-one-subject-out validation. In other words, we select one user as the left-out user, and the others as the trained ones. For this left-out user, we further evenly divide its data of each category into the unlabeled data set and testing set.  
# Semi-supervised learning experiment

CsiGANSemiSupervisedGAN is for the Semi-supervised learning using SignFi data (https://github.com/yongsen/SignFi).  For this semi-supervised learning experiment, we select 10% of unlabeled data from the left-out user for training the model with training set.  The SignFi data should be formatted to fit our model. And the processed data are too large to be uploaded here, and can be downloaded at: https://pan.baidu.com/s/1L-vrJxJ6v0HujbwjKhG4-g  and the extracted code is: 2stz .

The codes are tested under python3.5 + tensorflow 1.8 + win7 and NVIDIA TITAN X 12GB GPU. 
To get the recognition accuracy:
  1.	Download the data at https://pan.baidu.com/s/1L-vrJxJ6v0HujbwjKhG4-g (pwd: 2stz), and put the 11 .mat files under the folder “CsiGANSemiSupervisedGAN\data”.
  2.	Run 1mat2npyCSI.py on Dos command:  python 1mat2npyCSI.py
  3.	Run 2mat2npyLabel.py: python 2mat2npyLabel.py
  4.	Run 3train_CSI_original_V40_2Unlabelled_C101.py: Python 3train_CSI_original_V40_2Unlabelled_C101.py


# Supervised learning experiment

CsiGANSupervisedGAN is for the Supervised learning using SignFi data (https://github.com/yongsen/SignFi).  For this semi-supervised learning experiment, there are not any data from the left-out user.  The SignFi data should be formatted to fit our model. And the processed data are too large to be uploaded here, and can be downloaded at: https://pan.baidu.com/s/1LV1pjl0-ITrbzAe0HCWUrA  and the extracted code is: 49gk.
The running steps are the same to the Semi-supervised learning experiment



