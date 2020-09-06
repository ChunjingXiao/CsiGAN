# CsiGAN

These are the code and data for the paper: [CsiGAN: Robust Channel State Information-based Activity Recognition with GANs](https://github.com/ChunjingXiao/CsiGAN/blob/master/CsiGAN%20Robust%20Channel%20State%20Information-based%20Activity%20Recognition%20with%20GANs.pdf), IEEE Internet of Things Journal, 2019, 6(6): 10191-10204. https://ieeexplore.ieee.org/document/8808929 

This model aims to deal with the performance degradation problem of leave-one-subject-out validation for CSI-based activity recognition, and can efficiently improve the recognition accuracy. 


CsiGAN is designed based on the semi-supervised GAN[1]. Figure 1 and Figure 2 illustrate the main frameworks of CsiGAN and the semi-supervised GAN[1]. CsiGAN is different from this semi-supervised GAN in three aspects: (1) the complement generator Gc is introduced into CsiGAN to produce complement fake samples. (2) The k probability outputs (1’• • • k’) and corresponding loss term are added for the discriminator. (3) Based on the introduced generator, the manifold regularization is proposed to stabilize the learning process (not shown in the framework figure).

![Figure 1. Framework of CsiGAN](https://github.com/ChunjingXiao/CsiGAN/blob/master/Framework_CsiGAN.png).
<p align="center">Figure 1. Framework of CsiGAN</p>


![Figure 2. Framework of the semi-supervised GAN](https://github.com/ChunjingXiao/CsiGAN/blob/master/Framework_SemiGAN.png).
<p align="center">Figure 2. Framework of the semi-supervised GAN</p>



All the experiments are the leave-one-subject-out validation. In other words, we select one user as the left-out user, and the others as the trained ones. For this left-out user, we further evenly divide its data of each category into the unlabeled data set and testing set.  

# Abstract

As a cornerstone service for many Internet of Things applications, Channel State Information (CSI) based activity recognition has received immense attention over recent years. However, recognition performance of general approaches might significantly decrease when applying the trained model to the left-out user whose CSI data are not used for model training. To overcome this challenge, we propose a semi-supervised Generative Adversarial Network (GAN) for CSI-based activity recognition, CsiGAN.
Based on general semi-supervised GANs, we mainly design three components for CsiGAN to meet the scenarios that unlabeled data form left-out users are very limited and enhance recognition performance. 1) We introduce a new complement generator, which can use limited unlabeled data to produce diverse fake samples for training a robust discriminator.
2) For the discriminator, we change the number of probability outputs from k+1 into 2k+1 (here k is the number of categories), which can help obtain the correct decision boundary for each category. 3) Based on the introduced generator, we propose a manifold regularization, which can stabilize the learning process. The experiments suggest that CsiGAN attains significant gains compared to state-of-the-art methods.


# CsiGANSemiSupervisedGAN

CsiGANSemiSupervisedGAN is for the Semi-supervised learning experiment of CsiGAN using SignFi data (https://github.com/yongsen/SignFi).  For this semi-supervised learning experiment, we select 10% of unlabeled data from the left-out user for training the model with training set.  The SignFi data should be formatted to fit our model. And the processed data are too large to be uploaded here, and can be downloaded at: https://pan.baidu.com/s/1L-vrJxJ6v0HujbwjKhG4-g  and the extracted code is: 2stz .

The codes are tested under python3.5 + tensorflow 1.8 + win7 and NVIDIA TITAN X 12GB GPU. 
To get the recognition accuracy:
  1.	Download the data at https://pan.baidu.com/s/1L-vrJxJ6v0HujbwjKhG4-g (pwd: 2stz), and put the 11 .mat files under the folder “CsiGANSemiSupervisedGAN\data”.
  2.	Run 1mat2npyCSI.py on Dos command:  python 1mat2npyCSI.py
  3.	Run 2mat2npyLabel.py: python 2mat2npyLabel.py
  4.	Run 3train_CSI_original_V40_2Unlabelled_C101.py: Python 3train_CSI_original_V40_2Unlabelled_C101.py


# CsiGANSupervisedGAN

CsiGANSupervisedGAN is for the Supervised learning experiment of CsiGAN using SignFi data (https://github.com/yongsen/SignFi).  For this semi-supervised learning experiment, there are not any data from the left-out user.  The SignFi data should be formatted to fit our model. And the processed data are too large to be uploaded here, and can be downloaded at: https://pan.baidu.com/s/1LV1pjl0-ITrbzAe0HCWUrA  and the extracted code is: 49gk.

The running steps are the same to the Semi-supervised learning experiment



# BaselineSemiSupManiGAN


BaselineSemiSupManiGAN is for the Semi-supervised learning experiment of the baseline: ManiGAN (Semisupervised learning with gans: Revisiting manifold regularization,ICLR 2018)

This experiment use the same data with CsiGANSemiSupervisedGAN, and the running steps are also the same to CsiGANSemiSupervisedGAN.


# BaselineSemiSupSSGAN

BaselineSemiSupManiGAN is for the Semi-supervised learning experiment of the baseline: SSGAN (Improved techniques for training gans,NIPS 2016)

This experiment use the same data with CsiGANSemiSupervisedGAN, and the running steps are also the same to CsiGANSemiSupervisedGAN.


# BaselineSupervisedManiGAN

BaselineSupervisedManiGAN is for the Supervised learning experiment of the baseline: ManiGAN (Semisupervised learning with gans: Revisiting manifold regularization,ICLR 2018)

This experiment use the same data with CsiGANSupervisedGAN, and the running steps are also the same to CsiGANSupervisedGAN.


# BaselineSupervisedSSGAN

BaselineSupervisedManiGAN is for the Supervised learning experiment of the baseline: SSGAN (Improved techniques for training gans,NIPS 2016)

This experiment use the same data with CsiGANSupervisedGAN, and the running steps are also the same to CsiGANSupervisedGAN.

# DatasetCodeForSignFi

These codes are used to generate the data for our experiments. These codes require the file “dataset_lab_150.mat” which can be downloaded from the paper signFi (https://github.com/yongsen/SignFi).


[1] T. Salimans, I. Goodfellow, W. Zaremba, V. Cheung, A. Radford, and X. Chen, “Improved techniques for training gans,” NIPS 2016.

