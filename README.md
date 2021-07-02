#  Adapted from --> https://www.kaggle.com/bjoernjostein/physionet-challenge-2020/data


# ECG Arrhythmia Classifcation with Deep Neural Networks

## TODO
- Copy to SCC

# Overview of Included Files

models folder --> Stores the model weights
data --> contains only the MIT-BIH dataset and instructions on how to get the other datasets


# Datasets
  - MIT-BIH
    
  - Physionet Challange 2020 (SNOMED Mapping)
    - 43101 Recordings (See distribution of classes in data below)
    - ![alt text](https://github.com/varocarras/ECG-523/blob/main/Images/Arrhythmia-Distribution.png?raw=true)

# Results
  - Metics Used
    - F1 Score --> F1 
![alt text](https://github.com/varocarras/ECG-523/blob/main/Images/RESNET-ConfusionM.png?raw=true)

 - IAVB = 1st Degree AV Block
 - AF = Atrial Fibrillation
 - AFL = Atrial Flutter
 - Brady = Bradycardia
 - CRBBB = Complete Right Bundle Branch Block
 - IRBBB = Incomplete Right Bundle Branch Block
 - LAnFB = Left Anterior Fascicular Block
 - LAD = Left Axis Deviation
 - LBBB = Left Bundle Branch Block
 - LQRSV = Low QRS Voltage
 - NSIVCB = Nonspecific Intraventricular Conduction Disorder
 - PR = Pacing Rhythm
 - PAC = Premature Atrial Contraction
 - PVC = Premature Ventricular Contraction
 - PR = Prolonged PR Interval
 - LQT = Prolonged QT Interval
 - QAb = Qwave Abnormal
 - RAD = Right Axis Deviation
 - RBBB = Right Bundle Branch Block
 - SA = Sinus Arrhythmia
 - SB = Sinus Bradycardia
 - SNR = Sinus Rhythm
 - STach = Sinus Tachycardia
 - SVPB = Supraventricular Premature Beats
 - TAb = T Wave Abnormal
 - TInv = T Wave Inversion
 - VPB = Ventricular Premature Beats

# Demo Code

- The demo code allows you to use our trained model to predict a rhythm for a given waveform.
- Run the scipt to query the network and choose the algorithm to use

## (Detector)

## (Diagnoser)


# References

  1. Stanford Paper: Cardiologist Level Aryhthmia Detection with Convolutional Neural Networks -->
https://arxiv.org/abs/1707.01836

  2. Physionet Challange and Python Notebook -->
https://www.kaggle.com/bjoernjostein/physionet-challenge-2020/data
