#  Adapted from --> https://www.kaggle.com/bjoernjostein/physionet-challenge-2020/data


# ECG Arrhythmia Classifcation with Deep Neural Networks

## TODO
- Copy to SCC

# Overview of Included Files

- Models    --> Stores the models and their corresponding weights
- Data      --> Contains the MIT-BIH data and the links to the additional required data (Too big for GitHub)
- Utilities --> Contains helper functions provided for the Physionet challenge
- Slides    --> Contains slides for the in-class presentation
- Images    --> Contains data visualizations and other generated useful representations and results

# Example of Input file

![alt text](https://github.com/varocarras/ECG-523/blob/main/Images/ECG-Sample.png?raw=true)

# Datasets (Training)
  - MIT-BIH Arrhythmia dataset
  - China 12-Lead ECG Challenge Database
  - China Physiological Signal Challenge in 2018
  - Georgia 12-Lead ECG Challenge Database
  - Physionet Challenge 2020 (SNOMED Mapping)
    - 43101 Recordings (See distribution of classes in data below)
    - ![alt text](https://github.com/varocarras/ECG-523/blob/main/Images/Arrhythmia-Distribution.png?raw=true)
  - PTB Diagnosis ECG Database
  - PTB-XL electrocardiography Database
  - St Petersburg INCART 12-lead Arrhythmia Dataset
  

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

- The demo code allows for training the model as well as loading its weights to predict a rhythm for any given waveform.
- Run the script to generate visualizations and show an example of predicting a rhythm for a particular patient. 
- In order to retrain the model:
  - Run train.py (root folder file) or uncomment the training code from the Demo.ipynb
  - Weights will save to /Models/resnet_model.h5 and will load from there to test
  - We left the weights there in case you dont want to train the model again

## (Detector)

## (Diagnoser)


# References

  1. Stanford Paper: Cardiologist Level Aryhthmia Detection with Convolutional Neural Networks -->
https://arxiv.org/abs/1707.01836

  2. Physionet Challange and Python Notebook -->
https://www.kaggle.com/bjoernjostein/physionet-challenge-2020/data
