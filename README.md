

Note: Project and Data Adapted from --> https://www.kaggle.com/bjoernjostein/physionet-challenge-2020/

# ECG Arrhythmia Classifcation with Deep Neural Networks

## Team Members
- Anthony Sayegh (asayegh@bu.edu)
- Alvaro Carrascosa Penabad (varo@bu.edu)

# Included Files

- Models    --> Stores the models and their corresponding weights
- Data      --> Contains the MIT-BIH data and the links to the additional required data (Too big for GitHub)
- Utilities --> Contains helper functions provided for the Physionet challenge
- Slides    --> Contains slides for the in-class presentation
- Images    --> Contains data visualizations and other generated useful representations and results

# Overview of Problem

# Goals
Explore the following types of models for Arrythmia Classification using Deep Neural Networks
- Detector Scale Algorithm(CNN) : Simple architecture, basic detection of irregularities.
- Diagnosis Scale Algorithm(ResNet) : More classes, but utilizes a more complex architecture in addition to having more parameters.

# Example of Input file (12-lead ECG)
Traditional 12-lead ECG Plot
![alt text](https://github.com/varocarras/ECG-523/blob/main/Images/ECG-Sample.png?raw=true)
Alternate 12-lead ECG Plot(sample# vs. ADC count)
![alt text](https://github.com/varocarras/ECG-523/blob/main/Images/12lead.png?raw=true)

# Datasets
  - MIT-BIH Arrhythmia dataset <-- Used for Detector Model

  - Physionet Challenge 2020 (SNOMED Mapping) <-- Used for Diagnosis Model
    - China 12-Lead ECG Challenge Database
    - China Physiological Signal Challenge in 2018
    - Georgia 12-Lead ECG Challenge Database
    - PTB Diagnosis ECG Database
    - PTB-XL electrocardiography Database
    - St Petersburg INCART 12-lead Arrhythmia Dataset
   
    Total of 43101 Recordings (See distribution of classes in data below)
    ![alt text](https://github.com/varocarras/ECG-523/blob/main/Images/Arrhythmia-Distribution.png?raw=true)

  
# Results
- CNN Model (Detection)
  - F1 Score --> 
  - Number of Parameters --> ~ 15,000
  - Confusion Matrix
![alt text](https://github.com/varocarras/ECG-523/blob/main/Images/confdet.png?raw=true)

### List of Classes in Confusion Matrix 

- ResNet Model (Diagnosis)
  - F1 Score --> F1 
  - Number of Parameters --> ~500,000
  - Confusion Matrix
![alt text](https://github.com/varocarras/ECG-523/blob/main/Images/RESNET-ConfusionM.png?raw=true)

### List of Classes in Confusion Matrix 
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

# Link to Slides
- ADD LINK HERE

# Demo Code

- [DEMO link](https://github.com/varocarras/ECG-523/blob/main/DEMO.ipynb)
- The demo code allows for training the model as well as loading its weights to predict a rhythm for any given waveform.
- Run the script to generate visualizations and show an example of predicting a rhythm for a particular patient. 
- In order to retrain the model:
  - Run train.py (root folder file) or uncomment the training code from the Demo.ipynb
  - Weights will save to /Models/resnet_model.h5 and will load from there to test
  - We left the weights there in case you dont want to train the model again

# References

1. B. -J. Singstad and C. Tronstad, "Convolutional Neural Network and Rule-Based Algorithms for Classifying 12-lead ECGs," 2020 Computing in Cardiology, 2020, pp. 1-4, doi: 10.22489/CinC.2020.227. --> (https://ieeexplore.ieee.org/document/9344421)

2. Cardiologist Level Aryhthmia Detection with Convolutional Neural Networks -->
(https://arxiv.org/abs/1707.01836)

3. MIT-BIH Data in CSV format --> (https://www.kaggle.com/shayanfazeli/heartbeat)
