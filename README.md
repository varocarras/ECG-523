#  Adapted from --> https://www.kaggle.com/bjoernjostein/physionet-challenge-2020/data


# Classification of Arrhythmia from Raw ECG using Deep Neural Network for Detection and Diagnosis

## TO-DO
- Copy to SCC
- Create train.py module
- Generate new results (comparisons with new data)
- Edit README
- Train on Stanford's Model

# Overview of Included Files

models folder --> Stores the model weights
data --> contains only the MIT-BIH dataset and instructions on how to get the other datasets


# Datasets (Training)
  - MIT-BIH
    - 
    
  - Physionet Challange 2020 (SNOMED Mapping)
    - 
  - PTB Diagnosis ECG Database

# Datasets (Test)
- Irythm 

# Results
  - Metics Used
    - F1 Score --> F1 
![alt text](https://github.com/varocarras/ECG-523/blob/main/Images/Arrhythmia-Distribution.png?raw=true)

![alt text](https://github.com/varocarras/ECG-523/blob/main/Images/RESNET-ConfusionM.png?raw=true)




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
