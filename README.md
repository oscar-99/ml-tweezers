# Deep Learning For Classification of Particles in Optical Tweezers

The ultimate goal of the project is to develop a neural network which can take force and position data from a particle trapped optical tweezers and predict the radius and refractive index of the particle. 

## The Data 
A neural network which has been trained to predict the forces on a particle given its position, radius and refractive index was used to simulate a spherical particle in an optical trap to generate training data. The simulation uses a time step of 10e-4 and generates the forces and positions for 1100 steps, the first 100 steps being discarded due to these points corresponding to the particle falling into the trap (hence only data where the particle was settled in the trap was used). Therefore each time series is 1000 point long correponding to 0.1 seconds in the trap.

- Time series simulation and length 
    - Longer time series 
    - Information caputed

- Possibility of obtaining some experimentally generated data.
    - Pre training

## Classification Problem 
As a first step it will be attempted to classify time series of length 1000 into n (so far 5 and 10) classes of radii evenly spaced from 0.2 $\mu m$ to 1 $\mu m$. This problem will serve as a proof of concept as well as a base to build on for extending the model. The weights that are computed can be used as a base for a regression problem or for a base for classification of refractive index.

## Regression Problem
Regression replaces the discretised classes from the classification problem with a continuously varying output space. This is a more general and realistic problem as on any practical application of the network there will not be the same set of 5 or 10 radii.
- How is it different to the classification problem
- Challenges

## ResNet Model
It was decided from a review of time series classification literature (in particular using deep learning) [1] to use a ResNet classification Architecture. The architecture takes the highly successful image classification and modifies it to be used on time series data. The ResNet Architecture has several advantages:
- Deep network that can avoid the vanishing gradient problem.
- The best performance on the UCR Time Series Classifcation Dataset [1] among other leading time series classification architecture.
- Highly transferable: can use trained weights as initialisation for regression model or for pretraining an experimentally generated dataset.
- Can generate as much data as needed which suits a deep learning approach


### Performance
The network was trained on a GPU which significantly sped up the process a

![1000 Epoch Training Diagnostics](/Figures/Diagnostics-1000epoch.png)


# Summary 
- Got simulation working to generate some data using trained 5 degree of freedom model 
- Reading up on Neural Networks particularly convolutional neural networks and image classification.  
- Built a simple MLP model to help learn how keras works and how to process the generated data.
- Shifted reading focus to time series classification/regression.
- Reading leads to ResNet model and Deep learning. Begin process of building and training ResNet.
- Ran on GPU for huge speedup
- Running ResNet results in overfitting, more data is added.
    - Adding more data is not the most satisfying solution to overfitting 
- Why ResNet?
- Transfer learning from the 5 class to the ten resulted in 98.5% (best) validation accuracy within 20 epochs. 
    - Increased number of training points 
- Close to 0.1 starts getting dispersed positions.
- Changing radius appears to alter the equilibrium position.

# References
1. Deep learning for time series classification: a review (2019): https://arxiv.org/pdf/1809.04356.pdf


# Progress
## 15/1/2020
- So far:
    - ResNet 5 and 10 class classifier working to 99% accuracy on the testing set.
    - Trained 1000 epoch model for the 5 class classifier.
    - Regression ResNet coded and tested on low epoch runs (100-200) including 10000 examples, seeing 10-15% MAPE and a gap between training and testing set. Probable overfitting.
- To do:
    - Tighten up the possible range for regression model.
    - Figure out the overfitting problem
        - y force axis, linear vs. circular polarisation and information correlation between x and y.
    - Tighten up simulation, aim to get a longer time window. Probably go with longer simulation and sampling a portion of the points.
      - Compare this with previous results
    - Heatmap or analysis of point that tend not to classified/regressed correctly.
    - Code for analysis of simulated data (eg. position plots, histograms etc.)
 
 ## 22/1/2020

- So far:
  - Spent last few days days tightening up the simulation
    - Add in the capacity for generating data with varying refractive index.
    - The simulation now stores values in the format (simulations, time series length, axes) and saves them as they go.
    - Add capacity for down-sampling the number of points generated to save on storage space as well as the length of time series passed to the model
  - Built functions for visualizing and summarizing the data generated.
- To do:
  - Analyse the advantage of the longer simulation window.
  - Work on y axis and polarisation stuff. 
  - Extend regression