# Deep Learning For Classification of Particles in Optical Tweezers

The ultimate goal of the project is to develop a neural network which can take force and position data from a particle trapped optical tweezers and predict the radius and refractive index of the particle. This seems to be a complicated problem.

We already have a neural network that can take in positions, radii and refractive index and generate forces. This will be used to simulate the data which will be used to train the machine learning models. 

## Classification Sub-Problem 
As a first step it will be attempted to classify time series of length 1000 into 5 classes of radii evenly spaced from 0.2 $\mu m$ to 1 $\mu m$. This problem will serve as a proof of concept as well as a base to build on for extending the model. The weights that are computed can be used as a base for a regression problem or for a base for classification of refractive index.

### ResNet Model
It was decided from a review of time series classification literature (in particular using deep learning) [1] to use a ResNet classification Architecture. The architecture takes the highly successful image classification and modifies it to be used on time series data. The ResNet Architecture has several advantages:
- Deep network that can avoid the vanishing gradient problem.
- The best performance on the UCR Time Series Classifcation Dataset [1] among other leading time series classification architecture.
- Highly transferable: can use trained weights as initialisation for regression model or for different length time series.


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

# References
1. Deep learning for time series classification: a review (2019): https://arxiv.org/pdf/1809.04356.pdf