# Models
- Mk 1 - baseline simple model
    - Architecture:
      - ANN, multiple layers
    - Data
      - 5 DOF simulation data.
      - Forces and positions (newtons and metres)
      - 100 Radii, time points for each 10000. 
        - Uniformly distributed between 0.1 - 1 microns
      - First 100 points discarded (particle falling into trap)
        - Could also possibly centre particle at eq.
    - Output:
      - Continuous range of radii from 0.1 to 1 micron.

# Mk 2 - ResNet Model
Based off of https://arxiv.org/pdf/1809.04356.pdf.

## Architecture:
A ResNet deep learning network for univariate time series.
### Convolutional Layers
Based on 3 convolution bloacks with the number of filters for the convolutions in each block being respectively: (64, 128, 128).

Conv Block Structure:
Conv Layer ker=8 -> Normalisation -> ReLU -> Conv Layer ker=5 -> Normalisation -> ReLU -> Conv Layer ker=3 -> Normalisation -> ReLU

Each block has a skip connection from the block's input to the output. Skip connection consists of a kernel size 1 convolution then batch normalisation. 

505,290 trainable parameters

### Output: 
Global Average Pooling Layer -> Softmax
### Optimisation: 
Adam wtih learning rate: 0.001.
### Loss:
Catergorical Cross Entropy

## Data:
  
1000 Univariate force time series with 1000 time points each and varying radii. Radii values are spaced by 0.2 um from 0.2 to 1 um and randomly sampled. Radii encoded into a one hot vector.

Force values are z normalised.


## Expansion
Inclusion of more force axis.
Pretrained network for regression based network
