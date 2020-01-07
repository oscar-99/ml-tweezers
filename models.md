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
- Based on 3 convolution bloacks with the number of filters for the convolutions in each block being respectively: (64, 128, 128).

- Conv Block Structure:
Conv Layer ker=8 -> Normalisation -> ReLU -> Conv Layer ker=5 -> Normalisation -> ReLU -> Conv Layer ker=3 -> Normalisation -> ReLU

- Each block has a skip connection from the block's input to the output. Skip connection consists of a kernel size 1 convolution then batch normalisation. 

- ~500,000 trainable parameters

### Output: 
- Global Average Pooling Layer -> Softmax

### Optimisation: 
- Adam wtih learning rate: 0.001.

### Loss:
- Catergorical Cross Entropy

## Data:
  
- 1000 Univariate force time series with 1000 time points each and varying radii. Radii values are spaced by 0.2 um from 0.2 to 1 um and randomly sampled. Radii encoded into a one hot vector.

- Force values are z normalised.

- 0.9/0.1 Training/Testing Split

## Notes:
- Fitting model for ~40 epochs resulted in very high training accuracy and very low and volatile validation accuracy. Possibly caused by overfitting in the model. 
- Coding mistake, connected the first blocks short cut to the second shortcut directly. Fixed.

## Modifications:
To fix the overfitting problem 
- Increase Size of dataset
  - Examples increased to 2000
    - training time up to ~2 min/epoch
    - validation accuracy and loss seem to be tracking with traing (lagging a bit) before getting bounced to 0.5.
  - Further increase to 3000.
    - Training time up to ~3 min/epoch
    - Validation accuracy still volatile but still between ~0.5-0.75, below 20 epoch training accuracy.
    - More time training needed?
- Hyper parameters
  - Dropped learning rate to 1e-5 which didn't seem to make much of a difference.
- Add in some dropout/other types of regularisation.
- Ensure architecture is correct.
- Incorporate another force axis
  - Increase training time?
  - Greater Complexity, dimensionality increased.
  - More information available, works to fix the overtraining problem.
- Pre process method
  - Best type of normalisation?
    -  Do I lose information with z normalisation?
    - Maybe some kind of max divide method?
  - Truncation of first 100 points.
    - Start from random positions centred at the trap equilibrium 
      - Capture some of the away from equilibrium behavior. This may boost the information available to the model.
      - Generalisability. 
- Run the model on a UCR dataset to verify the architecture is functioning correctly
- Shuffle the order of the time series.



