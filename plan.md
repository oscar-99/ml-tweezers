# Plan 

## Generate data
- Run simulations for different refractive indices/sizes
    - OTT or Neural Net simulations?
      - Amount of data needed?
      - How many refractive indices?
        - Chosen arbitrarily or specific values?
        - Range of values?
      - Particle sizes
      - Verification of simulation
        - Level of accuracy? Cascading error?
        - T matrix method

## Features 
- Forces or forces and positions? (other variables?)
  - Position sampling is far lower frequency: probably not useful in final model.
  - Just forces will need time series structure.   Probably a novel move.
  - Which forces (radial, axial etc.)
  - Normalisation of forces. z-normalisation?
- Force histograms or frequency plots
  - Features of these varying in size/refractive index
  - Predicting refractive index and radius together 
  - Histogram gives some amount of information about the how the forces are distributed. Which can vary between size/refractive index.
  - https://www.tensorflow.org/tutorials/structured_data/feature_columns#bucketized_columns
- Time ordered?
  - Accuracy benefits, training time, memory?
  - What information does time ordering capture?
    - How much past information is valuable?
  - Data generation
  - 
## Outputs
- Refractive index
  - Other variables (particle size?)
- How do refractive indices vary?
  - Do only specific values matter corresponding to materials?
- Varying in time refractive index?


# Network Architecture 
- Type of network?
  - LSTM, other RNN?
- Layers, nodes, connectivity?
- More modern Architecture?


# Measurement of the Index of Refraction of Single Microparticles

- Measurement of refractive index difficult
- Model trap stiffness with respect to a range refractive indices. Measure actual trap stiffness to derive refractive index
- Transvers stiffness is used, axial force is minimised
- Trap stiffness $\alpha_x$ is given by Q (trapping efficiency) multiplied by $n_{med}P/c$. $P$ is power of the 
# Measurements of trapping efficiency and stiffness in optical tweezers

- Q is efficiency of trap, dimensionless

# https://www.pdx.edu/nanogroup/sites/www.pdx.edu.nanogroup/files/Constructing%20An%20Optimized%20Optical%20Tweezers.pdf

# Tasks
- Learn some linux to get getafix working
- <s>Fix 5 DOF </s>
  - <s>Generate some data</s>
- <s>Force histogram script</s>
- <s>Store simulated data so I dont have to keep simulating </s>
- Fourier transform?
- Build some test models.
- Generate large dataset on getafix.
- Research time structured architecture.
- Generate some time series plots of forces
- Generate some histograms of forces
- Run some time series analysis
- Generate some data for changing 

