# Models
- mk 1 - baseline simple model
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
    
- Time structured data
  - https://www.tensorflow.org/tutorials/structured_data/time_series
  - Time delay neural network 
  - https://en.wikipedia.org/wiki/Time_delay_neural_network
  - https://stats.stackexchange.com/questions/137721/neural-network-classification-from-time-series
  - https://www.danielpovey.com/files/2015_interspeech_multisplice.pdf
  - http://www.cs.toronto.edu/~fritz/absps/waibelTDNN.pdf
  - https://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/
  - https://arxiv.org/pdf/1809.04356.pdf - review 
  - Equivalence with CNN?
  - Maybe start with predicting forces rather than radius
  - Model is shown snippets and then predicts?


https://www.manning.com/books/deep-learning-with-python