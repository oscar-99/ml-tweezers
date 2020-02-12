- cont-data-nr-01-1-old: 
  - Accidentally used a fixed radius to compute gamma. Need to discard 
  - r = (0.4,0.6), n = (1.5, 1.7), 10000 at 40x40 
  - r = (0.4,0.6), n = (1.5, 1.7), 5000 at 20 x 20
- cont-data-nr-01-1-15000:
  - r = (0.4,0.6), n = (1.5, 1.7), 15000 at 40x40
  
- resnet3-nr-regression-xyz-100E trained on cont-data-nr-01-1-15000

- cont-data-nr-01-1-20000: 
  - r = (0.4,0.6), n = (1.5, 1.7), 15000 at 40x40 
  - r = (0.465, 0.500), n = (1.530, 1.560), 1000 at 6x7
  - r = (0.465, 0.500), n = (1.545, 1.575), 1000 at 6x6
  - r = (0.560, 0.6), n = (1.625, 1.675), 1000 at 10x8
  - r = (0.4, 0.6), n = (1.5, 1.525), 2000 at 5x40

- cont-data-nr-01-1-15000:
  - r = (0.4,0.6), n = (1.5, 1.7), 30000 at 40x40

- resnet3-nr-regression-xyz-100E
  - 100 epoch of training.

- resnet3-nr-regression-xyz-200E
  - 200 epochs of training, 100 from the 15000 point dataset and 100 from the 20000 point dataset.

- resnet3-nr-regression-xyz-2-100E
  - 100 epochs started fresh with the 20000 point dataset.

- resnet3-nr-regression-xyz-3
  - 500 epochs training on the 15000 point dataset