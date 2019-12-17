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
    