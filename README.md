# QReR

The running codes for "Quasi-Rerandomization for Observational Studies".

## Notebooks (`.ipynb`):

- `Generate_SimuDatasets_3Scenario`: Generate simulated datasets for comparisons.
- `QReR_Simulation_ReR-SATE`: Simulated studies to compare QReR and ReR for $\tau_{\rm SATE}$ estimation.
- `Benchmark_Simulation-PATE`: Simulated studies to compare different benchmark methods for $\tau_{\rm PATE}$ estimation.
- `QReR_Simulation-PATE`: Simulated studies to evaluate QReR for estimating $\tau_{\rm PATE}$. 
- `QReR_RealData_IHDP_SATE&PATE(100)`: Perform real data analysis over IHDP datasets.


## Scripts (`.py`):
- `benchmarks`: Wrappers to implement `IPW`,`FM`,`EBAL`,`SBW` and `EBCW`.
- `datagen`: Functions to generate simulated datasets.
- `network`: The network structure of QReR.

## Real Datasets (IHDP):

The real datasets (IHDP-100 (train), IHDP-100 (test)) can be found in folder `realdata`, which are downloaded from https://www.fredjo.com/.
