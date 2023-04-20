# QReR

The running codes for "Quasi-Rerandomization for Observational Studies".

## Notebooks (`.ipynb`):

- `Generate_SimuDatasets_3Scenario`: Generate simulated datasets for comparisons.
- `QReR_Simulation_ReR-SATE`: Simulated studies to compare QReR and ReR for $\tau_{\rm SATE}$ estimation. (Tables 1-3)
- `Benchmark_Simulation-PATE`: Simulated studies to compare different benchmark methods for $\tau_{\rm PATE}$ estimation. (Table 4)
- `QReR_Simulation-PATE`: Simulated studies to evaluate QReR for estimating $\tau_{\rm PATE}$. (Table 4)
- `QReR_RealData_IHDP_SATE&PATE(100)`: Perform real data analysis over IHDP datasets. (Figure 2 & Tables 5-6)
- `QReR_Demo`: Visually balance covariates based on QReR and ReR. (Figure 1)


## Scripts (`.py`):
- `benchmarks`: Wrappers to implement `IPW`,`FM`,`EBAL`,`SBW` and `EBCW`.
- `datagen`: Functions to generate simulated datasets.
- `network`: The network structure of QReR.

## Real Datasets (IHDP):

- The real datasets ([IHDP-100 (train)](https://www.fredjo.com/files/ihdp_npci_1-100.train.npz), [IHDP-100 (test)](https://www.fredjo.com/files/ihdp_npci_1-100.test.npz)) can be found in folder `realdata`, which are available at https://www.fredjo.com/.
- The data descriptions can be found at Section 5.1 of the paper ["Estimating individual treatment effect: generalization bounds and algorithms"](http://arxiv.org/abs/1606.03976).

