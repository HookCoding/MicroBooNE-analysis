# MicroBooNE Neutrino Analysis — Neural Network Extension & 3+1 Sterile Neutrino Framework

Extension to the core MicroBooNE muon neutrino oscillation analysis. This notebook replaces the scikit-learn Random Forest classifier with a neural network built entirely from scratch, and extends the oscillation analysis to the 3+1 sterile neutrino framework for comparison against published LSND and MiniBooNE results.

---

## What this notebook adds

### 1. Neural Network Classifier (from scratch)

A binary classifier implemented without any high-level ML libraries. The network identifies νμ CC signal events against background (cosmic EXT, ν NC, out-of-fiducial-volume, mis-ID).

**Architecture**
- Configurable input size, hidden layer widths, and output size
- Leaky ReLU activations (configurable α)
- L2 regularisation (λ)
- Weighted binary cross-entropy loss with separate positive/negative class weights to handle class imbalance

**Training**
- BFGS optimisation via `scipy.optimize.minimize`
- Numerical gradient checking included for validation
- Training and test loss tracked per epoch for overfitting diagnostics

**Input features**
Features are normalised by the training set maximum before being passed to the network:
```
_closestNuCosmicDist, trk_len_v, trk_distance_v, topological_score,
trk_score_v, trk_llr_pid_score_v, trk_energy_tot
```

**Output**
- Prediction histogram separating signal (category 21) from background
- Configurable classification threshold (default 0.75)
- Neural network predictions used to apply an additional selection cut on top of the standard variable cuts

---

### 2. Oscillation Parameter Scan

Two-flavour muon neutrino disappearance probability applied as a per-event weight to Monte Carlo:

```
P(νμ → νβ) = 1 − sin²(2θ) · sin²(1.27 · Δm² · L / E)
```

where L = 0.47 km and E is the true neutrino energy in GeV.

A χ² statistic comparing oscillation-weighted MC against real MicroBooNE data:

```
χ² = Σ (μᵢ(θ) − Mᵢ)² / σᵢ²
```

with a 15% flat systematic uncertainty per bin.

**Closure test** — the pipeline is validated by recovering known oscillation parameters from a pre-oscillated MC dataset (`oscillated_data.pkl`) before applying to real data.

**Full scan** — 500×500 log-spaced grid over sin²(2θ) ∈ [10⁻³, 1] and Δm² ∈ [10⁻¹, 10²]. Best-fit parameters extracted from the χ² minimum; 1σ, 2σ, and 3σ confidence contours drawn from the Δχ² surface (Δχ² = 4.61, 5.99, 9.21 for 2 degrees of freedom).

---

### 3. 3+1 Sterile Neutrino Framework

The two-flavour result is reinterpreted in the 3+1 model, which introduces a fourth sterile neutrino eigenstate. The oscillation parameter sin²(2θ) is rescaled to the νμ → νe appearance amplitude sin²(2θμe) using:

```
sin²(2θμe) = 4|Uμ4|²|Ue4|²
```

with sin²(2θ₁₄) = sin²(2θee) = 0.24 (from MicroBooNE published results). Confidence contours are overlaid against the allowed regions from LSND and MiniBooNE.

---

## Requirements

```
numpy
pandas
matplotlib
scipy
scikit-learn
seaborn
jupyter
```

Install with:
```bash
pip install numpy pandas matplotlib scipy scikit-learn seaborn jupyter
```

---

## Usage

This notebook depends on outputs from the core analysis (`MainAnalysis.ipynb`), specifically the selection cuts and the `MC_EXT_CUT` and `data_frame_cut` dataframes. Run the core notebook first, then open this one and run cells sequentially.

Data files required in `./data/`:
```
MC_EXT_flattened.pkl
data_flattened.pkl
oscillated_data.pkl
DataSet_LSND.csv
DataSet_MiniBooNE.csv
```
