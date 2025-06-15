# Quantum Dynamical Decoupling Toolkit

A Julia-based simulation framework for analyzing the performance of dynamical decoupling (DD) sequences on single-qubit systems under realistic, multiaxial classical noise.

---

## Features

- Modular design using Julia's package system
- Simulation of qubit fidelity under various DD sequences:
  - FID, CPMG, UDD, CDD, PDD, and XY-family
- Support for classical stationary noise models:
  - White noise, 1/f noise, Ornstein-Uhlenbeck (OU), and option for user to define their own noise model
- Filter function formalism to analyze and visualize decoherence
- Stretched exponential fitting for \( T_{2,\text{eff}} \) and decay profiles
- Clean visualization via `Plots.jl`
- Optional noise spectroscopy framework using inverse filter function techniques

---

## Repository Structure

```
Quantum-Dynamical-Decoupling-Toolkit/
├── src/                     # Core source files (pulse definitions, control logic)
├── notebooks/               # Jupyter notebooks demonstrating simulations
├── data/                    # Intermediate data outputs
├── plots/                   # Generated figures and visualizations
├── Manifest.toml            # Julia package environment (locked)
├── Project.toml             # Julia project definition
```

---

## Getting Started

### Requirements

- Julia ≥ 1.8
- QuantumToolbox.jl
- Plots.jl, FFTW.jl, LaTeXStrings.jl, etc.

### Installation

Clone this repository and instantiate the environment:

```bash
git clone https://github.com/VikramD2k/Quantum-Dynamical-Decoupling-Toolkit.git
cd Quantum-Dynamical-Decoupling-Toolkit
julia --project -e 'using Pkg; Pkg.instantiate()'
```

### Running a Simulation

Explore the pre-built notebooks in `notebooks/`.

---

## Outputs

Plots and figures generated during simulations are stored in the `plots/` directory. These visualize coherence decay, filter functions, power spectra, and DD performance under various noise models.

---

## Author

**Dorbala Trivikramahimamshu Shekhar**  
M.Tech in Quantum Technology, IISc Bangalore  
https://github.com/VikramD2k

---

## License

This repository is currently **private**. All rights reserved.

---

## Notes

- All simulations are based on time-domain sampled noise generated from analytic PSDs.
- The current toolkit is focused on *single-qubit* systems but can be extended.
- This work is part of an academic project report submitted to IISc, 2025.

---

## Acknowledgements

Special thanks to Prof. Baladitya Suri and the Quantum Technology group at IISc for the guidance and mentorship.
