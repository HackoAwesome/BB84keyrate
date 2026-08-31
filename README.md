# Evaluate Key Rates for QKD Protocols

## Overview

The aim of this project is to develop a numerical tool for evaluating the **key rates of different Quantum Key Distribution (QKD) protocols**.

The evaluation of the key rate for the **Entanglement-Based BB84 (EB-BB84)** QKD protocol is based on the theoretical bounds proposed in [arXiv:2405.05912](https://arxiv.org/abs/2405.05912), specifically Eq. (145).

Additionally, this project includes a tool for evaluating the key rate of the **Six-State QKD protocol**, based on theoretical bounds derived as part of this project.

## Features

* Computes **finite-size key-rate bounds** for EB-BB84.
* Performs **convex optimisation** to evaluate the key-rate bounds using `cvxpy`.
* Uses **piecewise-linear approximations** for non-atomic functions involving `cvxpy` variables.
* Plots **key rate versus number of signals** for visual analysis.
* Allows flexible input parameters, including:

  * Detection efficiency
  * Quantum Bit Error Rate (QBER)
  * Security parameters
  * Number of signals
  * Other protocol-specific parameters

## Requirements

* Python 3.8+
* `numpy`
* `scipy`
* `cvxpy`
* `matplotlib`
* MOSEK

The required Python packages can be installed using:

```bash
pip install numpy scipy cvxpy matplotlib
```

### MOSEK

The optimisation routines use the **MOSEK** solver. A MOSEK licence is therefore required to run the convex optimisation.

A trial/academic licence may be obtained from the [MOSEK website](https://www.mosek.com/).

## Usage

The repository contains different scripts for computing the key rates of different QKD protocols.

Each script contains the relevant protocol parameters and entropy expressions used in the key-rate calculation.

To evaluate the key rate for a particular protocol, run the corresponding Python script:

```bash
python Analysis/keyrate_computation.py
```

with the appropriate script file. The resulting key-rate data will then be plotted as a function of the number of signals.

## Adapting the Code to Other QKD Protocols

The numerical framework can also be adapted to evaluate key rates for other QKD protocols.

In particular, the entropy expression used in the optimisation can be modified through the `hterm()` function in the relevant script.

For example, if a different QKD protocol has a different entropy bound, the expression implemented in `hterm()` can be replaced with the corresponding theoretical expression.

This allows the same numerical optimisation framework to be used for investigating different QKD protocols, provided that the corresponding entropy bound can be expressed in a suitable form for the optimisation procedure.

## Repository Structure

```text
.
├── Analysis/
│   └── ...
├── Data/
│   └── ...
├── Data Visualisation/
│   └── ...
├── Drafts/
│   └── ...
├── Misc/
│   └── ...
├── scripts/
│   ├── BB84keyrate_improved.py/
│   ├── BB84keyrate.py/
│   ├── general_optimisation.py/
│   ├── sixstate_keyrate.py/
│   └── ...
├── README.md
```

## Output

The scripts produce numerical key-rate results for different numbers of signals. These results can be used to:

* Compare different theoretical bounds.
* Study finite-size effects.
* Investigate convergence towards the asymptotic key rate.
* Compare different optimisation approaches.
* Visualise the performance of different QKD protocols.

## References

The EB-BB84 key-rate calculation is based on the theoretical results presented in:

**A. Arqand et al., "Generalized Rényi entropy accumulation theorem and generalized quantum probability estimation" (2025).**

[arXiv:2405.05912](https://arxiv.org/abs/2405.05912)

The Six-State protocol implementation uses theoretical bounds derived as part of this project.

## Disclaimer

This repository is intended for **research and numerical analysis** of QKD key-rate bounds. The numerical results depend on the assumptions and theoretical bounds implemented in the corresponding scripts.

