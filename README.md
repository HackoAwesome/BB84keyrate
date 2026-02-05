# Entanglement-Based BB84 Key Rate Analyzer

The aim of this project is to create a tool to perform numerical analysis on the **key rate of an Entanglement-Based BB84 protocol**. The tool is based on the theoretical bounds proposed in [arXiv:2405.05912](https://arxiv.org/pdf/2405.05912).

## Features

- Computes **finite-size key rate bounds** for EB-BB84.
- Performs **convex optimization** to evaluate the bounds using cvxpy.
- Utilises **piecewise-linear and coarse secant approximations** for non-atomic functions on cvxpy variables. 
- Plots **key rate vs. number of signals** for visual analysis.
- Allows flexible input parameters such as detection efficiency, error rate, and security parameters.

## Requirements

- Python 3.8+
- `numpy`
- `cvxpy`
- `matplotlib`

```bash
pip install numpy cvxpy matplotlib
