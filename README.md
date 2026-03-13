# Deconstructing Taxi-V3 with Maximum Entropy Inverse Reinforcement Learning

This project implements the **Maximum Entropy Inverse Reinforcement Learning (MaxEnt IRL)** algorithm from scratch to recover the underlying reward function of an expert navigating the Gymnasium `Taxi-v3` environment.

Unlike standard RL where the reward is known, this project reverse-engineers agent behavior to find a reward function that makes expert demonstrations appear near-optimal. Key focus areas include algorithmic stability via **L2 Regularization** and robustness testing against **suboptimal (noisy) experts**.

<video autoplay loop muted playsinline width="100%">
  <source src="Demo.webm" type="video/webm">
</video>

---

### Highlights

- **92% success rate** using L2 Regularization (λ=0.05), vs. 48% for the unregularized baseline
- Trained on a limited dataset of just 200 expert trajectories
- Achieved 82% success rate even when trained on demonstrations with a 20% error rate (ε=0.2), showing the probabilistic nature of MaxEnt IRL naturally filters inconsistent behavior

---

### Repository Structure

- `src/synthex.py` — Generates ground-truth optimal policy via Value Iteration; records expert demonstrations using a handcrafted 5D feature vector
- `src/MaxEnt.py` — Baseline MaxEnt IRL with forward-backward passes and LogSumExp trick for numerical stability
- `src/L2RegMaxEnt.py` — Improved implementation with L2 penalty on gradient updates to prevent weight explosion and overfitting
- `src/SubOpt_MaxEnt.py` — Robustness evaluation by injecting ε-greedy noise into expert demonstrations
- `docs/final_report.pdf` — Full write-up covering mathematical formulation, convergence analysis, and results

---

### Methodology

**Feature Engineering**

States are mapped to a 5-dimensional feature space (d=5):
1. Successful dropoff
2. Successful pickup
3. Negative distance to passenger (Manhattan)
4. Negative distance to destination (Manhattan)
5. Step cost (efficiency penalty)

**Optimization Objective**

The standard MaxEnt objective maximizes the likelihood of expert trajectories. L2 Regularization is applied to ensure convergence and prevent deterministic overfitting:

$$\nabla J(w) = (\tilde{\phi}_{expert} - \sum_{\tau} P(\tau|w)\phi(\tau)) - \lambda w$$

Where λ=0.05 penalizes large weights, pushing the algorithm toward a simpler, more generalizable reward function.

---

### Results

Evaluated over 50 test episodes:

| Method | Success Rate | Policy Disagreement | Convergence |
| :--- | :---: | :---: | :---: |
| Baseline (Unregularized) | 48% | 74 states | Unstable |
| L2 Regularized (λ=0.05) | **92%** | **32 states** | **Stable** |
| Suboptimal Expert (20% Noise) | 82% | 68 states | Stable |

The unregularized baseline failed to converge in 26/50 episodes, frequently entering infinite loops due to extreme reward weight assignments. L2 Regularization smoothed the optimization landscape and resolved this.

---

### Installation & Usage
```bash
pip install gymnasium numpy
```

To train the regularized model and visualize the recovered policy:
```bash
python src/L2RegMaxEnt.py
```

---

### Contributors

Developed as part of the graduate curriculum at Carnegie Mellon University Africa.

- **Dev Rawal** — Synthetic expert implementation (Phase 1), feature engineering pipeline, core MaxEnt refinement, L2 Regularization, and final code refactoring
- **Peter Adeyemo** — Initial MaxEnt IRL baseline, suboptimal expert robustness investigation, experiment execution, and data interpretation for the final report
