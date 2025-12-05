# Controllable Patching for Compute-Adaptive Surrogate Modeling of Partial Differential Equations

This repository contains the reference implementation for **Convolutional Stride Modulators (CSM)** and **Convolutional Kernel Modulators (CKM)**. The full preprint is available **[here](https://arxiv.org/abs/2507.09264)**, and a workshop version appears in ICLR 2025: Machine Learning for Multiscale Processes **[here](https://openreview.net/forum?id=YM3koX4nHp)**.

<p align="center">
  <img width="524" height="259" alt="CKM-CSM diagram" src="https://github.com/user-attachments/assets/d8174f5c-15c3-4036-bd89-75ac063d7664" />
</p>

CSM and CKM represent a class of general flexible striding and patching strategies which allow adaptive striding and patching in autoregressive vision-transformer–based PDE surrogates. Across a range of challenging 2D and 3D PDE benchmarks, CSM and CKM dramatically improve long-term stability in video-like prediction tasks. Their plug-and-play design makes them broadly applicable across architectures—establishing a general foundation for compute-adaptive modeling in PDE surrogate tasks.

Its effectiveness is demonstrated at scale in **[Walrus](https://arxiv.org/abs/2511.15684)**, where CSM forms the encoder/decoder backbone of the current state-of-the-art foundation model for continuum dynamics.

---

## Installation

Clone the repository and install locally. Requirements are defined in `pyproject.toml`.  
Most examples assume access to **The Well** dataset collection.

Ensure Python **3.10+** is used.

```bash
# Clone the repository
git clone https://github.com/payelmuk150/patch-modulator.git
cd patch-modulator
pip install .

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate

# Install The Well dataset library
git clone https://github.com/PolymathicAI/the_well.git
pip install -e ./the_well
```

---

## Repository Structure

```text
patch-modulator/
├── controllable_patching_striding/
│   ├── configs/                  # Hydra configs for data, model, optimizer, trainer
│   ├── data/                     # Data loaders + Well → model formatters
│   ├── models/                   # Encoders, decoders, isotropic model, CKM/CSM code
│   ├── optim/                    # Optimizers and schedulers
│   ├── trainer/                  # FSDP-aware training loop and utilities
│   ├── utils/                    # Experiment setup + distributed utilities
│   ├── train.py                  # Main Hydra training entry point
│   ├── ckm_tr2d_100M_example_script.sh  # Example CKM job script
│── ├── csm_tr2d_100M_example_script.sh  # Example CSM job script     
├── pyproject.toml
├── README.md
└── tests/
```

---

## Running Example Training Jobs (SLURM)

The repository includes example SLURM submission scripts.

### CKM Training
```bash
sbatch controllable_patching_striding/ckm_tr2d_100M_example_script.sh
```

### CSM Training
```bash
sbatch controllable_patching_striding/csm_tr2d_100M_example_script.sh
```

The scripts automatically:
- load modules
- activate the environment
- configure Hydra experiments
- run `train.py` with appropriate overrides

---

## Contact

- **Issues:** Please open a GitHub Issue in this repository.  
- **Other inquiries:** pm858@cam.ac.uk

---

## Acknowledgements

The authors would like to thank **Geraud Krawezik** and the **Scientific Computing Core** at the Flatiron Institute, a division of the Simons Foundation, for compute support.

We also acknowledge support from:
- **Simons Foundation**
- **Schmidt Sciences, LLC**
- **Polymathic AI**

---

## Citing Controllable Patching

If you use CKM/CSM or this repository, please cite:

```bibtex
@misc{mukhopadhyay2025controllablepatchingcomputeadaptivesurrogate,
      title={Controllable Patching for Compute-Adaptive Surrogate Modeling of Partial Differential Equations},
      author={Payel Mukhopadhyay and Michael McCabe and Ruben Ohana and Miles Cranmer},
      year={2025},
      eprint={2507.09264},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2507.09264},
}
```
