This is the repository for our paper Efficient Learning of Deep State Space Models via Importance Smoothing.

Important: all code must be run with the top level directory i.e. /Parallel variational sequential monte carlo/ as the working directory.

We have exported our python environment as a requirements.txt file which you may install by

```
pip install -r requirements.txt
```

but it is strongly recommended that you install the version of PyTorch appropriate to your compute platform by the command given on https://pytorch.org/get-started/locally/.

Both the linear gaussian and lokta-volterra experiments use generated data, so make sure to run the data generation scripts first!


Algorithms are implemented at in the top-level directory.

The algorithms we implement here are:
diffusion DPF (diffusion_DPF.py)
d-SMC (dSMC.py)
Deep markov models (dmm.py)
Kalman Filter (parallel_kalman.py)
RTS Smoother (parallel_kalman.py)
parallel variation Monte Carlo (parallel_smoother_new.py)
time causal VAE (time_causal_VAE.py)
two filter smoother (two_filter_smoother.py)
mixture density particle smoother (mdps.py)

The differentiable particle filters, excluding diffusion dpf, are due to the pypi package pydpf.

Parallel scans and reductions are implemented in parallel_scan.py .

Code to run all the experiments and generate all the figures are housed in /experiments/
The models are defined programmatically in /models/