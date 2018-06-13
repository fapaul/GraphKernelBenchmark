# GraphKernelBenchmark
Evaluation and comparison tool for different graph kernels


# Installation
```
conda env create -f environment.yml
source activate graph_kernels

```

# Setup

Navigate to `/kernels/MLG/MLGkernel`
and edit path to Eigen in `Makefile.options` (even if it is included globally, it could be in `usr/local/include` or in `usr/include`)
```
make all
```

Navigate to `/kernels/glocalwl` and use
```
cmake ./
make
```

Change name of `config.py.example` to `config.py` and choose kernels to compare.

# Run

Set `-d` parameter to folder in names `datasets` directory. Folder names need to be separated by a whitespace.

```
usage: evaluate.py [-h] -d DATA

Starting benchmark

optional arguments:
  -h, --help            show this help message and exit
  -d DATA, --data DATA  Bechmark dataset
```
