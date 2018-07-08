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

# Run

Set `-d` parameter to folder names in `datasets` directory. In `config.py` you can find all currently supported kernels with their keys. 

```
usage: evaluate.py [-h] -d DATA [DATA ...] -k KERNELS [KERNELS ...]

Starting benchmark

optional arguments:
  -h, --help            show this help message and exit
  -d DATA [DATA ...], --data DATA [DATA ...]
                        Benchmark datasets
  -k KERNELS [KERNELS ...], --kernels KERNELS [KERNELS ...]
                        Benchmark kernels
```
