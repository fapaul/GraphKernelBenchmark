# GraphKernelBenchmark
Evaluation and comparison tool for different graph kernels


# Installation
```
conda env create -f environment.yml
source activate graph_kernels

```

Navigate to /kernels/MLG/MLGkernel
Edit path to Eigen in Makefile.options (even if it is included globally, it could be in usr/local/include or in usr/include)
```
make all
```

Navigate to /kernesl/glocalwl and use
```
cmake ./
make
```