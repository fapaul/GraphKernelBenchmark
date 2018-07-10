from kernels import *


def get_kernel(kernel_class, params=None):
    if params is None:
        params = []

    def base_kernel(dataset_name, output_path, data_dir, workers):
        return kernel_class(dataset_name, output_path, data_dir, *params, workers)

    return base_kernel


KERNELS = {
    'MLG': get_kernel(MLGKernel),
    'WL3L': get_kernel(GlocalWLKernel, [[['-l', '1', '-i']], 'WL3L']),
    # 'WL3L': get_kernel(GlocalWLKernel, [[['-l', '2', '-i']], 'WL3L']),
    'WL3G': get_kernel(GlocalWLKernel, [[['-l', '1', '-i']], 'WL3G']),
    'WL2G': get_kernel(GlocalWLKernel, [[['-l', '1', '-i']], 'WL2G']),
    'ColorRefinement': get_kernel(GlocalWLKernel, [[['-l', '1', '-i']], 'ColorRefinement']),
    'Graphlet': get_kernel(GlocalWLKernel, [[['-l', '1', '-i']], 'Graphlet']),
    'Propagation': get_kernel(GrakelKernel, [[], 'Propagation']),
    'VertexHistogram': get_kernel(GrakelKernel, [[], 'VertexHistogram']),
    'PyramidMatch': get_kernel(GrakelKernel, [[], 'PyramidMatch']),
    'WeisfeilerLehman': get_kernel(GrakelKernel, [['subtree_wl'], 'WeisfeilerLehman']),
    'ShortestPath': get_kernel(GrakelKernel, [[], 'ShortestPath']),
    'LovaszTheta': get_kernel(GrakelKernel, [[], 'LovaszTheta']),
    'SVMTheta': get_kernel(GrakelKernel, [[], 'SVMTheta']),
    'NeighborhoodHash': get_kernel(GrakelKernel, [[], 'NeighborhoodHash']),
    'OddSth': get_kernel(GrakelKernel, [[], 'OddSth']),
    'EdgeHistogram': get_kernel(GrakelKernel, [[], 'EdgeHistogram']),

    # Very high run-time
    'RandomWalk': get_kernel(GrakelKernel, [[], 'RandomWalk']),
    'NeighborhoodSubgraphPairwiseDistance': get_kernel(GrakelKernel, [[], 'NeighborhoodSubgraphPairwiseDistance']),
    'SubgraphMatching': get_kernel(GrakelKernel, [[], 'SubgraphMatching']),
    'GraphletSampling': get_kernel(GrakelKernel, [[], 'GraphletSampling']),
    'WL2L': get_kernel(GlocalWLKernel, [[['-l', '1', '-i']], 'WL2L']),
}


def get_benchmarking_kernels(dataset_name, output_path, data_dir, kernel_names):
    workers = 1
    for name in kernel_names:
        yield KERNELS[name](dataset_name, output_path, data_dir, workers)
