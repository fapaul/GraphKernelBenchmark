kernels = ['MLG', "WL3L", "WL3G", "WL2G", "ColorRefinement", "Graphlet", "Propagation", "VertexHistogram", "PyramidMatch", "WeisfeilerLehman", "ShortestPath", "LovaszTheta", "SVMTheta", "NeighborhoodHash", "OddSth", "EdgeHistogram"]
datasets = ["MUTAG", "ENZYMES", "EXP1", "EXP2", "EXP3"]

with open('parallel_config.txt', 'w+') as f:
	for dataset in datasets:
		for kernel in kernels:
			f.write("python evaluate.py -d " + dataset + " -k " + kernel + "\n")
