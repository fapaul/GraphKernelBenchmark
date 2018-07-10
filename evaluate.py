import os
import json

import time
import argparse
import numpy as np
from scipy.stats import sem
from collections import defaultdict, namedtuple
from sklearn import svm
from sklearn.model_selection import ShuffleSplit, cross_val_score
from config import get_benchmarking_kernels


def timer(runnable):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = runnable(*args, **kwargs)
        return result, time.time() - start

    return wrapper


def read_label_matrix(line):
    return line.strip()


def read_kernel_matrix(line):
    return line.strip().split(' ')


def read_matrix(file_path, formatter):
    def read():
        with open(file_path, 'r') as f:
            for line in f:
                yield formatter(line)

    return np.array(list(read()))


def score_n_fold(train, test, n, c):
    cv = ShuffleSplit(n_splits=n, test_size=0.33)
    clf = svm.SVC(kernel='precomputed', C=c, class_weight='balanced')
    return cross_val_score(clf, train, test, cv=cv).mean(), c


@timer
def generate_kernels(kernel, run_number=0):
    return kernel.compute_kernel_matrices(run_number)


def evaluate(kernel, dataset_name, data_dir, number_of_runs=10):
    print('Running: ', kernel.kernel_name)
    kernel.compile()
    Result = namedtuple('Result', ['acc', 'stderr'])

    kernel.load_data()
    penalties = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    if kernel.is_deterministic():
        kernel_matrices_paths, run_time = generate_kernels(kernel)
        kernel_matrices_path = kernel_matrices_paths[0]
        label_path = os.path.join(data_dir, dataset_name,
                                  '{}_graph_labels.txt'.format(dataset_name))
        kernel_matrices = read_matrix(kernel_matrices_path, read_kernel_matrix)
        labels = read_matrix(label_path, read_label_matrix)
        scores = [score_n_fold(kernel_matrices, labels, 10, c) for c in
                  penalties]
        print(scores)
        results = [(Result(s[0], 0), s[1]) for s in scores]
        return results, Result(run_time, 0)
    else:
        print('Validation runs: ', number_of_runs)
        scores = [[] for _ in penalties]
        run_times = []
        for i in range(number_of_runs):
            kernel_matrices_paths, run_time = generate_kernels(kernel, i)
            kernel_matrices_path = kernel_matrices_paths[0]
            label_path = os.path.join(data_dir, dataset_name,
                                      '{}_graph_labels.txt'.format(
                                          dataset_name))
            kernel_matrices = read_matrix(kernel_matrices_path,
                                          read_kernel_matrix)
            labels = read_matrix(label_path, read_label_matrix)
            for i in range(len(penalties)):
                scores[i].append(
                    score_n_fold(kernel_matrices, labels, 10, penalties[i]))
            run_times.append(run_time)
        result_scores = []
        for s in scores:
            c = s[0][1]
            run_scores = [x[0] for x in s]
            acc = np.mean(run_scores)
            std_e = sem(run_scores)
            result_scores.append((Result(acc, std_e), c))
        run_time = Result(np.mean(run_times), sem(run_times))
        return result_scores, run_time


def run_benchmark(dataset_names, kernel_names):
    print('Tested datasets: ', dataset_names)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.exists(os.path.join(current_dir, 'tmp', 'results')):
        os.makedirs(os.path.join(current_dir, 'tmp', 'results'))
    output_path = os.path.join(current_dir, 'tmp', 'results')
    data_dir = os.path.join(current_dir, 'datasets')
    result = defaultdict(list)
    for dataset_name in dataset_names:
        result_path = os.path.join(output_path, dataset_name)
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        kernels = get_benchmarking_kernels(dataset_name, output_path, data_dir,
                                           kernel_names)
        for kernel in kernels:
            path = os.path.join(result_path, kernel.kernel_name)
            benchmark_result = evaluate(kernel, dataset_name, data_dir)
            write_partial_result(path, benchmark_result)
            result[dataset_name].append((kernel.kernel_name, benchmark_result))
    return result


def write_partial_result(path, result):
    with open(path, 'w') as f:
        f.write(json.dumps(result))


def main():
    parser = argparse.ArgumentParser(description='Starting benchmark')
    parser.add_argument('-d', '--data', help='Benchmark datasets', required=True,
                        nargs='+')
    parser.add_argument('-k', '--kernels', help='Benchmark kernels', required=True,
                        nargs='+')
    args = vars(parser.parse_args())
    datasets = args['data']
    kernels = args['kernels']
    #print(json.dumps(run_benchmark(datasets, kernels), indent=4))


if __name__ == '__main__':
    main()
