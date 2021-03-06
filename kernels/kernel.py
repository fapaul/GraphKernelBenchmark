import os
from abc import ABC, abstractmethod


class Kernel(ABC):

    def __init__(self, dataset_name, output_path, dataset_path, workers=1):
        self.datasetname = dataset_name
        self.dataset = dataset_path
        self.output_path = output_path
        self.workers = workers

    def compile(self):
        pass

    @abstractmethod
    def load_data(self):
        pass

    @abstractmethod
    def compute_kernel_matrices(self, run_number=0):
        pass

    @staticmethod
    def get_tmp_dir():
        current_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(current_dir, '..', 'tmp')

    def is_deterministic(self):
        return True