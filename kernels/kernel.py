import os
from abc import ABC, abstractmethod


class Kernel(ABC):

    def __init__(self, dataset_name, output_path, dataset_path):
        self.name = dataset_name
        self.dataset = dataset_path
        self.output_path = output_path

    @abstractmethod
    def load_data(self):
        pass

    @abstractmethod
    def compute_kernel_matrices(self):
        pass

    @staticmethod
    def get_tmp_dir():
        current_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(current_dir, '..', 'tmp')
