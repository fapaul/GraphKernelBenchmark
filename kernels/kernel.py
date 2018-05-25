from abc import ABC, abstractmethod


class Kernel(ABC):

    def __init__(self, name, output_path, command, dataset_path):
        self.name = name
        self.command = command
        self.dataset = dataset_path

    @abstractmethod
    def load_data(self):
        pass

    @abstractmethod
    def compute_kernel_matrices(self):
        pass
