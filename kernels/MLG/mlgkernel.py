import subprocess
import os
import numpy as np
from .. import kernel
from scipy import sparse as sps


class MLGKernel(kernel.Kernel):

    def load_data(self):
        matrices = self.load_dense_matrix(self.dataset, self.datasetname)
        lines = self.convert_to_mlg_format(matrices)
        self.tmp_file = 'MLG_{}_converted.txt'.format(self.datasetname)
        with open(os.path.join(self.get_tmp_dir(), self.tmp_file), 'w') as f:
            f.write('\n'.join(lines))

    def compute_kernel_matrices(self):
        converted = os.path.join(self.get_tmp_dir(), self.tmp_file)
        env = os.environ.copy()
        env['DSET'] = self.datasetname
        env['DATA'] = converted
        output = os.path.join(self.output_path, 'MLG')
        env['OUTPUT'] = output
        repo_start = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  'MLGkernel')
        p = subprocess.Popen(['sh', 'sample.sh'], env=env, cwd=repo_start)
        p.wait()
        return [os.path.join(output, self.datasetname + '_output.txt')]

    @staticmethod
    def load_dense_matrix(data_dir, dataset):
        file_start = os.path.join(data_dir, dataset, dataset)
        offsets = np.loadtxt(file_start + '_graph_indicator.txt',
                             dtype=np.int, delimiter=',') - 1
        offs = np.append([0],
                         np.append(np.where((offsets[1:] -
                                             offsets[:-1]) > 0)[0] + 1,
                         len(offsets)))
        A_data = np.loadtxt(file_start + '_A.txt',
                            dtype=np.int, delimiter=',') - 1
        A_mat = sps.csr_matrix((np.ones(A_data.shape[0]),
                               (A_data[:, 0], A_data[:, 1])), dtype=np.int)
        As = []
        for i in range(1, len(offs)):
            As.append(A_mat[offs[i - 1]:offs[i], offs[i - 1]:offs[i]])
        am = [x.astype(np.float64) for x in As]
        return am

    @staticmethod
    def convert_to_mlg_format(matrices):
        lines = []
        lines.append(str(len(matrices)))
        for matrix in matrices:
            lines.append(str(matrix.shape[0]))
            for row in matrix.todense():
                lines.append(' '.join([str(int(x)) for x in row.tolist()[0]]))
        return lines
