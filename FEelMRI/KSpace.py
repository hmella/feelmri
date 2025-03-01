import matplotlib.pyplot as plt
import numpy as np

from FEelMRI.Filters import Hamming_filter, Riesz, Tukey
from FEelMRI.Helpers import build_idx, isodd, order
from FEelMRI.IO import rescale_image, scale_data
from FEelMRI.Math import itok, ktoi


# kspace class
class kspace:
    def __init__(self, shape, image, artifact):
        # Output kspace shape
        self.shape = shape
        self.acq_matrix = image.acq_matrix
        self.oversampling_factor = image.oversampling_factor
        self.artifact = artifact
        self.k = np.zeros(shape,dtype=np.complex128)
        self.k_msk = np.zeros(shape,dtype=np.float32)
        self.k_acq = np.zeros(np.append(self.acq_matrix,shape[2:5]),dtype=np.complex128)
        self.filter = image.filter
        self.filter_width = image.filter_width
        self.filter_lift = image.filter_lift

    def to_img(self):
        return ktoi(self.k)

    def gen_to_acq(self, k_gen, delta_ph, dir, slice, enc_dir, timestep):

        # Input kspace resolution
        resolution = k_gen.shape
        acq_matrix = self.acq_matrix[dir]

        # Number of additional lines
        n_lines = resolution[dir[1]] - acq_matrix[dir[1]]

        # Adapt kspace using the phase-corrected shape
        idx = build_idx(n_lines, acq_matrix, dir)
        k_acq = np.copy(k_gen).flatten(order[dir[0]])[idx[0]:idx[1]]

        # Add epi artifacts
        if self.artifact != None:
            k_acq = self.artifact.kspace(k_acq, delta_ph, dir, T2star=0.02)

        # Filter to reduce Gibbs ringing artifacts
        if self.filter == 'Tukey':
            Hm = Tukey(acq_matrix[dir[0]],width=self.filter_width,
                              lift=self.filter_lift)
            Hp = Tukey(acq_matrix[dir[1]],width=self.filter_width,
                              lift=self.filter_lift)
        if self.filter == 'Riesz':
            Hm = Riesz(acq_matrix[dir[0]],width=self.filter_width,
                              lift=self.filter_lift)
            Hp = Riesz(acq_matrix[dir[1]],width=self.filter_width,
                              lift=self.filter_lift)
        if self.filter is None:
            Hm = np.ones([acq_matrix[dir[0]],])
            Hp = np.ones([acq_matrix[dir[1]],])
        H = np.outer(Hm,Hp).flatten('F')
        k_acq_filt = H*k_acq

        # Fill final kspace
        pshape = np.copy(acq_matrix)
        pshape[dir[1]] += n_lines
        k_new = np.zeros(pshape, dtype=np.complex128).flatten(order[dir[0]])
        k_new[idx[0]:idx[1]] = k_acq_filt

        # Mask
        k_mask = np.zeros(pshape, dtype=float).flatten(order[dir[0]])
        k_mask[idx[0]:idx[1]] = H

        # Remove oversampled values
        pshape[dir[0]] /= self.oversampling_factor

        # Kspace shift
        if isodd(pshape[dir[0]]):
            shift = self.oversampling_factor - 1
        else:
            shift = 0

        # Store kspaces
        k_tmp_0 = np.reshape(k_new[shift::self.oversampling_factor], pshape, order=order[dir[0]])
        k_tmp_1 = np.reshape(k_acq, acq_matrix[dir], order=order[dir[enc_dir]])
        self.k[...,slice,enc_dir,timestep] = k_tmp_0
        self.k_acq[...,slice,enc_dir,timestep] = k_tmp_1

        # Store mask
        k_tmp_msk = np.reshape(k_mask[shift::self.oversampling_factor], pshape, order=order[dir[0]])
        self.k_msk[...,slice,enc_dir,timestep] = k_tmp_msk

    def scale(self,dtype=np.uint64):
        self.k = scale_data(self.k,mag=False,real=True,imag=True,dtype=dtype)
        self.k_msk = scale_data(self.k_msk,mag=False,real=True,imag=False,dtype=dtype)
        self.k_acq = scale_data(self.k_acq,mag=False,real=True,imag=True,dtype=dtype)

    def rescale(self):
        tmp = rescale_image(self.k)
        self.k = tmp['real'] + 1j*tmp['complex']
        tmp = rescale_image(self.k_msk)
        self.k_msk = tmp['real']
        tmp = rescale_image(self.k_acq)
        self.k_acq = tmp['real'] + 1j*tmp['complex']
