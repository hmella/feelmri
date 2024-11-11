import numpy as np

# Array orderings, which depend on the emasurement
# direction
order = ['F','C']

# Measurement and phase directions
# The first element of m_dirs[i] (with i the encoding)
# direction) denotes the measurement direction, and the
# second the phase direction
m_dirs = [[0,1], [1,0]]

# Check the croping indices inside the function
# FEelMRI.MRImaging.acq_to_res
def build_idx(n_lines, acq_matrix, dir):
    # Number of additional lines
    add = int(n_lines/2*acq_matrix[dir[0]])

    # Flattened array indices
    idx = np.array([add,-add])
    # idx = np.flip(np.sort(idx))

    # Check if the last element is zero
    if idx[-1] == 0:
        idx[-1] = np.prod(acq_matrix)

    return idx

# Cropping ranges to correct the shape of generated images
def cropping_ranges(im_resolution, gen_resolution, ovrs_fac):

    # Abbreviations
    gr = gen_resolution
    ir = im_resolution

    # Number of measurements in the extended kspace
    n = gr - ovrs_fac*ir

    # Ranges
    r = [int(n[0]/2), int(n[0]/2 + ovrs_fac*ir[0])]
    c = [int(n[1]/2), int(n[1]/2 + ovrs_fac*ir[1])]
    dr = [1, ovrs_fac]
    dc = [ovrs_fac, 1]

    return r, c, dr, dc

# Restore the virtual resolution to the acquired one. When the image resolution
# has odd values, the kspace resampling is shifted in one pixel
def restore_resolution(k,r,c,dr,dc,enc_dir,image_resolution,ovrs_fac):
    if isodd(image_resolution[enc_dir]):
        shift = ovrs_fac-1
        if enc_dir == 0:
            k_out = k[r[0]:r[1]:dr[enc_dir], c[0]+shift:c[1]:dc[enc_dir]]
        else:
            k_out = k[r[0]+shift:r[1]:dr[enc_dir], c[0]:c[1]:dc[enc_dir]]
    else:
        k_out = k[r[0]:r[1]:dr[enc_dir], c[0]:c[1]:dc[enc_dir]]
    return k_out

def iseven(arg):
    return np.mod(arg, 2) == 0

def isodd(arg):
    return np.mod(arg, 2) != 0

def round_to_even(arg):
    eps = 1e-10
    return np.round(eps + arg/2)*2