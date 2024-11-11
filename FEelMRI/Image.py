import numpy as np

from FEelMRI.Generator import (get_cdense_image, get_cspamm_image,
                                  get_exact_image, get_oriocspamm_image)


# Base Image object
class ImageBase(object):
  def __init__(self,
               FOV=np.array([0.1,0.1,0.08]),
               resolution=np.array([30, 30, 1]),
               center=np.array([0.0, 0.0, 0.0]),
               encoding_direction=[0, 1],
               flip_angle=0.5*np.pi,
               T1=1.0, T2=0.5, M0=1.0,
               off_resonance=[],
               kspace_factor=6.5,
               slice_thickness = [],
               slice_offset=0,
               oversampling_factor=2,
               phase_profiles=64,
               filter='Tukey',
               filter_width=0.9,
               filter_lift=0.3):
    self.FOV = FOV
    self.resolution = resolution
    self.center = center
    self.encoding_direction = encoding_direction
    self.flip_angle = flip_angle
    self.T1 = T1
    self.T2 = T2
    self.M0 = M0
    self.grid = self.generate_grid()
    self.off_resonance = off_resonance
    self.kspace_factor = kspace_factor
    if slice_thickness != []:
        self.slice_thickness = slice_thickness
    else:
        self.slice_thickness = self.FOV[-1]
    self.slice_offset = slice_offset
    self.oversampling_factor = oversampling_factor
    self.phase_profiles = phase_profiles
    self.acq_matrix = np.array([oversampling_factor*resolution[0], phase_profiles])
    self.filter = filter
    self.filter_width = filter_width
    self.filter_lift = filter_lift

    # Check input resolutions
    ass_err = 'The generation process can only handle square resolutions'
    assert resolution[0] == resolution[1], ass_err

  # Flip angles
  def flip_angle_t(self, alpha):
    if isinstance(alpha,float) or isinstance(alpha,int):
      alpha = alpha*np.ones([n_t],dtype=np.float)
    return alpha

  # Geometric dimension
  def geometric_dimension(self):
    return self.resolution.size

  # Pixel size
  def voxel_size(self):
    return self.FOV/self.resolution

  # Field of view
  def field_of_view(self):
    return self.FOV

  # Image grids
  def generate_grid(self, sparse=False):
    # Resolution
    resolution = self.resolution

    # Image dimension
    d = resolution.size

    # Pixel size
    pxsz = self.voxel_size()

    # np.meshgrid generation
    X = [pxsz[i]*np.linspace(0,resolution[i]-1,resolution[i]) + self.center[i] for i in range(d)]
    X = [X[i] - X[i].mean() if resolution[i]>1 else X[i] for i in range(d)]

    if d == 2:
      grid = np.meshgrid(X[0], X[1], indexing='ij', sparse=sparse)
    elif d == 3:
      grid = np.meshgrid(X[0], X[1], X[2], indexing='ij', sparse=sparse)

    return grid

  # Type of image dimension
  def type_dim(self):
    return self.geometric_dimension()

  # Voxel centers
  def voxel_midpoints(self):
    d  = len(self.resolution)
    Xc = np.zeros([self._grid[0].size, d])
    for i in range(d):
      Xc[:,i] = self._grid[i].flatten('C')

    return Xc


# DENSE Image
class DENSEImage(ImageBase):
  def __init__(self, encoding_frequency=None,**kwargs):
    self.encoding_frequency = encoding_frequency
    super(DENSEImage, self).__init__(**kwargs)

  def generate(self, epi, phantom, parameters, debug=False):
      return get_cdense_image(self, epi, phantom, parameters, debug)


# EXACT Image
class EXACTImage(ImageBase):
  def __init__(self,encoding_frequency=None,**kwargs):
    self.encoding_frequency = encoding_frequency
    super(EXACTImage, self).__init__(**kwargs)

  def generate(self, epi, phantom, parameters, debug=False):
      return get_exact_image(self, epi, phantom, parameters, debug)


# CSPAMM Image
class CSPAMMImage(ImageBase):
  def __init__(self, encoding_frequency=None,
               encoding_angle=15*np.pi/180,
               taglines=None,**kwargs):
    self.encoding_frequency = encoding_frequency
    self.encoding_angle = encoding_angle
    self.taglines = taglines
    super(CSPAMMImage, self).__init__(**kwargs)

  def generate(self, epi, phantom, parameters, debug=False):
      return get_cspamm_image(self, epi, phantom, parameters, debug)


# ORI-O-CSPAMM Image
class ORI_O_CSPAMMImage(ImageBase):
  def __init__(self, encoding_frequency=None,
               encoding_angle=15*np.pi/180,
               taglines=None,**kwargs):
    self.encoding_frequency = encoding_frequency
    self.encoding_angle = encoding_angle
    self.taglines = taglines
    super(ORI_O_CSPAMMImage, self).__init__(**kwargs)

  def generate(self, epi, phantom, parameters, debug=False):
      return get_oriocspamm_image(self, epi, phantom, parameters, debug)


# SINE Image
class SINEImage(ImageBase):
  def __init__(self, encoding_frequency=None,
               taglines=None,**kwargs):
    self.encoding_frequency = encoding_frequency
    self.taglines = taglines
    super(SINEImage, self).__init__(**kwargs)


# PCSPAMM Image
class PCSPAMMImage(ImageBase):
  def __init__(self, encoding_frequency=None,
               v_encoding_frequency=None,
               encoding_angle=15*np.pi/180,
               taglines=None,**kwargs):
    self.encoding_frequency = encoding_frequency
    self.vel_encoding_frequency = v_encoding_frequency
    self.encoding_angle = encoding_angle
    self.taglines = taglines
    super(PCSPAMMImage, self).__init__(**kwargs)
