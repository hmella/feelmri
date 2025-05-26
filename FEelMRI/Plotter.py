import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os

class MRIPlotter:
  """
  A class to plot MRI images with interactive slice and frame navigation.

  Attributes:
    images (list): List of MRI images.
    FOV (np.array): Field of view dimensions.
    caxis (list): Color axis limits.
    cmap (Colormap): Colormap for the images.
    title (list): Titles for each subplot.
    swap_axes (tuple): Axes to swap.
    shape (tuple): Shape of the subplot grid.
    next_frame_key (str): Key to go to the next frame.
    previous_frame_key (str): Key to go to the previous frame.
    next_slice_key (str): Key to go to the next slice.
    previous_slice_key (str): Key to go to the previous slice.
  """

  def __init__(self, images=[], FOV=[1, 1, 1], caxis=None, cmap=plt.get_cmap('Greys_r'), title=[], swap_axes=None, shape=None, next_frame_key='d', previous_frame_key='a', next_slice_key='w', previous_slice_key='s'):
    """
    Initializes the MRIPlotter with given parameters.

    Args:
      images (list): List of MRI images.
      FOV (list): Field of view dimensions.
      caxis (list): Color axis limits.
      cmap (Colormap): Colormap for the images.
      title (list): Titles for each subplot.
      swap_axes (tuple): Axes to swap.
      shape (tuple): Shape of the subplot grid.
      next_frame_key (str): Key to go to the next frame.
      previous_frame_key (str): Key to go to the previous frame.
      next_slice_key (str): Key to go to the next slice.
      previous_slice_key (str): Key to go to the previous slice.
    """
    self.images = images
    self.FOV = np.array(FOV)
    self.caxis = caxis
    self.cmap = cmap
    self.title = title if title else ['',] * len(self.images)
    self.swap_axes = swap_axes
    self.shape = shape
    self.next_frame_key = next_frame_key
    self.previous_frame_key = previous_frame_key
    self.next_slice_key = next_slice_key
    self.previous_slice_key = previous_slice_key

    self._images = [im.swapaxes(0, 1) for im in self.images]
    self._FOV = self.FOV[[1, 0, 2]]
    if self.swap_axes is not None:
      self._images = [im.swapaxes(self.swap_axes[0], self.swap_axes[1]) for im in self._images]
      self._FOV[self.swap_axes[::-1]] = self._FOV[self.swap_axes]

    self.check_dimensions()
    self.create_figures()

  def check_dimensions(self):
    """
    Ensures all images have 4 dimensions.
    """
    while len(self._images[0].shape) < 4:
      self._images = [im[..., np.newaxis] for im in self._images]

  def create_figures(self):
    """
    Creates the figure and axes for plotting.
    """
    if self.shape is not None:
      self.fig, self.ax = plt.subplots(self.shape[0], self.shape[1])
    else:
      if len(self.images) < 4:
        self.fig, self.ax = plt.subplots(1, len(self.images))
        if len(self.images) == 1:
          self.ax = [self.ax, None]
      else:
        cols = np.ceil(np.sqrt(len(self.images))).astype(int)
        rows = np.ceil(len(self.images) / cols).astype(int)
        self.fig, self.ax = plt.subplots(rows, cols)

  def export_images(self, output_dir):
    """
    Exports all frames and slices of the MRI images to PNG files.

    Args:
      output_dir (str): Directory to save the exported images.
    """

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    extent = [0, self._FOV[1], 0, self._FOV[0]]
    flat_ax = self.ax.flatten()

    for i, im in enumerate(self._images):
        num_slices = im.shape[2]
        num_frames = im.shape[3]

        for slice_idx in range(num_slices):
            for frame_idx in range(num_frames):
                vmin, vmax = (im.min(), im.max()) if self.caxis is None else (self.caxis[i][0], self.caxis[i][1]) if isinstance(self.caxis[0], list) else (self.caxis[0], self.caxis[1])

                fig, ax = plt.subplots()
                image = ax.imshow(im[..., slice_idx, frame_idx], cmap=self.cmap, vmin=vmin, vmax=vmax, extent=extent)
                ax.invert_yaxis()
                ax.set_title(f'{self.title[i]} - Slice {slice_idx}, Frame {frame_idx}')
                ax.set_xticklabels([])
                ax.set_yticklabels([])

                divider = make_axes_locatable(ax)
                colorbar_axes = divider.append_axes("right", size="10%", pad=0.1)
                cbar = fig.colorbar(image, cax=colorbar_axes)
                cbar.minorticks_on()

                filename = os.path.join(output_dir, f'image_{i}_slice_{slice_idx}_frame_{frame_idx}.png')
                plt.savefig(filename, bbox_inches='tight')
                plt.close(fig)

  def show(self):
    """
    Displays the MRI images with interactive navigation.
    """
    extent = [0, self._FOV[1], 0, self._FOV[0]]
    self.remove_keymap_conflicts({self.next_slice_key, self.previous_slice_key, self.next_frame_key, self.previous_frame_key})

    flat_ax = self.ax.flatten()
    for i, (im, ax) in enumerate(zip(self._images, flat_ax)):
        ax.im = im
        ax.slice = 0
        ax.frame = 0

        vmin, vmax = (im.min(), im.max()) if self.caxis is None else (self.caxis[i][0], self.caxis[i][1]) if isinstance(self.caxis[0], list) else (self.caxis[0], self.caxis[1])
        image = ax.imshow(im[..., ax.slice, ax.frame], cmap=self.cmap, vmin=vmin, vmax=vmax, extent=extent)

        if len(self._images) < len(flat_ax):
            [flat_ax[j].axis('off') for j in range(len(self._images), len(flat_ax))]

        ax.invert_yaxis()
        ax.set_title(self.title[i])
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        divider = make_axes_locatable(ax)
        colorbar_axes = divider.append_axes("right", size="10%", pad=0.1)
        cbar = self.fig.colorbar(image, cax=colorbar_axes)
        cbar.minorticks_on()

    self.fig.canvas.mpl_connect('key_press_event', self.process_key)

    self.fig.suptitle('Slice {:d}, frame {:d}'.format(ax.slice, ax.frame))
    self.fig.tight_layout()
    plt.show()

  def previous_slice(self, ax):
    """
    Goes to the previous slice.

    Args:
      ax (Axes): The axis to update.
    """
    im = ax.im
    ax.slice = (ax.slice - 1) % im.shape[2]
    ax.images[0].set_array(im[..., ax.slice, ax.frame])
    self.fig.suptitle('Slice {:d}, frame {:d}'.format(ax.slice, ax.frame))

  def next_slice(self, ax):
    """
    Goes to the next slice.

    Args:
      ax (Axes): The axis to update.
    """
    im = ax.im
    ax.slice = (ax.slice + 1) % im.shape[2]
    ax.images[0].set_array(im[..., ax.slice, ax.frame])
    self.fig.suptitle('Slice {:d}, frame {:d}'.format(ax.slice, ax.frame))

  def previous_frame(self, ax):
    """
    Goes to the previous frame.

    Args:
      ax (Axes): The axis to update.
    """
    im = ax.im
    ax.frame = (ax.frame - 1) % im.shape[3]
    ax.images[0].set_array(im[..., ax.slice, ax.frame])
    self.fig.suptitle('Slice {:d}, frame {:d}'.format(ax.slice, ax.frame))

  def next_frame(self, ax):
    """
    Goes to the next frame.

    Args:
      ax (Axes): The axis to update.
    """
    im = ax.im
    ax.frame = (ax.frame + 1) % im.shape[3]
    ax.images[0].set_array(im[..., ax.slice, ax.frame])
    self.fig.suptitle('Slice {:d}, frame {:d}'.format(ax.slice, ax.frame))

  def remove_keymap_conflicts(self, new_keys_set):
    """
    Removes keymap conflicts with defined keys for slices and frames.

    Args:
      new_keys_set (set): Set of new keys to avoid conflicts with.
    """
    for prop in plt.rcParams:
      if prop.startswith('keymap.'):
        keys = plt.rcParams[prop]
        remove_list = set(keys) & new_keys_set
        for key in remove_list:
          keys.remove(key)

  def process_key(self, event):
    """
    Processes key press events for navigation.

    Args:
      event (Event): The key press event.
    """
    fig = event.canvas.figure
    for ax in fig.axes[:len(self._images)]:
      if event.key == self.previous_slice_key:
        self.previous_slice(ax)
      elif event.key == self.next_slice_key:
        self.next_slice(ax)
      elif event.key == self.previous_frame_key:
        self.previous_frame(ax)
      elif event.key == self.next_frame_key:
        self.next_frame(ax)
    fig.canvas.draw()