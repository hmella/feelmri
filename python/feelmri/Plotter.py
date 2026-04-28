"""
Interactive and export-based MRI image visualization.
"""
import os
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

from feelmri.MPIUtilities import MPI_rank, MPI_comm


class MRIPlotter:
    """Interactive Matplotlib viewer for 4-D MR image stacks.

    Displays one or more images side by side with keyboard navigation
    across slices and time frames. Only the designated MPI rank
    initializes a GUI; all others skip memory allocation entirely.

    Parameters
    ----------
    images : list of np.ndarray
        List of image arrays. Each array must be at least 2-D; a 4-D
        layout ``(Nx, Ny, Nslices, Nframes)`` is assumed after internal
        axis swaps.
    FOV : list of float, optional
        Field-of-view ``[Fx, Fy, Fz]`` used to set physical axis extents.
        Default is ``[1, 1, 1]``.
    caxis : list or None, optional
        Color-axis limits. Either a single ``[vmin, vmax]`` pair shared
        across all images, or a list of ``[vmin, vmax]`` pairs (one per
        image). Default is None (auto-scaled per image).
    cmap : matplotlib colormap, optional
        Colormap to use. Default is ``'Greys_r'``.
    title : list of str, optional
        Subplot titles, one per image. Default is empty strings.
    swap_axes : list of int or None, optional
        Two-element ``[ax1, ax2]`` pair passed to ``np.swapaxes`` on the
        internal image representation after the default axis swap.
    shape : list of int or None, optional
        ``[rows, cols]`` grid layout for the subplots. Inferred
        automatically when None.
    next_frame_key : str, optional
        Keyboard key to advance one frame. Default is ``'d'``.
    previous_frame_key : str, optional
        Keyboard key to go back one frame. Default is ``'a'``.
    next_slice_key : str, optional
        Keyboard key to advance one slice. Default is ``'w'``.
    previous_slice_key : str, optional
        Keyboard key to go back one slice. Default is ``'s'``.
    rank : int, optional
        MPI rank that owns the GUI. Default is 0.
    """

    def __init__(self, images=[], FOV=[1, 1, 1], caxis=None, cmap=None,
                 title=[], swap_axes=None, shape=None, next_frame_key='d',
                 previous_frame_key='a', next_slice_key='w', previous_slice_key='s', rank=0):

        self.rank = rank

        # GUARD: If this is not the target rank, skip memory initialization entirely.
        if MPI_rank != self.rank:
            self.fig = None
            self.ax = None
            return

        # Default cmap handling (moved inside so it doesn't crash headless ranks during import)
        if cmap is None:
            cmap = plt.get_cmap('Greys_r')

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

        # Use transpose/swapaxes to create views rather than full copies where possible
        self._images = [im.swapaxes(0, 1) for im in self.images]
        self._FOV = self.FOV[[1, 0, 2]]

        if self.swap_axes is not None:
            self._images = [im.swapaxes(self.swap_axes[0], self.swap_axes[1]) for im in self._images]
            self._FOV[self.swap_axes[::-1]] = self._FOV[self.swap_axes]

        self._ensure_4d()
        self.fig = None
        self.ax = None

    def _ensure_4d(self):
        for i in range(len(self._images)):
            while self._images[i].ndim < 4:
                self._images[i] = self._images[i][..., np.newaxis]

    def _setup_axes(self):
        if self.fig is not None:
            return

        n = len(self._images)
        if self.shape is not None:
            self.fig, self.ax = plt.subplots(self.shape[0], self.shape[1])
        else:
            if n < 4:
                self.fig, self.ax = plt.subplots(1, n)
            else:
                cols = int(np.ceil(np.sqrt(n)))
                rows = int(np.ceil(n / cols))
                self.fig, self.ax = plt.subplots(rows, cols)

        if not isinstance(self.ax, np.ndarray):
            self.ax = np.array([self.ax])
        self.ax = self.ax.flatten()

    def export_images(self, output_dir, prefix='im', format='png', dpi=150):
        """Save all images to individual PNG (or other format) files.

        Parameters
        ----------
        output_dir : str
            Directory in which to save the images (created if absent).
        prefix : str, optional
            Filename prefix. Default is ``'im'``.
        format : str, optional
            Image file format (e.g. ``'png'``, ``'pdf'``). Default is ``'png'``.
        dpi : int, optional
            Output resolution in dots per inch. Default is 150.
        """
        # GUARD: Only the designated rank performs the file export
        if MPI_rank == self.rank:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            extent = [0, self._FOV[1], 0, self._FOV[0]]

            fig, ax = plt.subplots()
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="10%", pad=0.1)

            for i, im in enumerate(self._images):
                if self.caxis is None:
                    vmin, vmax = im.min(), im.max()
                elif isinstance(self.caxis[0], (list, np.ndarray)):
                    vmin, vmax = self.caxis[i][0], self.caxis[i][1]
                else:
                    vmin, vmax = self.caxis[0], self.caxis[1]

                img_obj = ax.imshow(im[..., 0, 0], cmap=self.cmap, vmin=vmin, vmax=vmax, extent=extent)
                cbar = fig.colorbar(img_obj, cax=cax)
                cbar.minorticks_on()
                ax.invert_yaxis()
                ax.set_xticks([])
                ax.set_yticks([])

                for s_idx in range(im.shape[2]):
                    for f_idx in range(im.shape[3]):
                        img_obj.set_data(im[..., s_idx, f_idx])
                        ax.set_title(f'{self.title[i]} - Slice {s_idx}, Frame {f_idx}')

                        fname = os.path.join(output_dir, f'{prefix}_{i}_s{s_idx}_f{f_idx}.{format}')
                        fig.savefig(fname, bbox_inches='tight', dpi=dpi)

                cax.cla()
                ax.cla()

            plt.close(fig)

        # Synchronize ranks so no one proceeds until the export is finished
        MPI_comm.Barrier()

    def show(self):
        """Display the images in an interactive Matplotlib window.

        Keyboard bindings (configurable via constructor arguments):

        * ``w`` / ``s`` — next / previous slice
        * ``d`` / ``a`` — next / previous frame

        Blocks rank 0 until the window is closed, then synchronizes all
        MPI ranks before returning.
        """
        # GUARD: Only the designated rank runs the GUI logic
        if MPI_rank == self.rank:
            self._setup_axes()
            extent = [0, self._FOV[1], 0, self._FOV[0]]
            self.remove_keymap_conflicts({self.next_slice_key, self.previous_slice_key,
                                          self.next_frame_key, self.previous_frame_key})

            for i, (im, ax) in enumerate(zip(self._images, self.ax)):
                ax.im_data = im
                ax.curr_slice = 0
                ax.curr_frame = 0

                if self.caxis is None:
                    vmin, vmax = im.min(), im.max()
                elif isinstance(self.caxis[0], (list, np.ndarray)):
                    vmin, vmax = self.caxis[i][0], self.caxis[i][1]
                else:
                    vmin, vmax = self.caxis[0], self.caxis[1]

                img_obj = ax.imshow(im[..., 0, 0], cmap=self.cmap, vmin=vmin, vmax=vmax, extent=extent)
                ax.invert_yaxis()
                ax.set_title(self.title[i])
                ax.set_xticks([])
                ax.set_yticks([])

                divider = make_axes_locatable(ax)
                cbar_ax = divider.append_axes("right", size="10%", pad=0.1)
                self.fig.colorbar(img_obj, cax=cbar_ax).minorticks_on()

            for j in range(len(self._images), len(self.ax)):
                self.ax[j].axis('off')

            self.fig.canvas.mpl_connect('key_press_event', self.process_key)
            self._update_suptitle()
            self.fig.tight_layout()

            # This blocks Rank 0 until you manually close the Matplotlib window
            plt.show()

        # Wait for Rank 0. Rank 1 will arrive here instantly and sleep.
        # Rank 0 will arrive here only AFTER the user closes the plot window.
        MPI_comm.Barrier()

    def _update_suptitle(self):
        ref_ax = self.ax[0]
        self.fig.suptitle(f'Slice {ref_ax.curr_slice}, Frame {ref_ax.curr_frame}')

    def process_key(self, event):
        """Handle keyboard navigation events."""
        if event.key not in [self.next_slice_key, self.previous_slice_key,
                              self.next_frame_key, self.previous_frame_key]:
            return

        for ax in self.ax[:len(self._images)]:
            if event.key == self.next_slice_key:
                ax.curr_slice = (ax.curr_slice + 1) % ax.im_data.shape[2]
            elif event.key == self.previous_slice_key:
                ax.curr_slice = (ax.curr_slice - 1) % ax.im_data.shape[2]
            elif event.key == self.next_frame_key:
                ax.curr_frame = (ax.curr_frame + 1) % ax.im_data.shape[3]
            elif event.key == self.previous_frame_key:
                ax.curr_frame = (ax.curr_frame - 1) % ax.im_data.shape[3]

            ax.get_images()[0].set_data(ax.im_data[..., ax.curr_slice, ax.curr_frame])

        self._update_suptitle()
        self.fig.canvas.draw_idle()

    def remove_keymap_conflicts(self, new_keys_set):
        """Remove Matplotlib default keybindings that conflict with navigation keys.

        Parameters
        ----------
        new_keys_set : set of str
            Set of key strings to free from Matplotlib's default keymap.
        """
        for prop in plt.rcParams:
            if prop.startswith('keymap.'):
                keys = plt.rcParams[prop]
                for key in (set(keys) & new_keys_set):
                    keys.remove(key)
