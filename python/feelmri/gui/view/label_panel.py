"""Assign labels to blocks and to individual MR objects.

The library's own labelling is read-only and per block: `PulseqImport` computes
the running `LABELSET` state and `filter_blocks` queries it. This panel is the
writing half, over `gui.model.labels.LabelStore`, plus a layer the file format
has no concept of at all -- labels on one RF pulse, gradient or ADC.

**Two destinations, and they are not equivalent.** A sidecar keeps everything,
including object labels, and is what a later session reloads.

A `.seq` export loses two things, and the panel says so rather than letting a
user discover them. `LABELSET` addresses blocks and nothing finer, so object
labels are merged down onto their block, the block value winning a collision.
And `LABELSET` is STICKY -- a value persists until something sets it again,
and the format cannot unset one -- so the export goes through
`to_running_labels()`, which carries each value forward. A label put on a
single block therefore applies from that block onward in the written file,
which is what the file would mean whatever was intended.
"""
from __future__ import annotations

from typing import Callable, List, Optional

from .theme import TEXT_OPTIONS


def supported_labels() -> List[str]:
  """Pulseq's own label names, or an empty list when pypulseq is absent.

  Offered as suggestions rather than enforced: the store happily holds any
  name, and only a `.seq` export needs one Pulseq knows. Refusing early would
  block a sidecar-only workflow for no reason.
  """
  try:
    import pypulseq as pp
    return list(pp.get_supported_labels())
  except Exception:
    return []


def parse_value(text: str):
  """A label value: an int where it reads as one, otherwise the text.

  `LABELSET` is integer-valued, so a `.seq` export refuses anything else -- but
  the store and the sidecar carry strings fine, and a name like `tag=systole`
  is useful while planning. The conversion happens here so the two paths can
  differ without the caller thinking about it.
  """
  token = text.strip()
  if not token:
    raise ValueError('a label needs a value')
  try:
    return int(token)
  except ValueError:
    return token


class LabelPanel:
  """Label the selected block or MR object, and save or export the result."""

  def __init__(self, parent, session,
               on_status: Optional[Callable[[str], None]] = None):
    import tkinter as tk
    from tkinter import ttk

    self.session = session
    self.on_status = on_status or (lambda _: None)
    self.block: Optional[int] = None
    self.obj = None

    self.widget = ttk.Frame(parent, padding=10)

    self._target = ttk.Label(self.widget, text='No block selected.',
                             wraplength=280, justify='left',
                             font=('TkDefaultFont', 10, 'bold'))
    self._target.pack(anchor='w')
    ttk.Label(self.widget, wraplength=280, justify='left', text=(
      'Click the sequence panel to choose a block, or a pulse, gradient or '
      'ADC within it.')).pack(anchor='w', pady=(2, 10))

    ttk.Label(self.widget, text='Label').pack(anchor='w')
    self._name = tk.StringVar(value='SET')
    ttk.Combobox(self.widget, textvariable=self._name,
                 values=supported_labels()).pack(fill='x')

    ttk.Label(self.widget, text='Value').pack(anchor='w', pady=(6, 0))
    self._value = tk.StringVar(value='0')
    ttk.Entry(self.widget, textvariable=self._value).pack(fill='x')

    self._scope = tk.StringVar(value='block')
    scope = ttk.Frame(self.widget)
    scope.pack(fill='x', pady=(8, 0))
    for text, value in (('This block', 'block'), ('This object', 'object'),
                        ('Block range', 'range')):
      ttk.Radiobutton(scope, text=text, value=value,
                      variable=self._scope).pack(anchor='w')

    self._range = tk.StringVar(value='')
    ttk.Label(self.widget, text='Range (first last)').pack(anchor='w',
                                                           pady=(6, 0))
    ttk.Entry(self.widget, textvariable=self._range).pack(fill='x')

    buttons = ttk.Frame(self.widget)
    buttons.pack(fill='x', pady=(10, 0))
    ttk.Button(buttons, text='Apply', command=self.apply).pack(side='left')
    ttk.Button(buttons, text='Clear', command=self.clear).pack(side='left',
                                                               padx=6)

    ttk.Separator(self.widget).pack(fill='x', pady=12)
    ttk.Label(self.widget, text='Labels in use',
              font=('TkDefaultFont', 10, 'bold')).pack(anchor='w')
    self._summary = tk.Text(self.widget, height=7, width=34,
                            font=('TkFixedFont', 9), **TEXT_OPTIONS)
    self._summary.pack(fill='x', pady=(2, 0))
    self._summary.configure(state='disabled')

    ttk.Separator(self.widget).pack(fill='x', pady=12)
    for text, command in (('Save sidecar...', self.save_sidecar),
                          ('Load sidecar...', self.load_sidecar),
                          ('Export labelled .seq...', self.export_seq)):
      ttk.Button(self.widget, text=text, command=command).pack(fill='x',
                                                               pady=2)

    session.sequence_changed.connect(lambda *_: self.set_target(None, None))
    session.labels_changed.connect(lambda *_: self.refresh())
    self.refresh()

  # -- selection ------------------------------------------------------------

  def set_target(self, block: Optional[int], obj=None) -> None:
    self.block, self.obj = block, obj
    if block is None:
      self._target.config(text='No block selected.')
    elif obj is None:
      self._target.config(text=f'Block {block}')
    else:
      self._target.config(text=obj.label)
    self.refresh()

  # -- editing --------------------------------------------------------------

  @property
  def store(self):
    return getattr(self.session, 'labels', None)

  def apply(self) -> None:
    from tkinter import messagebox

    store = self.store
    if store is None:
      messagebox.showinfo('No sequence', 'Open a sequence first.')
      return
    try:
      name = self._name.get().strip()
      if not name:
        raise ValueError('a label needs a name')
      value = parse_value(self._value.get())
      scope = self._scope.get()
      if scope == 'object':
        if self.obj is None:
          raise ValueError('no MR object is selected; click one in the '
                           'sequence panel')
        store.set_object(self.obj.block, self.obj.kind, self.obj.ordinal,
                         name, value)
        where = self.obj.label
      elif scope == 'range':
        first, last = self._parse_range(store.n_blocks)
        store.set_blocks(range(first, last + 1), name, value)
        where = f'blocks {first}-{last}'
      else:
        if self.block is None:
          raise ValueError('no block is selected')
        store.set_block(self.block, name, value)
        where = f'block {self.block}'
    except Exception as exc:
      messagebox.showerror('Could not set the label', str(exc))
      return
    self.session.notify_labels()
    self.on_status(f'{name}={value} on {where}')

  def clear(self) -> None:
    from tkinter import messagebox

    store = self.store
    if store is None:
      return
    name = self._name.get().strip() or None
    try:
      if self._scope.get() == 'object':
        if self.obj is None:
          raise ValueError('no MR object is selected')
        store.clear_object(self.obj.block, self.obj.kind, self.obj.ordinal,
                           name)
        where = self.obj.label
      elif self._scope.get() == 'range':
        first, last = self._parse_range(store.n_blocks)
        for block in range(first, last + 1):
          store.clear_block(block, name)
        where = f'blocks {first}-{last}'
      else:
        if self.block is None:
          raise ValueError('no block is selected')
        store.clear_block(self.block, name)
        where = f'block {self.block}'
    except Exception as exc:
      messagebox.showerror('Could not clear the label', str(exc))
      return
    self.session.notify_labels()
    self.on_status(f'cleared {name or "every label"} on {where}')

  def _parse_range(self, n_blocks: int):
    parts = self._range.get().replace(',', ' ').split()
    if len(parts) != 2:
      raise ValueError('a range is two block indices, for example "0 39"')
    first, last = int(parts[0]), int(parts[1])
    if first > last:
      first, last = last, first
    if not (0 <= first < n_blocks and 0 <= last < n_blocks):
      raise ValueError(f'range {first}-{last} is outside 0-{n_blocks - 1}')
    return first, last

  # -- persistence ----------------------------------------------------------

  def save_sidecar(self) -> None:
    from tkinter import filedialog, messagebox

    store = self.store
    if store is None:
      messagebox.showinfo('No sequence', 'Open a sequence first.')
      return
    initial = getattr(self.session, 'sequence_path', None)
    suggested = (str(store.sidecar_path(initial)) if initial else
                 'labels.yaml')
    path = filedialog.asksaveasfilename(
      title='Save labels', initialfile=suggested.rsplit('/', 1)[-1],
      defaultextension='.yaml',
      filetypes=[('Label sidecar', '*.yaml'), ('All files', '*')])
    if not path:
      return
    try:
      store.save(path)
    except Exception as exc:
      messagebox.showerror('Could not save the labels', str(exc))
      return
    self.on_status(f'labels saved to {path}')

  def load_sidecar(self) -> None:
    from tkinter import filedialog, messagebox

    from ..model.labels import LabelStore, StaleSidecarError

    path = filedialog.askopenfilename(
      title='Load labels',
      filetypes=[('Label sidecar', '*.yaml'), ('All files', '*')])
    if not path:
      return
    sequence_path = getattr(self.session, 'sequence_path', None)
    try:
      store = LabelStore.load(path, seq_path=sequence_path)
    except StaleSidecarError as exc:
      # Checksummed on purpose: a sidecar applied to the wrong file does not
      # fail, it MISLABELS, since block indices are all that tie the two
      # together. Offer the override rather than take it.
      if not messagebox.askyesno(
          'Sidecar does not match this sequence',
          f'{exc}\n\nLoad it anyway? Labels are tied to a sequence by block '
          f'index alone, so the wrong file will be silently mislabelled.'):
        return
      try:
        store = LabelStore.load(path, seq_path=sequence_path, check=False)
      except Exception as inner:
        messagebox.showerror('Could not load the labels', str(inner))
        return
    except Exception as exc:
      messagebox.showerror('Could not load the labels', str(exc))
      return

    self.session.labels = store
    self.session.notify_labels()
    self.on_status(f'labels loaded from {path}')

  def export_seq(self) -> None:
    from tkinter import filedialog, messagebox

    from ..model.labels import write_labelled_seq

    store = self.store
    source = getattr(self.session, 'sequence_path', None)
    if store is None or not source:
      messagebox.showinfo('No sequence file',
                          'Open a .seq first: the export rewrites that file '
                          'rather than building one.')
      return
    if not messagebox.askyesno(
        'How a .seq stores labels',
        'A .seq addresses blocks, not the objects inside them, so object '
        'labels are merged down onto their block and the block value wins a '
        'collision. The sidecar keeps both layers.\n\n'
        'LABELSET is also sticky: a value persists until something sets it '
        'again, and the format cannot unset one. A label put on a single '
        'block therefore applies from that block onward in the written '
        'file.\n\nContinue?'):
      return
    path = filedialog.asksaveasfilename(
      title='Export labelled sequence', defaultextension='.seq',
      filetypes=[('Pulseq', '*.seq'), ('All files', '*')])
    if not path:
      return
    try:
      write_labelled_seq(source, path, store.to_running_labels())
    except Exception as exc:
      messagebox.showerror('Could not export the sequence', str(exc))
      return
    self.on_status(f'labelled sequence written to {path}')

  # -- display --------------------------------------------------------------

  def refresh(self) -> None:
    store = self.store
    lines = []
    if store is None:
      lines.append('no sequence loaded')
    else:
      merged = store.to_block_labels()
      for name in store.names():
        values = {}
        for block, labels in enumerate(merged):
          if name in labels:
            values.setdefault(labels[name], []).append(block)
        parts = ', '.join(f'{value}: {len(blocks)}'
                          for value, blocks in sorted(values.items(),
                                                      key=lambda kv: str(kv[0])))
        lines.append(f'{name:<6s} {parts}')
      if store.object_labels:
        lines.append('')
        lines.append(f'{len(store.object_labels)} object label(s)')
      if not lines:
        lines.append('none yet')
    self._summary.configure(state='normal')
    self._summary.delete('1.0', 'end')
    self._summary.insert('end', '\n'.join(lines))
    self._summary.configure(state='disabled')
