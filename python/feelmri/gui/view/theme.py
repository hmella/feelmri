"""One dark palette, applied to ttk, to plain Tk widgets and to matplotlib.

Three widget families have to be dressed separately and none of them inherits
from the others:

- **ttk** widgets take colours only through a `Style`, and only on a theme
  that actually draws with them. The default theme on Linux (`clam` aside) and
  the native themes on Windows and macOS ignore `background` entirely, so the
  base theme is switched to `clam` first. Without that, half the window stays
  light and it looks like a bug rather than a theme.
- **plain Tk** widgets -- `Text`, `Frame`, `Canvas` -- have no style engine at
  all and need their options set one by one.
- **matplotlib** figures carry their own white background, which is the
  brightest thing on screen if it is missed.

The palette is data, so a panel asks for a colour by name rather than
inventing one, and there is a single place to change.
"""
from __future__ import annotations

#: The palette. Names say the ROLE, not the colour, so a later change to the
#: scheme does not turn every call site into a lie.
PALETTE = {
  'window': '#1b1b1f',          # behind everything
  'surface': '#232329',         # panels and tabs
  'raised': '#2b2b33',          # entries, the selected tab
  'border': '#3a3a44',
  'text': '#e6e6ea',
  'muted': '#9a9aa6',
  'accent': '#5b9cf8',          # selection, focus, the active tab strip
  'accent_text': '#0f0f12',
  'trace': '#7fb2ff',           # gradients
  'rf': '#ff6b6b',
  'adc': '#3fb950',
  'highlight': '#e3a008',       # the picked block, the FOV box
}

#: The scalar colour map, shared by the surface and the glyph arrows so one
#: reading of the bar covers both. `jet`, which is what this field is read in
#: elsewhere in MRI; matplotlib's own default is perceptually better and is not
#: what anyone here is used to.
COLOUR_MAP = 'jet'

#: Tk widget options that are not ttk and have to be set by hand. Passed as
#: keyword arguments, so a caller writes `tk.Text(parent, **TEXT_OPTIONS)`.
TEXT_OPTIONS = dict(background=PALETTE['surface'], foreground=PALETTE['text'],
                    insertbackground=PALETTE['text'],
                    selectbackground=PALETTE['accent'],
                    selectforeground=PALETTE['accent_text'],
                    highlightthickness=0, borderwidth=0, relief='flat')


def apply(root) -> dict:
  """Dress a Tk root and every ttk widget under it. Returns the palette.

  Safe to call on a root that is already themed, and safe to fail: a Tk build
  without `clam` keeps its own look rather than a half-applied one.
  """
  from tkinter import ttk

  style = ttk.Style(root)
  try:
    # `clam` is the only widely available theme that honours `background` on
    # every element. The native themes draw from the platform and silently
    # ignore most of what follows.
    style.theme_use('clam')
  except Exception:
    return PALETTE

  p = PALETTE
  root.configure(background=p['window'])

  style.configure('.', background=p['surface'], foreground=p['text'],
                  fieldbackground=p['raised'], bordercolor=p['border'],
                  lightcolor=p['surface'], darkcolor=p['surface'],
                  troughcolor=p['window'], focuscolor=p['accent'],
                  insertcolor=p['text'])
  style.configure('TFrame', background=p['surface'])
  style.configure('TLabel', background=p['surface'], foreground=p['text'])
  style.configure('TSeparator', background=p['border'])
  style.configure('TLabelframe', background=p['surface'],
                  bordercolor=p['border'])
  style.configure('TLabelframe.Label', background=p['surface'],
                  foreground=p['muted'])

  style.configure('TButton', background=p['raised'], foreground=p['text'],
                  bordercolor=p['border'], focusthickness=1, padding=(8, 3))
  style.map('TButton',
            background=[('pressed', p['accent']), ('active', p['border']),
                        ('disabled', p['surface'])],
            foreground=[('pressed', p['accent_text']),
                        ('disabled', p['muted'])])

  for widget in ('TEntry', 'TCombobox', 'TSpinbox'):
    style.configure(widget, fieldbackground=p['raised'],
                    background=p['raised'], foreground=p['text'],
                    bordercolor=p['border'], arrowcolor=p['text'],
                    insertcolor=p['text'], padding=2)
    style.map(widget,
              fieldbackground=[('readonly', p['raised']),
                               ('disabled', p['surface'])],
              foreground=[('disabled', p['muted'])],
              bordercolor=[('focus', p['accent'])])
  # The dropdown LIST is a plain Tk listbox owned by Tk, not by the style, so
  # it stays light unless told otherwise through the option database.
  root.option_add('*TCombobox*Listbox.background', p['raised'])
  root.option_add('*TCombobox*Listbox.foreground', p['text'])
  root.option_add('*TCombobox*Listbox.selectBackground', p['accent'])
  root.option_add('*TCombobox*Listbox.selectForeground', p['accent_text'])

  style.configure('TNotebook', background=p['window'], bordercolor=p['border'],
                  tabmargins=(4, 4, 4, 0))
  style.configure('TNotebook.Tab', background=p['window'],
                  foreground=p['muted'], bordercolor=p['border'],
                  padding=(12, 5))
  style.map('TNotebook.Tab',
            background=[('selected', p['surface'])],
            foreground=[('selected', p['text']), ('active', p['text'])],
            expand=[('selected', (0, 0, 0, 0))])

  style.configure('TCheckbutton', background=p['surface'],
                  foreground=p['text'], indicatorcolor=p['raised'])
  style.map('TCheckbutton',
            indicatorcolor=[('selected', p['accent'])],
            background=[('active', p['surface'])])
  style.configure('TRadiobutton', background=p['surface'],
                  foreground=p['text'], indicatorcolor=p['raised'])
  style.map('TRadiobutton',
            indicatorcolor=[('selected', p['accent'])],
            background=[('active', p['surface'])])

  style.configure('TScale', background=p['surface'],
                  troughcolor=p['window'], bordercolor=p['border'])
  style.configure('Horizontal.TProgressbar', background=p['accent'],
                  troughcolor=p['window'], bordercolor=p['border'])
  style.configure('TPanedwindow', background=p['window'])

  # The status bar, which wants to read as a strip rather than a panel.
  style.configure('Status.TLabel', background=p['window'],
                  foreground=p['muted'], padding=(8, 4))
  # A dimmer label for hints, so a note does not compete with a value.
  style.configure('Muted.TLabel', background=p['surface'],
                  foreground=p['muted'])
  style.configure('Heading.TLabel', background=p['surface'],
                  foreground=p['text'])
  return p


#: A `tk.Menu` is not a ttk widget and keeps the platform's own colours, so a
#: themed window ends up with one bright strip across the top. These are
#: passed to every menu the shell builds.
MENU_OPTIONS = dict(background=PALETTE['surface'], foreground=PALETTE['text'],
                    activebackground=PALETTE['accent'],
                    activeforeground=PALETTE['accent_text'],
                    borderwidth=0, relief='flat')


def style_figure(figure, axes=()) -> None:
  """Give a matplotlib figure the same ground as the window around it.

  Done per figure rather than through `rcParams`: the GUI must not change how
  a plot looks anywhere else in the library, and `Sequence.plot` and
  `MRIPlotter` both build their own figures outside it.
  """
  p = PALETTE
  figure.set_facecolor(p['surface'])
  for ax in axes:
    ax.set_facecolor(p['window'])
    for spine in ax.spines.values():
      spine.set_color(p['border'])
    ax.tick_params(colors=p['muted'], which='both')
    ax.xaxis.label.set_color(p['muted'])
    ax.yaxis.label.set_color(p['muted'])
    ax.title.set_color(p['text'])
