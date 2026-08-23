"""
live_view.py - live per-step visualisation for any agent built on this architecture.

Call it once per step with the observation you handed the agent and the step_dict it
gave back:

    view = LiveView(layout = {'see_image' : 'channels'},
                    channel_names = {'see_image' : ['paddle', 'ball', 'trail', 'brick']})
    ...
    step_dict = agent.step_in_episode(observation)
    view.update(observation, step_dict, text = f"epoch {e}, step {s}")

It works out what to draw from the tensors themselves, so it handles any number of
modalities, any mix of images and vectors, and any channel count. Nothing is declared
up front; the figure is built on the first update.

WHAT THE COLUMNS MEAN

    actual            the observation the agent was given at this step
    prior             what it expected before looking, from the previous hidden
                      state and the previous action
    posterior         what it says after looking
    prior error       prior minus actual
    posterior error   posterior minus actual
    prior vs post     prior minus posterior

All three difference columns are symmetric grey on ONE SHARED scale: mid-grey is
agreement, white means the left term is higher, black means it is lower. Sharing the
scale is the point -- auto-scaling each column to its own maximum would make a
posterior with tiny errors look every bit as dramatic as a prior with large ones,
which is exactly the comparison this view exists to make. Pass error_limit = 1.0 to
pin the scale across steps as well, for observations already bounded to [0, 1].

Prior against actual is the honest test of the world model. Posterior against actual is
the easier one -- the posterior has already seen the frame -- so a posterior that looks
good while the prior looks like mush means the latent is carrying the observation but
the dynamics have not been learned.

THE DISAGREEMENT COLUMN

prior minus posterior is exactly (prior error) minus (posterior error), which is why it
belongs on the same scale as the other two. It is the observation-space picture of the
dkl already printed in the text panel: the posterior has seen the frame and the prior
has not, so wherever the two disagree is what the agent gained by looking. That is the
quantity curiosity is paid for, drawn where you can see WHICH parts of the observation
carried the information rather than only how much of it there was.

When the posterior is good this column looks almost identical to the prior error column,
and that is the identity above rather than a bug. It earns its keep in the opposite
case: if both error columns look bad but this one is flat grey, prior and posterior are
wrong TOGETHER, which points at the decoder or at observation scaling rather than at the
dynamics. Divergence between them is the dynamics; shared error is everything
downstream of the latent.

A trained model should drive this column dark almost everywhere, staying bright only
where the frame was genuinely unpredictable. Dark everywhere INCLUDING at surprising
events means the posterior has stopped extracting anything from the observation, and
the curiosity signal is about to flatline -- which should show up in the same text line
as dkl collapsing.

THE BASELINE LINE

The text panel compares both predictions against the running average of every frame
seen so far. A world model that has learned nothing except which cells are usually on
will still score well on MSE when frames are sparse, and that trivial solution is easy
to mistake for progress. If 'vs baseline' is not comfortably below 1.00, the model is
not predicting, it is describing the average.

Quick self-test, no agent needed:
    python live_view.py
"""

from __future__ import annotations
import numbers

import numpy as np
import matplotlib
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------------
# Backend: a live, separate window needs an interactive matplotlib backend.
# ----------------------------------------------------------------------------
_INTERACTIVE = {
    "qtagg", "qt5agg", "qt4agg", "tkagg", "macosx",
    "gtk3agg", "gtk4agg", "wxagg", "nbagg", "webagg",
}


def _ensure_interactive_backend(verbose = True):
    if matplotlib.get_backend().lower() in _INTERACTIVE:
        return matplotlib.get_backend()
    for candidate in ("QtAgg", "Qt5Agg", "TkAgg"):
        try:
            plt.switch_backend(candidate)
            if verbose:
                print(f"[live_view] switched matplotlib backend to {candidate}")
            return candidate
        except Exception:
            continue
    if verbose:
        print(
            f"[live_view] WARNING: backend {matplotlib.get_backend()!r} may not show a "
            "live window.\n"
            "  - In a VS Code Interactive / Jupyter window, run:  %matplotlib qt\n"
            "  - Or put  matplotlib.use('QtAgg')  at the very TOP of main.py,\n"
            "    before 'import matplotlib.pyplot as plt'.")
    return matplotlib.get_backend()


# ----------------------------------------------------------------------------
# Array helpers. torch is imported lazily, so this file has no hard dependency.
# ----------------------------------------------------------------------------
def to_numpy(x):
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().to("cpu").float().numpy()
    except Exception:
        pass
    return np.asarray(x, dtype = np.float32)


def strip_batch(x):
    """(batch, episode, ...) -> (...), taking the first of each leading dimension."""
    a = to_numpy(x)
    while a.ndim > 0 and a.shape[0] == 1:
        a = a[0]
    return a


def as_image(a):
    """(H, W, C) channels-last, or None if this is not an image."""
    if a.ndim == 3:
        if a.shape[-1] <= 8:
            return a
        if a.shape[0] <= 8:                      # channels-first
            return np.transpose(a, (1, 2, 0))
    if a.ndim == 2:
        return a[..., None]
    return None


def describe(obj, _depth = 0, _max_depth = 4):
    """Print the nested structure of e.g. a step_dict, to find a field by eye."""
    pad = "  " * _depth
    if _depth > _max_depth:
        print(pad + "...")
        return
    if isinstance(obj, dict):
        for key, value in obj.items():
            print(f"{pad}{key}:")
            describe(value, _depth + 1, _max_depth)
    elif isinstance(obj, (list, tuple)):
        print(f"{pad}{type(obj).__name__}[{len(obj)}]")
        if len(obj):
            describe(obj[0], _depth + 1, _max_depth)
    else:
        shape = getattr(obj, "shape", None)
        if shape is not None:
            print(f"{pad}<{type(obj).__name__} shape={tuple(shape)}>")
        elif isinstance(obj, numbers.Number):
            print(f"{pad}{obj}")
        else:
            print(f"{pad}<{type(obj).__name__}>")


# ----------------------------------------------------------------------------
# Pulling predictions out of a step_dict without the caller naming keys.
# ----------------------------------------------------------------------------
_PRIOR_KEYS = ('prior_predictions', 'list_of_prior_predictions', 'list_of_predictions')
_POSTERIOR_KEYS = ('posterior_predictions', 'list_of_posterior_predictions',
                   'list_of_predictions')


# Diverging columns: name -> (left, right), both keys into the per-row `frames` dict.
# Each is drawn as left minus right on the shared symmetric grey scale. Note that
# ('prior', 'posterior') is the difference of the two rows above it, which is why one
# shared limit across all three is the honest choice.
_ERROR_COLUMNS = {
    "prior error"     : ("prior", "actual"),
    "posterior error" : ("posterior", "actual"),
    "prior vs post"   : ("prior", "posterior"),
}


def _predictions_from(step_dict, keys, layer):
    for key in keys:
        if key in step_dict:
            value = step_dict[key]
            if isinstance(value, (list, tuple)):
                return value[layer] if layer < len(value) else {}
            return value
    return {}


def _dkl_from(step_dict, layer, name):
    inner = step_dict.get('list_of_inner_states')
    if not inner or layer >= len(inner):
        return None
    entry = inner[layer].get(name)
    if entry is None or 'dkl' not in entry:
        return None
    return float(to_numpy(entry['dkl']).mean())


# ----------------------------------------------------------------------------
# The live window.
# ----------------------------------------------------------------------------
class LiveView:

    def __init__(
            self,
            layout = None,          # name -> 'channels' | 'rgbd' | 'rgb' | 'gray' | 'auto'
            channel_names = None,   # name -> list of labels, one per channel
            layer = 0,              # which world model layer's predictions to show
            show_error = True,      # two more columns: prior and posterior minus actual
            show_disagreement = True,   # one more column: prior minus posterior
            error_limit = None,     # symmetric half-range for the difference columns.
                                    # None = one auto limit per row, shared by all.
            show_baseline = True,   # compare both predictions to a running mean frame
            max_panels = 12,        # refuse to build something unreadable
            pause = 0.001,
            title = "Agent - live view"):

        _ensure_interactive_backend()
        self.layout = dict(layout or {})
        self.channel_names = dict(channel_names or {})
        self.layer = layer
        self.show_error = show_error
        self.show_disagreement = show_disagreement
        self.error_limit = error_limit
        self.show_baseline = show_baseline
        self.max_panels = max_panels
        self.pause = pause
        self.title = title

        self.fig = None                 # built lazily, on the first update
        self._panels = []               # (row_label, modality, extractor, is_rgb)
        self._images = {}
        self._bars = {}
        self._baseline_sum = {}
        self._baseline_count = 0

    # ---- deciding what to draw -------------------------------------------

    def _panels_for(self, name, image):
        mode = self.layout.get(name, 'auto')
        channels = image.shape[-1]

        if mode == 'auto':
            mode = {1 : 'gray', 3 : 'rgb'}.get(channels, 'channels')

        if mode == 'rgbd' and channels >= 4:
            return [(f"{name}: RGB", lambda a: np.clip(a[..., :3], 0, 1), True),
                    (f"{name}: depth", lambda a: a[..., 3], False)]
        if mode == 'rgb' and channels >= 3:
            return [(f"{name}: RGB", lambda a: np.clip(a[..., :3], 0, 1), True)]
        if mode == 'gray' or channels == 1:
            return [(name, lambda a: a[..., 0], False)]

        labels = self.channel_names.get(name) or [f"ch {i}" for i in range(channels)]
        panels = []
        for index in range(channels):
            label = labels[index] if index < len(labels) else f"ch {index}"
            panels.append((f"{name}: {label}",
                           (lambda i: (lambda a: a[..., i]))(index),
                           False))
        return panels

    def _build(self, observation, prior, posterior):
        # One row per panel; images get their own rows, vectors get one row each.
        self._panels = []
        for name in sorted(observation):
            actual = strip_batch(observation[name])
            image = as_image(actual)
            if image is not None:
                for label, extractor, is_rgb in self._panels_for(name, image):
                    self._panels.append((label, name, extractor, is_rgb, 'image'))
            else:
                self._panels.append((name, name, lambda a: a, False, 'vector'))

        if len(self._panels) > self.max_panels:
            raise ValueError(
                f"live_view would need {len(self._panels)} rows, over max_panels="
                f"{self.max_panels}. Narrow it with layout=, e.g. "
                f"{{'see_image' : 'rgb'}}, or raise max_panels.")

        columns = ["actual", "prior", "posterior"]
        if self.show_error:
            columns += ["prior error", "posterior error"]
        if self.show_disagreement:
            columns += ["prior vs post"]
        self._columns = columns
        self._error_columns = {name : _ERROR_COLUMNS[name]
                               for name in columns if name in _ERROR_COLUMNS}
        rows = len(self._panels)

        plt.ion()
        self.fig = plt.figure(figsize = (2.1 * len(columns) + 0.6, 2.0 * rows + 1.4))
        try:
            self.fig.canvas.manager.set_window_title(self.title)
        except Exception:
            pass

        grid = self.fig.add_gridspec(
            rows + 1, len(columns),
            height_ratios = [1] * rows + [0.5 + 0.12 * rows],
            hspace = 0.42, wspace = 0.18)

        self._axes = {}
        for row, (label, _, _, _, kind) in enumerate(self._panels):
            for column, column_name in enumerate(columns):
                axes = self.fig.add_subplot(grid[row, column])
                if row == 0:
                    axes.set_title(column_name, fontsize = 10, fontweight = "bold")
                if column == 0:
                    axes.set_ylabel(label, fontsize = 9)
                if kind == 'image':
                    axes.set_xticks([])
                    axes.set_yticks([])
                else:
                    axes.tick_params(labelsize = 7)
                self._axes[(row, column)] = axes

        self.text_axes = self.fig.add_subplot(grid[rows, :])
        self.text_axes.axis("off")
        self._text = self.text_axes.text(
            0.0, 0.95, "", va = "top", ha = "left", fontsize = 10,
            family = "monospace", transform = self.text_axes.transAxes)

        self.fig.show()

    # ---- drawing ---------------------------------------------------------

    def _draw_image(self, row, column, data, is_rgb, diverging = False, limit = None):
        axes = self._axes[(row, column)]
        key = (row, column)
        if diverging and limit is None:
            limit = max(float(np.abs(data).max()), 1e-6)
        if key not in self._images:
            if is_rgb:
                self._images[key] = axes.imshow(data, interpolation = "nearest")
            elif diverging:
                # Symmetric grey, so agreement lands on mid-grey rather than on an
                # end of the colour map. The left term being higher goes white,
                # lower goes black.
                self._images[key] = axes.imshow(
                    data, interpolation = "nearest", cmap = "gray",
                    vmin = -limit, vmax = limit)
            else:
                self._images[key] = axes.imshow(
                    data, interpolation = "nearest", cmap = "gray",
                    vmin = 0.0, vmax = 1.0)
        else:
            self._images[key].set_data(data)
            if diverging:
                self._images[key].set_clim(-limit, limit)

    def _draw_vector(self, row, column, data, diverging = False, limit = None):
        axes = self._axes[(row, column)]
        key = (row, column)
        data = np.atleast_1d(data).ravel()
        if key not in self._bars:
            self._bars[key] = axes.bar(np.arange(len(data)), data,
                                       color = "0.45" if diverging else "tab:blue")
            axes.axhline(0, color = "black", lw = 0.6)
        else:
            for bar, height in zip(self._bars[key], data):
                bar.set_height(height)
        if diverging:
            # Same shared scale the image difference panels use, for the same reason.
            if limit is None:
                limit = max(float(np.abs(data).max()), 1e-6)
            axes.set_ylim(-1.1 * limit, 1.1 * limit)
        else:
            low, high = float(np.min(data)), float(np.max(data))
            pad = max(0.1, 0.2 * (high - low))
            axes.set_ylim(min(low, 0) - pad, max(high, 0) + pad)

    # ---- the call you make each step -------------------------------------

    def update(self, observation, step_dict = None, prior = None, posterior = None,
               text = "", layer = None):
        """
        observation   {name : tensor} exactly as handed to agent.step_in_episode
        step_dict     what step_in_episode returned; prior and posterior are read
                      out of it. Pass prior=/posterior= directly instead if you like.
        text          anything you want under the panels
        """
        layer = self.layer if layer is None else layer

        if step_dict is not None:
            prior = _predictions_from(step_dict, _PRIOR_KEYS, layer) if prior is None else prior
            posterior = (_predictions_from(step_dict, _POSTERIOR_KEYS, layer)
                         if posterior is None else posterior)
        prior = prior or {}
        posterior = posterior or {}

        if self.fig is None:
            self._build(observation, prior, posterior)

        # Running mean of every frame seen: the trivial predictor to beat.
        if self.show_baseline:
            for name, value in observation.items():
                actual = strip_batch(value)
                self._baseline_sum[name] = self._baseline_sum.get(name, 0.0) + actual
            self._baseline_count += 1

        report = []
        for row, (label, name, extractor, is_rgb, kind) in enumerate(self._panels):
            actual = strip_batch(observation[name])
            frames = {'actual' : actual}
            frames['prior'] = strip_batch(prior[name]) if name in prior else None
            frames['posterior'] = strip_batch(posterior[name]) if name in posterior else None

            # Image rows take every difference in channels-last space, because that is
            # what `extractor` slices. as_image returns None on a shape mismatch, which
            # the None checks below then skip instead of crashing.
            source_frames = ({key : (None if value is None else as_image(value))
                              for key, value in frames.items()}
                             if kind == 'image' else frames)

            differences = {}
            for column_name, (left, right) in self._error_columns.items():
                if source_frames[left] is None or source_frames[right] is None:
                    differences[column_name] = None
                else:
                    differences[column_name] = source_frames[left] - source_frames[right]

            # One limit across every diverging column. prior error minus posterior error
            # IS prior vs post, so the three genuinely live on one scale.
            limit = self.error_limit
            if limit is None:
                limit = max([float(np.abs(d).max())
                             for d in differences.values() if d is not None] + [1e-6])

            for column, column_name in enumerate(self._columns):
                if column_name in self._error_columns:
                    difference = differences[column_name]
                    if difference is None:
                        continue
                    if kind == 'image':
                        self._draw_image(row, column, extractor(difference), False,
                                         diverging = True, limit = limit)
                    else:
                        self._draw_vector(row, column, difference,
                                          diverging = True, limit = limit)
                else:
                    source = source_frames[column_name]
                    if source is None:
                        continue
                    if kind == 'image':
                        self._draw_image(row, column, extractor(source), is_rgb)
                    else:
                        self._draw_vector(row, column, source)

        # One line per modality: how each prediction does against the trivial baseline,
        # then how far the two predictions are from each other.
        for name in sorted(observation):
            actual = strip_batch(observation[name])
            pieces = [f"{name}"]
            baseline = None
            if self.show_baseline and self._baseline_count:
                mean_frame = self._baseline_sum[name] / self._baseline_count
                baseline = float(np.mean((mean_frame - actual) ** 2))
            for which, source in (("prior", prior.get(name)), ("post", posterior.get(name))):
                if source is None:
                    continue
                error = float(np.mean((strip_batch(source) - actual) ** 2))
                piece = f"{which} mse {error:.5f}"
                if baseline:
                    piece += f" ({error / baseline:.2f}x baseline)"
                pieces.append(piece)
            if prior.get(name) is not None and posterior.get(name) is not None:
                disagreement = float(np.mean(
                    (strip_batch(prior[name]) - strip_batch(posterior[name])) ** 2))
                pieces.append(f"p-vs-p {disagreement:.5f}")
            dkl = _dkl_from(step_dict, layer, name) if step_dict is not None else None
            if dkl is not None:
                pieces.append(f"dkl {dkl:.4f}")
            report.append("  ".join(pieces))

        message = text or ""
        if report:
            message = (message + "\n" + "\n".join(report)) if message else "\n".join(report)
        if self.show_error or self.show_disagreement:
            message += "\n(difference columns: mid-grey is agreement, white means the "
            message += "left term is higher, black lower; all share one scale)"
        if self.show_baseline and self._baseline_count:
            message += "\n(baseline = running mean frame; over 1.00x means the model is "
            message += "not beating a constant)"
        self._text.set_text(message)

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        if self.pause:
            plt.pause(self.pause)

    def close(self):
        try:
            plt.close(self.fig)
        except Exception:
            pass


# ----------------------------------------------------------------------------
# Standalone self-test with fake data, no agent needed.
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    import time

    print("A) MinAtar-style: 4 semantic channels, one row each, plus a vector modality.")
    view = LiveView(
        layout = {'see_image' : 'channels'},
        channel_names = {'see_image' : ['paddle', 'ball', 'trail', 'brick']},
        title = "self-test: MinAtar")

    for step in range(40):
        frame = (np.random.rand(1, 1, 10, 10, 4) < 0.08).astype("float32")
        observation = {'see_image' : frame, 'speed' : np.random.rand(1, 1, 3)}
        step_dict = {
            'prior_predictions' : [{'see_image' : np.random.rand(1, 1, 10, 10, 4) * 0.2,
                                    'speed' : np.random.rand(1, 1, 3)}],
            'posterior_predictions' : [{'see_image' : frame * 0.8 + 0.05,
                                        'speed' : np.random.rand(1, 1, 3)}],
            'list_of_inner_states' : [{'see_image' : {'dkl' : np.array([0.03])},
                                       'speed' : {'dkl' : np.array([0.01])}}]}
        view.update(observation, step_dict, text = f"self-test step {step}")
        time.sleep(0.02)
    view.close()

    print("B) Maze-style: RGB plus a depth channel.")
    view = LiveView(layout = {'see_image' : 'rgbd'}, title = "self-test: maze")
    for step in range(20):
        frame = np.random.rand(1, 1, 8, 8, 4).astype("float32")
        observation = {'see_image' : frame}
        step_dict = {'prior_predictions' : [{'see_image' : np.random.rand(1, 1, 8, 8, 4)}],
                     'posterior_predictions' : [{'see_image' : frame * 0.9}]}
        view.update(observation, step_dict, text = f"self-test step {step}")
        time.sleep(0.02)

    print("C) Disagreement column with a deliberately blind posterior: both error "
          "columns look bad, 'prior vs post' stays flat.")
    view = LiveView(layout = {'see_image' : 'gray'}, title = "self-test: shared error")
    for step in range(20):
        frame = np.random.rand(1, 1, 6, 6, 1).astype("float32")
        wrong_together = np.full((1, 1, 6, 6, 1), 0.5, dtype = "float32")
        observation = {'see_image' : frame}
        step_dict = {'prior_predictions' : [{'see_image' : wrong_together}],
                     'posterior_predictions' : [{'see_image' : wrong_together}]}
        view.update(observation, step_dict, text = f"self-test step {step}")
        time.sleep(0.02)

    print("self-test done - close the window to exit")
    plt.ioff()
    plt.show()