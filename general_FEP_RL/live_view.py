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

    actual      the observation the agent was given at this step
    prior       what it expected before looking, from the previous hidden state and
                the previous action
    posterior   what it says after looking
    error       prior minus actual, red high and blue low

Prior against actual is the honest test of the world model. Posterior against actual is
the easier one -- the posterior has already seen the frame -- so a posterior that looks
good while the prior looks like mush means the latent is carrying the observation but
the dynamics have not been learned.

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
            show_error = True,      # a fourth column: prior minus actual
            show_baseline = True,   # compare both predictions to a running mean frame
            max_panels = 12,        # refuse to build something unreadable
            pause = 0.001,
            title = "Agent - live view"):

        _ensure_interactive_backend()
        self.layout = dict(layout or {})
        self.channel_names = dict(channel_names or {})
        self.layer = layer
        self.show_error = show_error
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

        columns = ["actual", "prior", "posterior"] + (["prior error"] if self.show_error else [])
        self._columns = columns
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

    def _draw_image(self, row, column, data, is_rgb, diverging = False):
        axes = self._axes[(row, column)]
        key = (row, column)
        if key not in self._images:
            if is_rgb:
                self._images[key] = axes.imshow(data, interpolation = "nearest")
            elif diverging:
                limit = max(float(np.abs(data).max()), 1e-6)
                self._images[key] = axes.imshow(
                    data, interpolation = "nearest", cmap = "bwr",
                    vmin = -limit, vmax = limit)
            else:
                self._images[key] = axes.imshow(
                    data, interpolation = "nearest", cmap = "gray",
                    vmin = 0.0, vmax = 1.0)
        else:
            self._images[key].set_data(data)
            if diverging:
                limit = max(float(np.abs(data).max()), 1e-6)
                self._images[key].set_clim(-limit, limit)

    def _draw_vector(self, row, column, data, diverging = False):
        axes = self._axes[(row, column)]
        key = (row, column)
        data = np.atleast_1d(data).ravel()
        if key not in self._bars:
            self._bars[key] = axes.bar(np.arange(len(data)), data,
                                       color = "tab:red" if diverging else "tab:blue")
            axes.axhline(0, color = "black", lw = 0.6)
        else:
            for bar, height in zip(self._bars[key], data):
                bar.set_height(height)
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

            if kind == 'image':
                actual_image = as_image(actual)
                for column, column_name in enumerate(self._columns):
                    if column_name == "prior error":
                        if frames['prior'] is None:
                            continue
                        difference = as_image(frames['prior']) - actual_image
                        self._draw_image(row, column, extractor(difference), False,
                                         diverging = True)
                    else:
                        source = frames[column_name]
                        if source is None:
                            continue
                        self._draw_image(row, column, extractor(as_image(source)), is_rgb)
            else:
                for column, column_name in enumerate(self._columns):
                    if column_name == "prior error":
                        if frames['prior'] is None:
                            continue
                        self._draw_vector(row, column, frames['prior'] - actual,
                                          diverging = True)
                    else:
                        source = frames[column_name]
                        if source is None:
                            continue
                        self._draw_vector(row, column, source)

        # One line per modality: how each prediction does against the trivial baseline.
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
            dkl = _dkl_from(step_dict, layer, name) if step_dict is not None else None
            if dkl is not None:
                pieces.append(f"dkl {dkl:.4f}")
            report.append("  ".join(pieces))

        message = text or ""
        if report:
            message = (message + "\n" + "\n".join(report)) if message else "\n".join(report)
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

    print("self-test done - close the window to exit")
    plt.ioff()
    plt.show()