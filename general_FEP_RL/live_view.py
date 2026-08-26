"""
live_view.py - live per-step visualisation for any agent built on this architecture.

Call it once per step with the observation you handed the agent and the step_dict it
gave back:

    view = LiveView(layout = {'see_image' : 'channels'},
                    channel_names = {'see_image' : ['paddle', 'ball', 'trail', 'brick']},
                    discrete_actions = ['action'])
    ...
    step_dict = agent.step_in_episode(observation)
    view.update(observation, step_dict, text = f"epoch {e}, step {s}")

It works out what to draw from the tensors themselves, so it handles any number of
modalities, any mix of images and vectors, any channel count, and any number of action
heads. Nothing is declared up front; the figure is built on the first update.

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

THE ACTION ROW

Actions get their own row rather than a column, because there is nothing to compare
them against: an action is a prior INPUT, never a prediction, so it has no actual /
prior / posterior triple. One small bar chart per action head.

Two actions are in play at any step, one step apart, and the row shows both:

    filled bars     the action just chosen, which will drive the NEXT frame
    orange ticks    the action that drove the prior currently on screen

That offset is the whole reason both are drawn. The prior above came from the previous
hidden state and the PREVIOUS action, so if you want to ask "was that prediction a
reasonable thing to expect given what the agent did", the ticks are the action you
want, not the bars. LiveView remembers the previous action itself; you do not pass it.
Call view.begin() alongside agent.begin() to clear it between episodes -- forgetting
costs you one stale tick on step 0 and nothing else.

The y-axis is pinned to +/- action_limit (1.0 by default, the range of a tanh-squashed
action) rather than auto-scaled, for the same reason the error columns share a scale:
an action vector that has collapsed toward the origin should LOOK collapsed. Auto-
scaling would redraw a policy outputting +/-0.05 as though it were saturating. If your
actions are not tanh-bounded, set action_limit to their range, or None to auto-scale.

DISCRETE ACTIONS

Name any head in discrete_actions if you turn its vector into a choice with argmax.
That head then highlights its winning component and draws a dashed line at the runner-
up, so the top-two margin is visible as a gap, and the text panel prints that margin.
This is worth watching: a policy squeezed toward the origin can have a margin of a few
hundredths, which means the discrete choice reaching the environment is being decided
by noise even though the continuous vector looks like it carries an opinion.

THE BASELINE LINE

The text panel compares both predictions against the running average of every frame
seen so far. A world model that has learned nothing except which cells are usually on
will still score well on MSE when frames are sparse, and that trivial solution is easy
to mistake for progress. If 'vs baseline' is not comfortably below 1.00, the model is
not predicting, it is describing the average.

Note that the baseline is an MSE yardstick. If a modality is trained under a different
loss -- weighted BCE on sparse binary frames, say -- the optimal prediction under that
loss is deliberately not the MSE-optimal one, and a ratio slightly over 1.00 need not
mean the model has learned nothing. Read it alongside the training curve.

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


def strip_batch_vector(x):
    """Same, but never collapses past 1-D. A one-element action stays shape (1,)."""
    return np.atleast_1d(strip_batch(x)).ravel()


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
# Pulling predictions and actions out of a step_dict without the caller naming keys.
# ----------------------------------------------------------------------------
_PRIOR_KEYS = ('prior_predictions', 'list_of_prior_predictions', 'list_of_predictions')
_POSTERIOR_KEYS = ('posterior_predictions', 'list_of_posterior_predictions',
                   'list_of_predictions')
_ACTION_KEYS = ('action', 'actions', 'action_dict')


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


def _actions_from(step_dict):
    """step_dict['action'] is {head_name : (batch, episode, size)}."""
    for key in _ACTION_KEYS:
        value = step_dict.get(key)
        if isinstance(value, dict) and value:
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
            show_action = True,     # a row of bar charts, one per action head
            discrete_actions = None,    # names of heads you turn into a choice with
                                        # argmax; they get a highlighted winner and a
                                        # printed top-two margin
            action_limit = 1.0,     # symmetric y-range for the action bars. 1.0 suits a
                                    # tanh-squashed action. None = auto-scale, which
                                    # hides collapse toward the origin.
            action_labels = None,   # name -> list of labels, one per action component
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
        self.show_action = show_action
        self.discrete_actions = set(discrete_actions or ())
        self.action_limit = action_limit
        self.action_labels = dict(action_labels or {})
        self.max_panels = max_panels
        self.pause = pause
        self.title = title

        self.fig = None                 # built lazily, on the first update
        self._panels = []               # (row_label, modality, extractor, is_rgb, kind)
        self._images = {}
        self._bars = {}
        self._baseline_sum = {}
        self._baseline_count = 0

        self._action_names = []
        self._action_axes = {}
        self._action_bars = {}
        self._action_previous_marks = {}
        self._action_runner_up = {}
        self._previous_actions = {}     # the action that drove the prior on screen

    # ---- episode boundaries ----------------------------------------------

    def begin(self):
        """Call alongside agent.begin(). Only clears the remembered previous action,
        so that the first step of an episode does not show a tick left over from the
        last step of the one before."""
        self._previous_actions = {}

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

    def _build(self, observation, prior, posterior, actions):
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

        self._action_names = sorted(actions) if self.show_action else []
        has_actions = bool(self._action_names)

        height_ratios = [1] * rows
        if has_actions:
            height_ratios.append(0.62)
        height_ratios.append(0.5 + 0.12 * rows)

        plt.ion()
        self.fig = plt.figure(figsize = (2.1 * len(columns) + 0.6,
                                         2.0 * rows + 1.4 + (1.3 if has_actions else 0)))
        try:
            self.fig.canvas.manager.set_window_title(self.title)
        except Exception:
            pass

        grid = self.fig.add_gridspec(
            len(height_ratios), len(columns),
            height_ratios = height_ratios, hspace = 0.42, wspace = 0.18)

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

        # The action row: one small bar chart per head, side by side.
        self._action_axes = {}
        if has_actions:
            action_grid = grid[rows, :].subgridspec(
                1, len(self._action_names), wspace = 0.32)
            for index, name in enumerate(self._action_names):
                axes = self.fig.add_subplot(action_grid[0, index])
                axes.set_title(f"action: {name}", fontsize = 9, fontweight = "bold")
                axes.tick_params(labelsize = 7)
                self._action_axes[name] = axes

        self.text_axes = self.fig.add_subplot(grid[len(height_ratios) - 1, :])
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

    def _draw_action(self, name, current, previous):
        """Filled bars: the action just chosen. Orange ticks: the action that drove
        the prior on screen, one step earlier."""
        axes = self._action_axes[name]
        size = len(current)
        positions = np.arange(size)
        discrete = name in self.discrete_actions

        if name not in self._action_bars:
            self._action_bars[name] = axes.bar(
                positions, current, color = "0.55", zorder = 2)
            self._action_previous_marks[name] = axes.plot(
                positions, np.full(size, np.nan), linestyle = "none", marker = "_",
                markersize = 13, markeredgewidth = 2.0, color = "tab:orange",
                zorder = 4)[0]
            # Dashed line at the runner-up, so an argmax margin is a visible gap.
            self._action_runner_up[name] = axes.axhline(
                np.nan, color = "tab:red", lw = 1.0, linestyle = "--", zorder = 3)
            axes.axhline(0, color = "black", lw = 0.6, zorder = 1)
            labels = self.action_labels.get(name)
            axes.set_xticks(positions)
            axes.set_xticklabels(
                [str(labels[i]) if labels is not None and i < len(labels) else str(i)
                 for i in range(size)], fontsize = 7)
        else:
            for bar, height in zip(self._action_bars[name], current):
                bar.set_height(height)

        # Highlight the component that actually reaches the environment.
        if discrete and size >= 1:
            winner = int(np.argmax(current))
            for index, bar in enumerate(self._action_bars[name]):
                bar.set_color("tab:blue" if index == winner else "0.72")
            if size >= 2:
                runner_up = float(np.sort(current)[-2])
                self._action_runner_up[name].set_ydata([runner_up, runner_up])
            else:
                self._action_runner_up[name].set_ydata([np.nan, np.nan])
        else:
            for bar in self._action_bars[name]:
                bar.set_color("0.55")
            self._action_runner_up[name].set_ydata([np.nan, np.nan])

        if previous is not None and len(previous) == size:
            self._action_previous_marks[name].set_ydata(previous)
        else:
            self._action_previous_marks[name].set_ydata(np.full(size, np.nan))

        # Pinned, not auto-scaled: a collapsed action should look collapsed.
        limit = self.action_limit
        observed = float(np.abs(current).max()) if size else 0.0
        if previous is not None and len(previous):
            observed = max(observed, float(np.abs(previous).max()))
        limit = max(observed, 1e-6) if limit is None else max(limit, observed)
        axes.set_ylim(-1.08 * limit, 1.08 * limit)

    def _action_report(self, name, current, previous):
        pieces = [name]
        pieces.append(f"|a| {float(np.linalg.norm(current)):.3f}")
        if len(current) <= 4:
            pieces.append("[" + ", ".join(f"{value:+.2f}" for value in current) + "]")
        if name in self.discrete_actions and len(current) >= 2:
            ordered = np.sort(current)
            pieces.append(f"argmax {int(np.argmax(current))}")
            pieces.append(f"margin {float(ordered[-1] - ordered[-2]):.3f}")
        if previous is not None and len(previous):
            if name in self.discrete_actions and len(previous) >= 2:
                pieces.append(f"(prior above: argmax {int(np.argmax(previous))})")
            else:
                pieces.append(f"(prior above: |a| {float(np.linalg.norm(previous)):.3f})")
        return "  ".join(pieces)

    # ---- the call you make each step -------------------------------------

    def update(self, observation, step_dict = None, prior = None, posterior = None,
               actions = None, text = "", layer = None, new_episode = False):
        """
        observation   {name : tensor} exactly as handed to agent.step_in_episode
        step_dict     what step_in_episode returned; prior, posterior and actions are
                      read out of it. Pass them directly instead if you like.
        new_episode   same effect as calling begin() first
        text          anything you want under the panels
        """
        layer = self.layer if layer is None else layer
        if new_episode:
            self.begin()

        if step_dict is not None:
            prior = _predictions_from(step_dict, _PRIOR_KEYS, layer) if prior is None else prior
            posterior = (_predictions_from(step_dict, _POSTERIOR_KEYS, layer)
                         if posterior is None else posterior)
            actions = _actions_from(step_dict) if actions is None else actions
        prior = prior or {}
        posterior = posterior or {}
        actions = actions or {}

        if self.fig is None:
            self._build(observation, prior, posterior, actions)

        # Running mean of every frame seen: the trivial predictor to beat.
        if self.show_baseline:
            for name, value in observation.items():
                actual = strip_batch(value)
                self._baseline_sum[name] = self._baseline_sum.get(name, 0.0) + actual
            self._baseline_count += 1

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

        # The action row, and the action lines of the report.
        action_report = []
        for name in self._action_names:
            if name not in actions:
                continue
            current = strip_batch_vector(actions[name])
            previous = self._previous_actions.get(name)
            self._draw_action(name, current, previous)
            action_report.append(self._action_report(name, current, previous))
        for name in self._action_names:
            if name in actions:
                self._previous_actions[name] = strip_batch_vector(actions[name])

        # One line per modality: how each prediction does against the trivial baseline,
        # then how far the two predictions are from each other.
        report = []
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

        report += action_report

        message = text or ""
        if report:
            message = (message + "\n" + "\n".join(report)) if message else "\n".join(report)
        if self.show_error or self.show_disagreement:
            message += "\n(difference columns: mid-grey is agreement, white means the "
            message += "left term is higher, black lower; all share one scale)"
        if self.show_baseline and self._baseline_count:
            message += "\n(baseline = running mean frame; over 1.00x means the model is "
            message += "not beating a constant)"
        if self._action_names:
            message += "\n(action bars = just chosen, drives the NEXT frame; orange "
            message += "ticks = the action that drove the prior above)"
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

    print("A) MinAtar-style: 4 semantic channels, a vector modality, and a 6-way "
          "discrete action taken by argmax.")
    view = LiveView(
        layout = {'see_image' : 'channels'},
        channel_names = {'see_image' : ['paddle', 'ball', 'trail', 'brick']},
        discrete_actions = ['action'],
        title = "self-test: MinAtar")

    view.begin()
    for step in range(40):
        frame = (np.random.rand(1, 1, 10, 10, 4) < 0.08).astype("float32")
        observation = {'see_image' : frame, 'speed' : np.random.rand(1, 1, 3)}
        step_dict = {
            'prior_predictions' : [{'see_image' : np.random.rand(1, 1, 10, 10, 4) * 0.2,
                                    'speed' : np.random.rand(1, 1, 3)}],
            'posterior_predictions' : [{'see_image' : frame * 0.8 + 0.05,
                                        'speed' : np.random.rand(1, 1, 3)}],
            'action' : {'action' : np.tanh(np.random.randn(1, 1, 6) * 0.8)},
            'list_of_inner_states' : [{'see_image' : {'dkl' : np.array([0.03])},
                                       'speed' : {'dkl' : np.array([0.01])}}]}
        view.update(observation, step_dict, text = f"self-test step {step}")
        time.sleep(0.02)
    view.close()

    print("B) Maze-style: RGB plus depth, and a 2-d continuous action.")
    view = LiveView(layout = {'see_image' : 'rgbd'},
                    action_labels = {'make_velocity' : ['yaw', 'speed']},
                    title = "self-test: maze")
    view.begin()
    for step in range(20):
        frame = np.random.rand(1, 1, 8, 8, 4).astype("float32")
        observation = {'see_image' : frame}
        step_dict = {'prior_predictions' : [{'see_image' : np.random.rand(1, 1, 8, 8, 4)}],
                     'posterior_predictions' : [{'see_image' : frame * 0.9}],
                     'action' : {'make_velocity' : np.tanh(np.random.randn(1, 1, 2))}}
        view.update(observation, step_dict, text = f"self-test step {step}")
        time.sleep(0.02)
    view.close()

    print("C) A collapsed policy: the same 6-way action, but squeezed toward the "
          "origin. The bars nearly vanish and the argmax margin goes to noise.")
    view = LiveView(layout = {'see_image' : 'gray'}, discrete_actions = ['action'],
                    title = "self-test: collapsed action")
    view.begin()
    for step in range(20):
        frame = np.random.rand(1, 1, 6, 6, 1).astype("float32")
        observation = {'see_image' : frame}
        step_dict = {'prior_predictions' : [{'see_image' : frame * 0.9}],
                     'posterior_predictions' : [{'see_image' : frame * 0.95}],
                     'action' : {'action' : np.tanh(np.random.randn(1, 1, 6) * 0.09)}}
        view.update(observation, step_dict, text = f"self-test step {step}")
        time.sleep(0.02)

    print("self-test done - close the window to exit")
    plt.ioff()
    plt.show()