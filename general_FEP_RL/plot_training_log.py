import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Defensive helpers: a missing log key skips a line instead of crashing.
# ---------------------------------------------------------------------------

def _line(ax, x, y, label=None, **kw):
    if y is None or len(y) == 0:
        return False
    n = min(len(x), len(y))
    ax.plot(x[:n], y[:n], label=label, **kw)
    return True


def _lines_from_dict(ax, x, dct, prefix="", **kw):
    if not dct:
        return
    for k, y in dct.items():
        _line(ax, x, y, label=f"{prefix}{k}", **kw)


def _lines_from_list(ax, x, lst, prefix="", **kw):
    if not lst:
        return
    for i, y in enumerate(lst):
        _line(ax, x, y, label=f"{prefix}{i}", **kw)


# The world model's per-source logs are now nested layer -> name -> series, because
# every layer has its own set of inner states and one of them may be named
# lower_layer_posterior_sample. Colour is by source, dash style is by layer, so the
# same modality stays recognisable across layers.
_DASHES = ["-", "--", ":", "-."]


def _source_style(sources):
    palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    return {name: palette[i % len(palette)] for i, name in enumerate(sorted(sources))}


def _all_sources(nested):
    return {name for layer in (nested or {}).values() for name in layer}


def _lines_from_nested(ax, x, nested, style=None, prefix="", label_layer=True, **kw):
    if not nested:
        return
    style = style or _source_style(_all_sources(nested))
    for layer_index, (layer_key, by_name) in enumerate(sorted(nested.items())):
        for name, y in by_name.items():
            label = f"{prefix}{name}"
            if label_layer:
                label = f"{prefix}{layer_key.replace('layer_', 'L')} {name}"
            _line(ax, x, y, label=label,
                  color=style.get(name), ls=_DASHES[layer_index % len(_DASHES)], **kw)


def _finish(ax, title, xlabel="Epoch", ylabel="Value", log=False, ylim=None):
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    # log = "auto" means: only worth it when the series actually spans decades.
    if log == "auto":
        values = [v for line in ax.get_lines() for v in line.get_ydata()
                  if v is not None and v > 0]
        log = bool(values) and max(values) / max(min(values), 1e-12) > 5
    if log:
        ax.set_yscale("log")
    else:
        ax.axhline(0, color="black", lw=0.6, alpha=0.4)
    if ylim:
        ax.set_ylim(*ylim)
    h, _ = ax.get_legend_handles_labels()
    if h:
        ax.legend(fontsize=7, loc="best")


def _discounted_horizon(gamma, horizon):
    if gamma is None or horizon is None:
        return None
    if gamma >= 1.0:
        return float(horizon)
    return (1.0 - gamma ** horizon) / (1.0 - gamma)


def _last(seq):
    return seq[-1] if seq else None


def _flatten_last(nested):
    """layer -> name -> series  becomes  'L0 touch' -> final value."""
    out = {}
    for layer_key, by_name in (nested or {}).items():
        for name, series in by_name.items():
            if series:
                out[f"{layer_key.replace('layer_', 'L')} {name}"] = series[-1]
    return out


# ---------------------------------------------------------------------------

def plot_training_log(agent, figsize=(19, 17)):
    tl = agent.training_log
    tla = agent.training_log_actor

    x = tl.get("epoch_num", [])
    xa = tla.get("epoch_num", [])

    gamma = getattr(agent, "gamma", None)
    horizon = getattr(getattr(agent, "buffer", None), "max_episode_len", None)
    horizon_factor = _discounted_horizon(gamma, horizon)

    source_style = _source_style(
        _all_sources(tl.get("accuracy_losses_prior")) | _all_sources(tl.get("curiosities")))

    fig, axs = plt.subplots(4, 3, figsize=figsize)
    axs = axs.flatten()

    # --- Row 1: the world model ------------------------------------------------

    # 1. Accuracy, split by layer and inner state.
    ax = axs[0]
    _lines_from_nested(ax, x, tl.get("accuracy_losses_prior"), source_style, lw=1.8)
    _lines_from_nested(ax, x, tl.get("accuracy_losses_posterior"), source_style, prefix="post ", alpha=0.35, lw=1.0)
    _finish(ax, "World model: accuracy  (bold = prior, faint = posterior)", log="auto")

    # 2. Complexity, on its own axis. Accuracy and complexity now live on very
    #    different scales, so sharing one panel hides whichever is smaller.
    ax = axs[1]
    _lines_from_nested(ax, x, tl.get("complexity_losses"), source_style)
    _finish(ax, "World model: complexity (DKL, before beta)")

    # 3. Inner-state sigma. This is the panel to look at first. Accuracy has an
    #    incentive to shrink prior_std, and prior_std is the denominator of the
    #    complexity term, so a collapsing line here poisons every curiosity value
    #    downstream. Prior is solid-ish, posterior is faint.
    ax = axs[2]
    _lines_from_nested(ax, x, tl.get("prior_stds"), source_style, lw=1.8)
    _lines_from_nested(ax, x, tl.get("posterior_stds"), source_style,
                       prefix="post ", alpha=0.35, lw=1.0)
    _finish(ax, "Inner-state sigma  (bold = prior, faint = posterior)", log="auto")

    # --- Row 2: curiosity and reward -------------------------------------------

    # 4. Curiosity by source, after clipping and eta.
    ax = axs[3]
    _lines_from_nested(ax, x, tl.get("curiosities"), source_style)
    _finish(ax, "Curiosity by source (after eta and clipping)")

    # 5. How much of each source is pinned at the clip ceiling. A source near 1.0
    #    reports the same number for a mild surprise and a total one, so it has
    #    stopped discriminating no matter how large its raw DKL is.
    ax = axs[4]
    _lines_from_nested(ax, x, tl.get("curiosity_saturations"), source_style)
    _finish(ax, "Curiosity pinned at the clip ceiling (1.0 = no signal left)",
            ylabel="Fraction", ylim=(-0.02, 1.02))

    # 6. Reward composition.
    ax = axs[5]
    ent_net = tl.get("entropy_target_critic")
    total_rew = tl.get("total_reward")
    _line(ax, x, tl.get("average_reward"), label="extrinsic reward", color="tab:blue")
    _line(ax, x, tl.get("curiosity"), label="curiosity (all sources)",
          color="tab:gray", alpha=0.7)
    _line(ax, x, tl.get("sac_entropy_target_critic"), label="SAC entropy (critic)",
          color="tab:red", lw=2.2)
    _line(ax, x, tl.get("normal_entropy_target_critic"), label="normal prior (critic)",
          color="tab:orange", ls=":", lw=2)
    if total_rew is not None and ent_net is not None:
        n = min(len(total_rew), len(ent_net))
        _line(ax, x, [total_rew[i] + ent_net[i] for i in range(n)],
              label="effective value/step", color="black", ls="--", lw=1.4)
    _finish(ax, "Reward composition  —  what the critic sums each step")

    # --- Row 3: the critic -----------------------------------------------------

    # 7. Q build-up.
    ax = axs[6]
    _line(ax, x, total_rew, label="immediate: total_reward", color="tab:blue")
    _line(ax, x, tl.get("future_Q_value"), label="bootstrapped future",
          color="tab:purple")
    _line(ax, x, tl.get("Q_target"), label="= Q target", color="black", ls="--", lw=1.4)
    _finish(ax, "Q build-up: immediate reward vs bootstrap")

    # 8. Critic value against the value entropy alone would imply. If the dashed
    #    line accounts for most of the critic's output, the critic is mostly
    #    regressing on its own entropy bonus rather than on the task.
    ax = axs[7]
    _line(ax, x, tl.get("Q_target"), label="Q target", color="tab:blue", lw=2)
    _lines_from_list(ax, x, tl.get("critic_predictions"), prefix="critic ", alpha=0.8)
    _line(ax, x, tl.get("target_critic_output"), label="bootstrap Q(t+1)",
          color="tab:green", alpha=0.7)
    if ent_net is not None and horizon_factor is not None:
        _line(ax, x, [e * horizon_factor for e in ent_net],
              label="entropy-only Q (implied)", color="tab:red", ls="--", lw=2)
    title = "Critic value vs entropy-implied value"
    if gamma is not None and horizon is not None:
        title += f"  (gamma={gamma:g}, H={horizon})"
    _finish(ax, title)

    # 9. Critic TD loss.
    ax = axs[8]
    _lines_from_list(ax, x, tl.get("critic_losses"), prefix="critic ")
    _finish(ax, "Critic TD loss")

    # --- Row 4: the actor ------------------------------------------------------

    # 10. Actor objective terms.
    ax = axs[9]
    _line(ax, xa, tla.get("Q_for_actor"), label="Q for actor")
    _line(ax, xa, tla.get("entropy_for_actor"), label="entropy for actor")
    _line(ax, xa, tla.get("total_imitation_loss"), label="imitation")
    _line(ax, xa, tla.get("actor_loss"), label="actor loss", color="tab:red")
    _finish(ax, "Actor objective terms")

    # 11. Policy entropy against its target, per action part. target_entropy does not
    #     scale with an action part's width by default, so a wide part and a narrow
    #     part chase the same number and the wide one is asked for far more
    #     compression. Gaps that never close are the symptom.
    ax = axs[10]
    entropies = tla.get("entropies")
    targets = tla.get("target_entropies")
    action_style = _source_style(set(entropies or {}))
    for name, series in (entropies or {}).items():
        _line(ax, xa, series, label=f"entropy {name}", color=action_style.get(name), lw=1.8)
    for name, series in (targets or {}).items():
        if series:
            ax.axhline(series[-1], color=action_style.get(name), ls="--", lw=1.2,
                       alpha=0.7)
    _finish(ax, "Policy entropy vs target (dashed = target_entropy)")

    # 12. Alpha, log scale: it moves multiplicatively.
    ax = axs[11]
    _lines_from_dict(ax, xa, tla.get("alphas"), prefix="alpha ")
    _finish(ax, "Entropy temperature (alpha)", log="auto")

    # --- Footer: the things worth catching early -------------------------------

    bits, warnings = [], []

    stds = _flatten_last(tl.get("prior_stds"))
    if stds:
        worst, value = min(stds.items(), key=lambda kv: kv[1])
        bits.append(f"lowest prior_std: {worst} = {value:.3f}")
        if value < 0.05:
            warnings.append(
                f"{worst}'s prior_std has collapsed ({value:.3f}); it is the "
                "denominator of that source's DKL, so its curiosity is inflated.")

    sat = _flatten_last(tl.get("curiosity_saturations"))
    if sat:
        worst, value = max(sat.items(), key=lambda kv: kv[1])
        bits.append(f"most saturated curiosity: {worst} at {value:.0%}")
        if value > 0.5:
            warnings.append(
                f"{worst} is pinned at the clip ceiling {value:.0%} of the time; "
                "lower its eta_before_clamp or that source has no usable signal.")

    ent_last, rew_last = _last(ent_net), _last(total_rew)
    if ent_last is not None and rew_last is not None:
        denom = abs(ent_last) + abs(rew_last) + 1e-8
        share = 100 * abs(ent_last) / denom
        bits.append(f"entropy is {share:.0f}% of per-step value")
        if share > 80:
            warnings.append(
                "the critic is mostly regressing on its own entropy bonus; "
                "check target_entropy against each action part's width.")

    dead = [k for k, v in _flatten_last(tl.get("curiosities")).items() if abs(v) < 1e-6]
    if dead:
        warnings.append(
            f"contributing no curiosity at all: {', '.join(dead)} "
            "(prior and posterior agree; that source's latent may have collapsed).")

    footer = "Latest:  " + "   |   ".join(bits) if bits else ""
    if warnings:
        footer += "\n" + "\n".join("!  " + w for w in warnings)
    if footer:
        fig.text(0.5, 0.005, footer, ha="center", va="bottom", fontsize=9,
                 family="monospace")

    fig.tight_layout(rect=(0, 0.055 if warnings else 0.02, 1, 1))
    return fig