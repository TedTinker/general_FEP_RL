import numpy as np
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


def _finish(ax, title, xlabel="Epoch", ylabel="Value"):
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="black", lw=0.6, alpha=0.4)
    h, _ = ax.get_legend_handles_labels()
    if h:
        ax.legend(fontsize=8, loc="best")


def _discounted_horizon(gamma, horizon):
    if gamma is None or horizon is None:
        return None
    if gamma >= 1.0:
        return float(horizon)
    return (1.0 - gamma ** horizon) / (1.0 - gamma)


def _np(t):
    """Logged tensors are torch on CPU; fall back to np.asarray otherwise."""
    if hasattr(t, "numpy"):
        return t.numpy()
    return np.asarray(t)


# ---------------------------------------------------------------------------

def plot_training_log(agent, figsize=(19, 13)):
    tl = agent.training_log
    tla = agent.training_log_actor

    x = tl.get("epoch_num", [])
    xa = tla.get("epoch_num", [])

    gamma = getattr(agent, "gamma", None)
    horizon = getattr(getattr(agent, "buffer", None), "max_episode_len", None)
    horizon_factor = _discounted_horizon(gamma, horizon)

    fig, axs = plt.subplots(3, 3, figsize=figsize)
    axs = axs.flatten()

    # 1. World-model losses ---------------------------------------------------
    ax = axs[0]
    _lines_from_dict(ax, x, tl.get("accuracy_losses"), prefix="accuracy ")
    _lines_from_dict(ax, x, tl.get("complexity_losses"), prefix="complexity ")
    _finish(ax, "World model: accuracy & complexity")

    # 2. Reward composition (now with both entropy kinds) ---------------------
    ax = axs[1]
    extrinsic = tl.get("average_reward")
    curiosity = tl.get("curiosity")
    ent_net = tl.get("entropy_target_critic")          # net (sac - prior)
    ent_sac = tl.get("sac_entropy_target_critic")      # new
    ent_nrm = tl.get("normal_entropy_target_critic")   # new
    total_rew = tl.get("total_reward")

    _line(ax, x, extrinsic, label="extrinsic reward", color="tab:blue")
    _line(ax, x, curiosity, label="curiosity", color="tab:gray", alpha=0.6)
    _line(ax, x, ent_sac, label="SAC entropy (critic)", color="tab:red", lw=2.2)
    _line(ax, x, ent_nrm, label="normal prior (critic)", color="tab:orange",
          ls=":", lw=2)
    if total_rew is not None and ent_net is not None:
        n = min(len(total_rew), len(ent_net))
        effective = [total_rew[i] + ent_net[i] for i in range(n)]
        _line(ax, x, effective, label="effective value/step (reward+net entropy)",
              color="black", ls="--", lw=1.4)
    _finish(ax, "Reward composition  —  what the critic sums each step")

    # 3. Critic value vs entropy-implied --------------------------------------
    ax = axs[2]
    _line(ax, x, tl.get("Q_target"), label="Q target", color="tab:blue", lw=2)
    _lines_from_list(ax, x, tl.get("critic_predictions"), prefix="critic ",
                     alpha=0.8)
    _line(ax, x, tl.get("target_critic_output"), label="bootstrap Q(t+1)",
          color="tab:green", alpha=0.7)
    if ent_net is not None and horizon_factor is not None:
        implied = [e * horizon_factor for e in ent_net]
        _line(ax, x, implied, label="entropy-only Q (implied)",
              color="tab:red", ls="--", lw=2)
    title3 = "Critic value vs entropy-implied value"
    if gamma is not None and horizon is not None:
        title3 += f"  (γ={gamma:g}, H={horizon})"
    _finish(ax, title3)

    # 4. Q build-up -----------------------------------------------------------
    ax = axs[3]
    _line(ax, x, total_rew, label="immediate: total_reward", color="tab:blue")
    _line(ax, x, tl.get("future_Q_value"), label="bootstrapped future",
          color="tab:purple")
    _line(ax, x, tl.get("Q_target"), label="= Q target", color="black",
          ls="--", lw=1.4)
    _finish(ax, "Q build-up: immediate reward vs bootstrap")

    # 5. Actor losses ---------------------------------------------------------
    ax = axs[4]
    _line(ax, xa, tla.get("Q_for_actor"), label="Q for actor")
    _line(ax, xa, tla.get("entropy_for_actor"), label="entropy for actor")
    _line(ax, xa, tla.get("total_imitation_loss"), label="imitation")
    _line(ax, xa, tla.get("actor_loss"), label="actor loss", color="tab:red")
    _finish(ax, "Actor objective terms")

    # 6. Entropy terms (actor) ------------------------------------------------
    ax = axs[5]
    _lines_from_dict(ax, xa, tla.get("alpha_entropies"),
                     prefix="alpha*entropy ", color="tab:green")
    _lines_from_dict(ax, xa, tla.get("alpha_normal_entropies"),
                     prefix="normal prior ", color="tab:orange")
    _lines_from_dict(ax, xa, tla.get("total_entropies"),
                     prefix="net ", color="tab:blue")
    _finish(ax, "Entropy terms (actor: SAC bonus vs normal prior)")

    # 7. Alphas ---------------------------------------------------------------
    ax = axs[6]
    _lines_from_dict(ax, xa, tla.get("alphas"), prefix="alpha ")
    _lines_from_dict(ax, xa, tla.get("alpha_losses"), prefix="alpha loss ",
                     alpha=0.5, ls=":")
    _finish(ax, "Entropy temperature (alpha)")

    # 8. Critic TD loss -------------------------------------------------------
    ax = axs[7]
    _lines_from_list(ax, x, tl.get("critic_losses"), prefix="critic ")
    _finish(ax, "Critic TD loss")

    # 9. BEHAVIOR: episode length & exit rate (from logged mask) --------------
    #    Early termination only happens on reaching an exit, so
    #    episode length < max-length  <=>  the agent exited.
    ax = axs[8]
    masks = tl.get("mask")
    if masks:
        lengths_per_epoch = []
        ref_len = 0.0
        for m in masks:
            arr = _np(m)
            arr = arr.reshape(arr.shape[0], arr.shape[1], -1)
            L = arr.sum(axis=1).reshape(-1)        # steps per sampled episode
            lengths_per_epoch.append(L)
            if L.size:
                ref_len = max(ref_len, float(L.max()))
        avg_len = [float(L.mean()) if L.size else np.nan
                   for L in lengths_per_epoch]
        exit_rate = [float((L < ref_len).mean()) * 100.0 if L.size else np.nan
                     for L in lengths_per_epoch]

        n = min(len(x), len(avg_len))
        ax.plot(x[:n], avg_len[:n], color="tab:blue", label="avg episode length")
        ax.set_ylabel("steps", color="tab:blue")
        ax.tick_params(axis="y", labelcolor="tab:blue")
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.3)
        ax.set_title("Behavior: episode length & exit rate",
                     fontsize=11, fontweight="bold")

        ax2 = ax.twinx()
        ax2.plot(x[:n], exit_rate[:n], color="tab:green", label="exit rate")
        ax2.set_ylabel("exit %", color="tab:green")
        ax2.tick_params(axis="y", labelcolor="tab:green")
        ax2.set_ylim(-2, 102)

        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="best")
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "no 'mask' logged", ha="center", va="center",
                transform=ax.transAxes)

    # Footer: the two headline numbers + the single thing to watch -----------
    def _last(seq):
        return seq[-1] if seq else None

    bits = []
    e_last, r_last = _last(ent_net), _last(total_rew)
    q_last = None
    cps = tl.get("critic_predictions")
    if cps and cps[0]:
        q_last = cps[0][-1]
    if e_last is not None and r_last is not None:
        denom = abs(e_last) + abs(r_last) + 1e-8
        bits.append(f"entropy ≈ {100*abs(e_last)/denom:.0f}% of per-step value")
    if e_last is not None and horizon_factor is not None and q_last is not None:
        bits.append(f"implied entropy Q ≈ {e_last*horizon_factor:.1f} vs critic Q ≈ {q_last:.1f}")
    footer = ("Watch panel 9: episode length pinned at max + exit rate ~0 "
              "= the agent never reaches an exit (exploration problem, not reward scale).")
    if bits:
        footer = "Latest:  " + "   |   ".join(bits) + "\n" + footer
    fig.text(0.5, 0.005, footer, ha="center", va="bottom", fontsize=9,
             family="monospace")

    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig