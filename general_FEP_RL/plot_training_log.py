import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Small, defensive helpers so a missing log key skips a line instead of
# crashing the whole figure (older runs / different agents stay plottable).
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
    """sum_{k=0}^{H-1} gamma^k  -- how much a constant per-step bonus inflates
    into a bootstrapped value. This is the bridge between a small per-step
    entropy bonus and a large Q."""
    if gamma is None or horizon is None:
        return None
    if gamma >= 1.0:
        return float(horizon)
    return (1.0 - gamma ** horizon) / (1.0 - gamma)


# ---------------------------------------------------------------------------

def plot_training_log(agent, figsize=(19, 13)):
    tl = agent.training_log
    tla = agent.training_log_actor

    x = tl.get("epoch_num", [])
    xa = tla.get("epoch_num", [])

    # Discounting context (used to turn a per-step entropy bonus into an
    # implied Q contribution). Both are read defensively.
    gamma = getattr(agent, "gamma", None)
    horizon = getattr(getattr(agent, "buffer", None), "max_episode_len", None)
    horizon_factor = _discounted_horizon(gamma, horizon)

    fig, axs = plt.subplots(3, 3, figsize=figsize)
    axs = axs.flatten()

    # -----------------------------------------------------------------------
    # 1. World-model losses: is perception learning?
    # -----------------------------------------------------------------------
    ax = axs[0]
    _lines_from_dict(ax, x, tl.get("accuracy_losses"), prefix="accuracy ")
    _lines_from_dict(ax, x, tl.get("complexity_losses"), prefix="complexity ")
    _finish(ax, "World model: accuracy & complexity")

    # -----------------------------------------------------------------------
    # 2. Reward composition  (THE entropy panel).
    #    The critic accumulates  total_reward + entropy_bonus  every step.
    #    Plotting them together shows which one the policy is actually paid for.
    # -----------------------------------------------------------------------
    ax = axs[1]
    extrinsic = tl.get("average_reward")
    curiosity = tl.get("curiosity")
    ent_bonus = tl.get("entropy_target_critic")   # mean alpha*(-logpi) per step
    total_rew = tl.get("total_reward")            # extrinsic + curiosity

    _line(ax, x, extrinsic, label="extrinsic reward", color="tab:blue")
    _line(ax, x, curiosity, label="curiosity", color="tab:orange")
    _line(ax, x, ent_bonus, label="entropy bonus (per step)",
          color="tab:red", lw=2.2)
    if total_rew is not None and ent_bonus is not None:
        n = min(len(total_rew), len(ent_bonus))
        effective = [total_rew[i] + ent_bonus[i] for i in range(n)]
        _line(ax, x, effective, label="effective value/step (reward+entropy)",
              color="black", ls="--", lw=1.4)
    _finish(ax, "Reward composition  —  what the critic sums each step")

    # -----------------------------------------------------------------------
    # 3. Critic value vs an entropy-only estimate.
    #    implied entropy Q = mean per-step entropy bonus * discounted horizon.
    #    If the dashed line sits on top of Q, the value function is ~all entropy.
    # -----------------------------------------------------------------------
    ax = axs[2]
    _line(ax, x, tl.get("Q_target"), label="Q target", color="tab:blue", lw=2)
    _lines_from_list(ax, x, tl.get("critic_predictions"), prefix="critic ",
                     alpha=0.8)
    _line(ax, x, tl.get("target_critic_output"), label="bootstrap Q(t+1)",
          color="tab:green", alpha=0.7)
    if ent_bonus is not None and horizon_factor is not None:
        implied = [e * horizon_factor for e in ent_bonus]
        _line(ax, x, implied, label="entropy-only Q (implied)",
              color="tab:red", ls="--", lw=2)
    title3 = "Critic value vs entropy-implied value"
    if gamma is not None and horizon is not None:
        title3 += f"  (γ={gamma:g}, H={horizon})"
    _finish(ax, title3)

    # -----------------------------------------------------------------------
    # 4. Q build-up: immediate reward vs bootstrapped future.
    #    Q_target = total_reward(now) + future_Q_value(bootstrap).
    #    Shows almost all of Q comes from the (entropy-fed) bootstrap.
    # -----------------------------------------------------------------------
    ax = axs[3]
    _line(ax, x, total_rew, label="immediate: total_reward", color="tab:blue")
    _line(ax, x, tl.get("future_Q_value"), label="bootstrapped future",
          color="tab:purple")
    _line(ax, x, tl.get("Q_target"), label="= Q target", color="black",
          ls="--", lw=1.4)
    _finish(ax, "Q build-up: immediate reward vs bootstrap")

    # -----------------------------------------------------------------------
    # 5. Actor losses (kept, lightly relabeled).
    # -----------------------------------------------------------------------
    ax = axs[4]
    _line(ax, xa, tla.get("Q_for_actor"), label="Q for actor")
    _line(ax, xa, tla.get("entropy_for_actor"), label="entropy for actor")
    _line(ax, xa, tla.get("total_imitation_loss"), label="imitation")
    _line(ax, xa, tla.get("actor_loss"), label="actor loss", color="tab:red")
    _finish(ax, "Actor objective terms")

    # -----------------------------------------------------------------------
    # 6. Entropy terms: SAC entropy bonus vs the normal action prior.
    #    These pull in different directions inside the actor loss.
    # -----------------------------------------------------------------------
    ax = axs[5]
    _lines_from_dict(ax, xa, tla.get("alpha_entropies"),
                     prefix="alpha*entropy ", color="tab:green")
    _lines_from_dict(ax, xa, tla.get("alpha_normal_entropies"),
                     prefix="normal prior ", color="tab:orange")
    _lines_from_dict(ax, xa, tla.get("total_entropies"),
                     prefix="net ", color="tab:blue")
    _finish(ax, "Entropy terms (SAC bonus vs normal prior)")

    # -----------------------------------------------------------------------
    # 7. Alphas and their update loss.
    # -----------------------------------------------------------------------
    ax = axs[6]
    _lines_from_dict(ax, xa, tla.get("alphas"), prefix="alpha ")
    _lines_from_dict(ax, xa, tla.get("alpha_losses"), prefix="alpha loss ",
                     alpha=0.5, ls=":")
    _finish(ax, "Entropy temperature (alpha)")

    # -----------------------------------------------------------------------
    # 8. Critic TD loss (was logged but never plotted) — stability check.
    # -----------------------------------------------------------------------
    ax = axs[7]
    _lines_from_list(ax, x, tl.get("critic_losses"), prefix="critic ")
    _finish(ax, "Critic TD loss")

    # -----------------------------------------------------------------------
    # 9. Reading guide + a couple of live, concrete numbers.
    # -----------------------------------------------------------------------
    ax = axs[8]
    ax.axis("off")

    def _last(seq):
        return seq[-1] if seq else None

    notes = [
        "How to read this dashboard",
        "",
        "(1) World model should fall — perception is learning.",
        "(2) Per-step value the critic sums. If the red entropy",
        "     bonus sits above extrinsic+curiosity, the policy is",
        "     paid to stay alive, not to reach exits.",
        "(3) If the dashed entropy-only line tracks Q, the value",
        "     function is essentially all entropy, not task reward.",
        "(4) Q is mostly the bootstrapped future, which carries",
        "     the accumulated entropy. Reaching an exit (done=1)",
        "     zeroes that future — so exits look 'expensive'.",
        "(6) green = exploration bonus, orange = pull toward a=0.",
    ]

    e_last = _last(ent_bonus)
    r_last = _last(total_rew)
    q_last = None
    cps = tl.get("critic_predictions")
    if cps and cps[0]:
        q_last = cps[0][-1]
    if e_last is not None and r_last is not None:
        denom = abs(e_last) + abs(r_last) + 1e-8
        share = 100.0 * abs(e_last) / denom
        notes += ["", f"Latest: entropy ≈ {share:.0f}% of per-step value"]
    if e_last is not None and horizon_factor is not None and q_last is not None:
        notes += [f"        implied entropy Q ≈ {e_last * horizon_factor:.1f}"
                  f"   vs   critic Q ≈ {q_last:.1f}"]

    ax.text(0.0, 1.0, "\n".join(notes), transform=ax.transAxes,
            ha="left", va="top", fontsize=9.5, family="monospace")

    fig.tight_layout()
    return fig