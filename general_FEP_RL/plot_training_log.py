import matplotlib.pyplot as plt

def plot_training_log(agent, figsize=(16, 10)):
    training_log = agent.training_log
    training_log_actor = agent.training_log_actor

    epoch_nums = training_log["epoch_num"]
    epoch_nums_actor = training_log_actor["epoch_num"]

    fig, axs = plt.subplots(2, 3, figsize=figsize)
    axs = axs.flatten()

    # --- Plot 1: Accuracy + Complexity losses ---
    ax = axs[0]
    for key, value in training_log["accuracy_losses"].items():
        ax.plot(epoch_nums, value, label=f"accuracy loss {key}")
    for key, value in training_log["complexity_losses"].items():
        ax.plot(epoch_nums, value, label=f"complexity loss {key}")
    ax.set_title("Losses (Accuracy & Complexity)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Value")
    ax.grid(True)
    ax.legend()

    # --- Plot 2: Rewards ---
    ax = axs[1]
    ax.plot(epoch_nums, training_log["average_reward"], label="reward")
    for key, value in training_log["curiosities"].items():
        ax.plot(epoch_nums, value, label=f"curiosity {key}")
    ax.plot(epoch_nums, training_log["total_reward"], label="total")
    ax.set_title("Rewards")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Value")
    ax.grid(True)
    ax.legend()

    # --- Plot 3: Actor losses ---
    ax = axs[2]
    ax.plot(epoch_nums_actor, training_log_actor["Q_for_actor"], label="Q_for_actor")
    ax.plot(epoch_nums_actor, training_log_actor["entropy_for_actor"], label="entropy_for_actor")
    ax.plot(epoch_nums_actor, training_log_actor["total_imitation_loss"], label="total_imitation_loss")
    ax.plot(epoch_nums_actor, training_log_actor["actor_loss"], label="actor_loss")
    ax.set_title("Actor Losses")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Value")
    ax.grid(True)
    ax.legend()

    # --- Plot 4: Critic predictions ---
    ax = axs[3]
    ax.plot(epoch_nums, training_log["Q_target"], label="Q Target")
    for i, critic_pred in enumerate(training_log["critic_predictions"]):
        ax.plot(epoch_nums, critic_pred, label=f"Critic {i} Predictions")
    ax.set_title("Critic Predictions")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Value")
    ax.grid(True)
    ax.legend()

    # --- Plot 5: Alphas ---
    ax = axs[4]
    for key, alpha in training_log_actor["alphas"].items():
        ax.plot(epoch_nums_actor, alpha, label=f"Alpha {key}")
    ax.set_title("Alphas")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Value")
    ax.grid(True)
    ax.legend()

    # --- Plot 6: Empty / extra space ---
    axs[5].axis("off")

    fig.tight_layout()

    return fig

