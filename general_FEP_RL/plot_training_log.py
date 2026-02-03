import matplotlib.pyplot as plt

def plot_training_log(training_log, folder="", epoch = 0):
    
    epoch_nums = training_log["epoch_num"]
        
    plt.figure(figsize=(6, 6))
    for key, value in training_log["accuracy_losses"].items():
        plt.plot(epoch_nums, value, label=f"accuracy loss {key}")
    for key, value in training_log["complexity_losses"].items():
        plt.plot(epoch_nums, value, label=f"complexity loss {key}")
    plt.title(f"Losses for Accuracy and Complexity over epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    #if folder != "":
    #    os.makedirs(f"{folder}/accuracy", exist_ok=True)
    #    plt.savefig(f"{folder}/accuracy/{epoch}.png")
    plt.show()
    plt.close()
    
    plt.figure(figsize=(6, 6))
    plt.plot(epoch_nums, training_log["average_reward"], label="reward")
    for key, value in training_log["curiosities"].items():
        plt.plot(epoch_nums, value, label=f"curiosity {key}")
    plt.plot(epoch_nums, training_log["total_reward"], label="total")
    plt.title(f"Rewards over epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    #if folder != "":
    #    os.makedirs(f"{folder}/reward", exist_ok=True)
    #    plt.savefig(f"{folder}/reward/{epoch}.png")
    plt.show()
    plt.close()
    
    plt.figure(figsize=(6, 6))
    plt.plot(epoch_nums, training_log["Q_for_actor"], label="-Q_for_actor")
    plt.plot(epoch_nums, training_log["entropy_for_actor"], label="-entropy_for_actor")
    plt.plot(epoch_nums, training_log["total_imitation_loss"], label="-total_imitation_loss")
    plt.plot(epoch_nums, training_log["actor_loss"], label="actor_loss")
    plt.title(f"Actor loss over epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    #if folder != "":
    #    os.makedirs(f"{folder}/actor", exist_ok=True)
    #    plt.savefig(f"{folder}/actor/{epoch}.png")
    plt.show()
    plt.close()
    
    plt.figure(figsize=(6, 6))
    plt.plot(epoch_nums, training_log["Q_target"], label="Q Target")
    for i, critic_loss in enumerate(training_log["critic_predictions"]):
        plt.plot(epoch_nums, critic_loss, label=f"Critic {i} Predictions")
    #for i, loss in enumerate(training_log["critic_losses"]):
    #    plt.plot(epoch_nums, loss, label=f"Critic {i} Loss")
    plt.title(f"Critic predictions over epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    #if folder != "":
    #    os.makedirs(f"{folder}/critic", exist_ok=True)
    #    plt.savefig(f"{folder}/critic/{epoch}.png")
    plt.show()
    plt.close()
    
    plt.figure(figsize=(6, 6))
    for key, alpha in training_log["alphas"].items():
        plt.plot(epoch_nums, alpha, label=f"Alpha {key}")
    plt.title(f"Alphas over epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    #if folder != "":
    #    os.makedirs(f"{folder}/critic", exist_ok=True)
    #    plt.savefig(f"{folder}/critic/{epoch}.png")
    plt.show()
    plt.close()