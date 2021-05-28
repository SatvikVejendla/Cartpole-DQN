import matplotlib.pyplot as plt
from config.files import Paths

def plot_progress(rewards):
    print(rewards)
    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.savefig(Paths.plot_path.value)
