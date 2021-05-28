from cartpole_env import Environment
from q_agent import Agent
from helpers.plot import plot_progress

if __name__ == "__main__":
    gym = Environment()
    agent = Agent()

    rewards = gym.train(agent, 1000)
    plot_progress(rewards)
