from cartpole_env import Environment
from q_agent import Agent


if __name__ == "__main__":
    gym = Environment()
    agent = Agent()

    gym.train(agent, 1000)
