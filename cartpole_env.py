import gym

from config.train import Config
class Environment:
    def __init__(self, max_timesteps=Config.max_ts.value):
        self.max_timesteps = max_timesteps

        self.env = gym.make("CartPole-v0")

    def get_reward(self, state, action, r, next_state, done):
        return r

    def train(self, agent, num_episodes):
        self.play(agent, num_episodes, train=True)

    def play(self, agent, num_episodes, train=False):
        for episode in range(num_episodes):
            state = self.env.reset()
            state = state.reshape(1, self.env.observation_space.shape[0])
            total_reward = 0

            for t in range(self.max_timesteps):
                action = agent.get_action(state, train)

                next_state, reward, done, _ = self.env.step(action)
                next_state = next_state.reshape(1, self.env.observation_space.shape[0])
                reward = self.get_reward(state, action, reward, next_state, done)
                
                if train:
                    agent.remember(state, action, reward, next_state, done)
                else:
                    self.env.render()
                

                total_reward += reward
                state = next_state
                if done:
                    break

            if train:
                agent.train_memory()

            print("episode: {}/{} | score: {}".format(
                episode + 1, num_episodes, total_reward))
