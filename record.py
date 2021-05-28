import gym
from PIL import Image

from q_agent import Agent


# Untrained Agent
env = gym.make('CartPole-v0')
agent = Agent()

state = env.reset()
frames = []
finished = 0
while True:
    state = state.reshape(1, env.observation_space.shape[0])
    frames.append(Image.fromarray(env.render(mode='rgb_array')))
    action = agent.get_action(state, False)
    state, reward, done, info = env.step(action)
    if done:
        finished+=1
        state = env.reset()
    if finished > 4:
        break
        
with open('./assets/untrained.gif', 'wb') as f: 
    im = Image.new('RGB', frames[0].size)
    im.save(f, save_all=True, append_images=frames)




# Trained Agent


agent = Agent(False)

state = env.reset()
frames = []
finished = 0
while True:
    state = state.reshape(1, env.observation_space.shape[0])
    frames.append(Image.fromarray(env.render(mode='rgb_array')))
    action = agent.get_action(state, False)
    state, reward, done, info = env.step(action)
    if done:
        finished+=1
        state = env.reset()
    if finished > 2:
        break
        
with open('./assets/trained.gif', 'wb') as f: 
    im = Image.new('RGB', frames[0].size)
    im.save(f, save_all=True, append_images=frames)



