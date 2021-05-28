# Cartpole-DQN

A Deep Q Learning Keras AI which uses the Epsilon Greedy Policy to solve the Cartpole Environment by OpenAI Gym


The agent starts out by knowing nothing of the environment. It makes random decisions that it stores into its memory. After a specific number of rounds, the agent will train itself based on the items in its memory. To calculate the target reward for each observation state, a gamma discount rate is applied to the Bellman Equation.

The agent will continue to learn until it beats the game. At around episode 600 is when the agent makes a drastic improvement.

This is the initial model:

![Untrained Model](./assets/untrained.gif?raw=true)


This is the model after 1000 rounds of training:

![Trained Model](./assets/trained.gif?raw=true)



The rewards over the episodes can be shown in this plot:

![Progress](./assets/progress.png?raw=true)
