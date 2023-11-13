from src import DoubleDQN, create_model
import gymnasium as gym
env = gym.make('ALE/Breakout-v5')
model = create_model(env)
# display(model)