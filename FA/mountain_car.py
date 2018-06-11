
import numpy as np
import gym

env = gym.make('MountainCar-v0')

state = env.reset()

def Q(state,action,w):
	pass

for i in range(1000):

	env.render()
	action = env.action_space.sample()
	next_state, reward, done, _ = env.step(action)
	print(state)
	print(action)
	state = next_state


env.close()