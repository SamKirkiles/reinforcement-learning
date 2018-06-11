
import numpy as np
import gym

env = gym.make('MountainCar-v0')

state = env.reset()

num_episodes = 3

#Parameter vector
w = np.random.rand(3)

def Q(state,action,w):
	x = np.append(state,action)
	print(x.dot(w))



for e in range(num_episodes):
	for i in range(1000):

		env.render()
		action = env.action_space.sample()
		next_state, reward, done, _ = env.step(action)
		value = Q(state,action,w)
		if done:
			print("finished episode")
			break
		state = next_state


env.close()