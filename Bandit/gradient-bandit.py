import gym
import numpy as np
import gym_bandits
import matplotlib.pyplot as plt

env = gym.make('BanditTenArmedGaussian-v0')
env.reset()

iterations = 10000
learning_rate = 0.1
k = 10
epsilon = 0.1
beta = 0.1

# Preferece vector
H = np.zeros(k)
H = np.exp(H)/np.sum(np.exp(H))
# Average rewards
running_reward = 0
x = np.arange(iterations)
y = np.zeros(iterations)

np.random.seed(0)

for i in range(iterations):

	pi_a = np.exp(H)/np.sum(np.exp(H))

	# Choose action at time step t
	if np.random.rand() < (1 - epsilon):
		# Exploit
		# Sample from probabilty distribution
		A = np.random.choice(np.arange(k).tolist(),p=pi_a)
	else:
		# Explore
		A = np.random.randint(low=0,high=k)


	observation, reward, done, info = env.step(A)

	running_reward = (beta * reward) + (1-beta) * running_reward
	y[i] = running_reward

	# Update rule
	k_ = np.zeros(k)
	k_[A] = 1

	H += learning_rate * (reward - running_reward) * (k_-pi_a)


plt.plot(x,y)
plt.xlabel("Iterations")
plt.xlabel("Reward")
plt.show()


