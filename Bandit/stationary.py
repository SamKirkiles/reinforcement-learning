import gym
import numpy as np
import gym_bandits
import matplotlib.pyplot as plt

# A k-bandit algorithm to maximize reward from a series of 10 slot machines

#BanditTenArmedGaussian-v0
#BanditTenArmedRandomFixed-v0
#BanditTenArmedRandomRandom-v0
env = gym.make('BanditTenArmedGaussian-v0')
env.reset()


k = 10
epsilon = 0.1
iterations = 10000
beta = 0.01
running_reward = 0.0

# Estimated values at time t
Q = np.zeros((k))
# Number of tries for each arm
N = np.zeros((k))

env.seed(3)

x = np.arange(iterations)
y = np.zeros(iterations)

for i in range(iterations):
	
	if np.random.rand() < (1 - epsilon):
		# Exploit
		A = np.argmax(Q)
	else:
		# Explore
		A = np.random.randint(low=0,high=k)

	observation, reward, done, info = env.step(A)
	
	N[A] += 1
	Q[A] = Q[A] + (1/N[A]) * (reward - Q[A])

	# Take a running average of the reward

	running_reward = (beta * reward) + (1-beta) * running_reward
	y[i] = running_reward

plt.plot(x,y)
plt.xlabel("Iterations")
plt.xlabel("Reward")
plt.show()


