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
step = 0.2
c = 0.2
iterations = 10000
beta = 0.01
running_reward = 0.0

# Estimated values at time t
Q = np.zeros((k))
# Number of tries for each arm
N = np.ones((k))

env.seed(3)
np.random.seed(3)

x = np.arange(iterations)
y = np.zeros(iterations)

for i in range(1,iterations):
	
	if np.random.rand() < (1 - epsilon):
		# Exploit
		A = np.argmax(Q + c * np.sqrt(np.log(i)/N))
	else:
		# Explore
		A = np.random.randint(low=0,high=k)

	observation, reward, done, info = env.step(A)
	
	N[A] += 1
	Q[A] += step * (reward - Q[A])

	# Take a running average of the reward

	running_reward = (beta * reward) + (1-beta) * running_reward
	y[i] = running_reward

plt.plot(x,y)
plt.xlabel("Iterations")
plt.xlabel("Reward")
plt.show()
