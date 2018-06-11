
import numpy as np
import gym
import matplotlib.pyplot as plt

env = gym.make('MountainCar-v0')

num_episodes = 100
discount_factor = 1.0
alpha = 0.0001

#Parameter vector
w = np.zeros((3,2))

# Plots
plt_actions = np.zeros(3)
episode_rewards = np.zeros(num_episodes)

observation_examples = np.array([env.observation_space.sample() for x in range(10000)])
scale_mean = np.mean(observation_examples,axis=0)
scale_max = np.max(observation_examples,axis=0)
scale_min = np.min(observation_examples,axis=0)

def Q(state,action,w):
	# rescale action value
	x = (state - scale_mean)/(scale_max - scale_min)
	value = x.T.dot(w[action])
	return value

# Epsilon greedy policy
def policy(state, w, epsilon=0.1):
	A = np.ones(3,dtype=float) * epsilon/3
	best_action =  np.argmax([Q(state,a,w) for a in range(3)])
	A[best_action] += (1.0-epsilon)
	sample = np.random.choice(3,p=A)
	return sample

for e in range(num_episodes):

	state = env.reset()

	for i in range(1000):

		#env.render()
		action = policy(state,w)
		plt_actions[action] +=1

		next_state, reward, done, _ = env.step(action)
		next_action = policy(next_state,w)

		feature_vector = state

		episode_rewards[e] += reward

		target = reward + discount_factor * Q(next_state,next_action,w)		
		td_error = target - Q(state,action,w)
		dw = alpha * td_error * feature_vector

		w[action] -= dw

		if done:
			#print("finished episode")
			break

		state = next_state

#Plotting
plt.bar(np.arange(3),plt_actions)

plt.figure()

print(episode_rewards)
plt.plot(np.arange(100),episode_rewards)
plt.show()

env.close()
