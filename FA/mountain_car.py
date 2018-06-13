
import numpy as np
import gym
import matplotlib.pyplot as plt
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier

env = gym.make('MountainCar-v0')
env._max_episode_steps = 300


num_episodes = 100
discount_factor = 1.0
alpha = 0.01

#Parameter vector
w = np.zeros((3,200))

# Plots
plt_actions = np.zeros(3)
episode_rewards = np.zeros(num_episodes)

# Get satistics over observation space samples for normalization
observation_examples = np.array([env.observation_space.sample() for x in range(10000)])
scale_mean = np.mean(observation_examples,axis=0)
scale_max = np.max(observation_examples,axis=0)
scale_min = np.min(observation_examples,axis=0)

# Create radial basis function sampler to convert states to features for nonlinear function approx
rbf_feature = RBFSampler(gamma=1,n_components=200, random_state=1)

# Normalize and turn into feature
def featurize_state(state):
	X = (state - scale_mean)/(scale_max - scale_min)
	X = np.array([X])
	X_features = rbf_feature.fit_transform(X)
	return X_features

def Q(state,action,w):
	value = state.dot(w[action])
	return value

# Epsilon greedy policy
def policy(state, weight, epsilon=0.1):
	A = np.ones(3,dtype=float) * epsilon/3
	best_action =  np.argmax([Q(state,a,w) for a in range(3)])
	A[best_action] += (1.0-epsilon)
	sample = np.random.choice(3,p=A)
	return sample

def check_gradients(index,state,next_state,next_action,weight,reward):

	ew1 = np.array(weight, copy=True) 
	ew2 = np.array(weight, copy=True)  
	epsilon = 1e-6
	ew1[action][index] += epsilon
	ew2[action][index] -= epsilon
	
	test_target_1 = reward + discount_factor * Q(next_state,next_action,ew1)		
	td_error_1 = target - Q(state,action,ew1)



	test_target_2 = reward + discount_factor * Q(next_state,next_action,ew2)		
	td_error_2 = target - Q(state,action,ew2)

	grad = (td_error_1 - td_error_2) / (2 * epsilon)
	
	return grad[0]


# Our main training loop
for e in range(num_episodes):

	state = env.reset()
	state = featurize_state(state)

	while True:

		#env.render()
		# Sample from our policy
		action = policy(state,w)
		# Staistic for graphing
		plt_actions[action] += 1
		# Step environment and get next state and make it a feature
		next_state, reward, done, _ = env.step(action)
		next_state = featurize_state(next_state)

		# Figure out what our policy tells us to do for the next state
		next_action = policy(next_state,w)

		# Statistic for graphing
		episode_rewards[e] += reward

		# Figure out target and td error
		target = reward + discount_factor * Q(next_state,next_action,w)		
		td_error = target - Q(state,action,w)

		# Find gradient with code to check it commented below (check passes)
		dw = (state).T.dot(td_error)
		
		#for i in range(4):
		#	print("First few gradients")
		#	print(str(i) + ": " + str(check_gradients(i,state,next_state,next_action,w,reward)) + " " + str(dw[i]))

		# Update weight
		w[action] -= alpha * dw

		if done:
			break
		# update our state
		state = next_state


# Show bar graph of actions chosen
plt.bar(np.arange(3),plt_actions)

plt.figure()
# Plot the reward over all episodes
plt.plot(np.arange(num_episodes),episode_rewards)
plt.show()


env.close()
