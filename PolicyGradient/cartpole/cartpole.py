import gym
import numpy as np
import sklearn.pipeline
import sklearn.preprocessing
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier
import matplotlib.pyplot as plt
import copy 

env = gym.make('CartPole-v0')
num_episodes=3000
nA = env.action_space.n
#env._max_episode_steps = 100000
np.random.seed(1)
w = np.zeros((400,nA))

# Get satistics over observation space samples for normalization
env.reset()
observation_examples = []
for i in range(200):
	s,r,d,_ = env.step(1)
	observation_examples.append(s)

scaler = sklearn.preprocessing.StandardScaler()
scaler.fit(np.array(observation_examples))	

# Create radial basis function sampler to convert states to features for nonlinear function approx
featurizer = sklearn.pipeline.FeatureUnion([
        ("rbf1", RBFSampler(gamma=5.0, n_components=100)),
        ("rbf2", RBFSampler(gamma=2.0, n_components=100)),
        ("rbf3", RBFSampler(gamma=1.0, n_components=100)),
        ("rbf4", RBFSampler(gamma=0.5, n_components=100))
		])
# Fit featurizer to our scaled inputs
featurizer.fit(scaler.transform(observation_examples))


def policy(state,w):
	z = state.dot(w)
	exp = np.exp(z)
	return exp/np.sum(exp)

def featurize_state(state):
	# Transform data

	scaled = scaler.transform([state])
	featurized = featurizer.transform(scaled)
	return featurized

episode_rewards = []

def softmax_grad(softmax):
    s = softmax.reshape(-1,1)
    return np.diagflat(s) - np.dot(s, s.T)


for e in range(num_episodes):

	state = env.reset()
	state = featurize_state(state)

	score = 0
	actions = np.zeros(2)
	avg_probs = np.zeros(2)
	i = 0
	while True:
		#env.render()
		probs = policy(state,w)
		avg_probs += probs[0]
		action = np.random.choice(nA,p=probs[0])
		actions[action] += 1
		i += 1
		next_state,reward,done,_ = env.step(action)
		next_state = featurize_state(next_state)

		# Calculate Gradient
		dsoftmax = softmax_grad(probs)
		v = dsoftmax[action,:] / probs[0,action]
		grad = state.T.dot(v[None,:]) 

		w += 0.0001 * grad * reward 

		# CHECK GRADSS
		w1 = np.copy(w)
		w2 = np.copy(w)	
		w3 = np.copy(w)	
		w4 = np.copy(w)	
		w5 = np.copy(w)	
		w6 = np.copy(w)	

		i = 50
	
		w1[i,0] += 1e-8
		w2[i,0] -= 1e-8

		w3[i,1] += 1e-8
		w4[i,1] -= 1e-8
		#w5[i,2] += 1e-8
		#w6[i,2] -= 1e-8

		# we want this to match up for all dimensions

		check0 = (np.log(policy(state,w1))[0,action] - np.log(policy(state,w2))[0,action])/(2.*1e-8)
		check1 = (np.log(policy(state,w3))[0,action] - np.log(policy(state,w4))[0,action])/(2.*1e-8)
		#check2 = (np.log(policy(state,w5))[0,action] - np.log(policy(state,w6))[0,action])/(2.*1e-8)
		#print("Approximate Theta: " + str([check0,check1,check2]))
		#print("Actual Theta     : " + str(grad[i]) + "\n\n\n\n" )

		score+=reward
		
		if done:
			break
		
		state = next_state
	
	episode_rewards.append(score) 
	print("EP: " + str(e) + " Score: " + str(score) + " Actions: " + str(actions) + " Probs: " + str(avg_probs/i) + "         ",end="\r", flush=False) 

plt.plot(np.arange(num_episodes),episode_rewards)
plt.show()
env.close()

