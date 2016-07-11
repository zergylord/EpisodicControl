import gym
import numpy as np
import pylab
import scipy.misc as misc
import time
from sklearn.neighbors import NearestNeighbors
env = gym.make('SpaceInvaders-v0')
n_act = env.action_space.n
knn = []
for a in range(n_act):
    knn.append(NearestNeighbors(n_neighbors=11))
s_dim = 84*84
cur_time = time.clock()
s = env.reset()
env.render()
def process_obs(obs):
    s = np.float32(obs)/255.0
    s = .299*s[:,:,0]+.587*s[:,:,1]+.114*s[:,:,2]
    s = misc.imresize(s,[84,84])
    return np.reshape(s,s_dim)
def get_action(rep):
    #global knn,n_act
    max_q_val = float("-inf")
    act = 0
    tied = []
    for a in range(n_act):
        _,inds = knn[a].kneighbors(np.expand_dims(rep,0))
        q_val = np.sum(R[a][inds])
        if q_val > max_q_val:
            max_q_val = q_val
            act = a
        elif q_val == max_q_val:
            tied.append(a)
    if tied:
        act = tied[np.random.randint(len(tied))]
        print('tie!')
    else:
        print('nooooooooope')
    return act



mem_size = int(1e5)
rep_dim = 64
S = np.zeros((n_act,mem_size,rep_dim))
R = np.zeros((n_act,mem_size,))
mem_ind = 0
episode_states = []
episode_actions = []
Ret = 0
warming = True
for i in range(10000):
    obs = process_obs(s)
    episode_states.append(obs)
    if np.random.rand() < .005 or warming:
        action = env.action_space.sample()
    else:
        action = get_action(np.matmul(obs,np.random.randn(s_dim,rep_dim)))
    episode_actions.append(action)
    for _ in range(4):
        s,r,done,_ = env.step(action)
    env.render()
    if r > 0:
        reward = 1.0
        print(i,done,r,reward)
    elif r < 0:
        reward = -1.0
        print(i,done,r,reward)
    else:
        reward = 0.0
    Ret+=reward
    if done:
        print(i,'done!',Ret,mem_ind)
        if warming:
            if i > 1:
                warming = False
        s = env.reset()
        episode_reps = np.matmul(np.asarray(episode_states),np.random.randn(s_dim,rep_dim))
        action_array = np.asarray(episode_actions)
        n_reps = episode_reps.shape[0]
        if mem_ind + n_reps >= mem_size:
            remaining = mem_size-mem_ind
            overflow = n_reps-remaining
            #TODO:fix action indexing
            for a in rnage(n_act):
                S[action_array[:remaining],mem_ind:] = episode_reps[:remaining]
                R[action_array[:remaining],mem_ind:] = Ret
                S[action_array[remaining:],:overflow] = episode_reps[remaining:]
                R[action_array[remaining:],:overflow] = Ret
            mem_ind = overflow
        else:
            S[action_array,mem_ind:mem_ind+n_reps] = episode_reps
            R[action_array,mem_ind:mem_ind+n_reps] = Ret
            mem_ind += n_reps
        max_ind = min(i,mem_size)
        for a in range(n_act):
            knn[a].fit(S[a][:max_ind])
        episode_states = []
        episode_actions = []
        Ret = 0
