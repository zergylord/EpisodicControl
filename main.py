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
        inds = inds[0]
        q_val = np.sum(R[a][inds])
        if q_val > max_q_val:
            max_q_val = q_val
            act = a
            tied = []
        elif q_val == max_q_val:
            tied.append(a)
    if tied:
        act = tied[np.random.randint(len(tied))]
    return act



mem_size = int(1e5)
rep_dim = 64
S = np.zeros((n_act,mem_size,rep_dim))
R = np.zeros((n_act,mem_size,))
mem_ind = np.zeros((n_act,),dtype=int)
episode_states = []
mem_full = []
for a in range(n_act):
    episode_states.append([])
    mem_full.append(False)
episode_actions = []
Ret = 0
warming = True
for i in range(int(1e7)):
    obs = process_obs(s)
    if np.random.rand() < .005 or warming:
        action = env.action_space.sample()
    else:
        action = get_action(np.matmul(obs,np.random.randn(s_dim,rep_dim)))
    episode_states[action].append(obs)
    for _ in range(4):
        s,r,done,_ = env.step(action)
    env.render()
    if r > 0:
        reward = 1.0
        #print(i,done,r,reward)
    elif r < 0:
        reward = -1.0
        #print(i,done,r,reward)
    else:
        reward = 0.0
    Ret+=reward
    if done:
        print(i,'done!',Ret)
        if warming:
            if i > 1:
                warming = False
        s = env.reset()
        for a in range(n_act):
            n_reps = len(episode_states[a])
            if n_reps > 0:
                episode_reps = np.matmul(np.asarray(episode_states[a]),np.random.randn(s_dim,rep_dim))
                if mem_ind[a] + n_reps >= mem_size:
                    mem_full[a] = True
                    remaining = mem_size-mem_ind[a]
                    overflow = n_reps-remaining
                    S[a,mem_ind[a]:] = episode_reps[:remaining]
                    R[a,mem_ind[a]:] = Ret
                    S[a,:overflow] = episode_reps[remaining:]
                    R[a,:overflow] = Ret
                    mem_ind[a] = overflow
                else:
                    S[a,mem_ind[a]:mem_ind[a]+n_reps] = episode_reps
                    R[a,mem_ind[a]:mem_ind[a]+n_reps] = Ret
                    mem_ind[a] += n_reps
                if mem_full[a]:
                    max_ind = mem_size
                else:
                    max_ind = mem_ind[a]
                knn[a].fit(S[a][:max_ind])
                episode_states[a] = []
        Ret = 0
