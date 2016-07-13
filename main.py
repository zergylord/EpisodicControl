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
#env.render()
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
            a_not_in_tied = True
        elif a != act and q_val == max_q_val:
            if not tied:
                tied.append(act)
            tied.append(a)
            act = a
    if tied:
        act = np.random.choice(tied) 
    return act



mem_size = int(1e5)
rep_dim = 64
M = np.random.randn(s_dim,rep_dim)
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
cumr = 0.0
episodes = 0.0
warming = True
refresh = int(1e3)
for i in range(int(1e7)):
    obs = process_obs(s)
    if np.random.rand() < .005 or warming:
        action = env.action_space.sample()
    else:
        action = get_action(np.matmul(obs,M))
    episode_states[action].append(obs)
    reward = 0.0
    for _ in range(4):
        s,r,done,_ = env.step(action)
        reward+=r
        '''
        if r > 0:
            reward += 1.0
        elif r < 0:
            reward += -1.0
        '''
    #env.render()
    Ret+=reward
    if done:
        episodes+=1
        #print(i,'done!',Ret)
        cumr+=Ret
        if warming:
            if i > 1:
                warming = False
        s = env.reset()
        for a in range(n_act):
            n_reps = len(episode_states[a])
            if n_reps > 0:
                episode_reps = np.matmul(np.asarray(episode_states[a]),M)
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
    if i >0 and i % refresh == 0:
        print(i,cumr/episodes,time.clock()-cur_time)
        episodes = 0.0
        cumr = 0.0
        cur_time = time.clock()
