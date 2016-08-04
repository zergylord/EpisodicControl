import gym
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc as misc
import time
from sklearn.neighbors import NearestNeighbors
from ops import *
gamma = 1.0
epsilon = .005
num_neighbors = 5
env = gym.make('Frostbite-v0')
#env = gym.make('SpaceInvaders-v0')
n_act = env.action_space.n
print(n_act)
knn = []
for a in range(n_act):
    knn.append(NearestNeighbors(n_neighbors=num_neighbors,metric='euclidean'))
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
    nearby = []
    exact_match = {}
    for a in range(n_act):
        nearby.append([])
        dists,inds = knn[a].kneighbors(np.expand_dims(rep,0))
        dists = dists[0]
        inds = inds[0]
        nearby[a] = inds
        if 0.0 in dists:
            hit_ind = inds[dists == 0.0][0]
            #print('hit!',hit_ind,R[a][hit_ind])
            q_val = R[a][hit_ind]
            exact_match[a] = hit_ind
        else:
            q_val = np.mean(R[a][inds])
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
    return act,nearby,exact_match



mem_size = int(1e6)
rep_dim = 64
M = np.random.randn(s_dim,rep_dim)
S = np.zeros((n_act,mem_size,rep_dim))
R = np.zeros((n_act,mem_size,))
last_used = np.tile(np.asarray(range(-mem_size,0)),[n_act,1])
mem_ind = np.zeros((n_act,),dtype=int)
episode_states = []
episode_rets = []
episode_matches = []
for a in range(n_act):
    episode_states.append([])
    episode_rets.append([])
    episode_matches.append([])
episode_actions = []
episode_rewards = []
total_hits = 0
Ret = 0
cumr = 0.0
episodes = 0.0
warming = True
refresh = int(1e3)
plt.ion()
r_hist = []
for i in range(int(1e7)):
    obs = process_obs(s)
    state = np.matmul(obs,M)
    
    if not warming:
        action,nearby,match = get_action(state)
        for a in range(n_act):
            last_used[a,nearby[a]] = i

    if np.random.rand() < epsilon or warming:
        action = env.action_space.sample()
    if not warming and action in match:
        episode_matches[action].append(match[action])
        total_hits+=1
    else:
        episode_matches[action].append(-np.inf)
    
    episode_actions.append(action)
    episode_states[action].append(state)
    reward = 0.0
    for _ in range(1): #builtin 2-4 frameskip
        s,r,done,_ = env.step(action)
        reward+=r
        '''
        if r > 0:
            reward += 1.0
        elif r < 0:
            reward += -1.0
        '''
    episode_rewards.append(reward)
    #env.render()
    Ret+=reward
    if done:
        step_reps = compute_return(episode_rewards,gamma)
        for step in range(len(step_reps)):
            episode_rets[episode_actions[step]].append(step_reps[step])
        episodes+=1
        #print(i,'done!',Ret)
        cumr+=Ret
        if warming:
            if i > 250:
                warming = False
        s = env.reset()
        for a in range(n_act):
            n_reps = len(episode_states[a])
            if n_reps > 0:
                episode_reps = np.asarray(episode_states[a])
                replace_these = np.argpartition(last_used[a],n_reps-1)[:n_reps]
                last_used[a][replace_these] = i
                S[a,replace_these] = episode_reps
                R[a,replace_these] = np.maximum(episode_matches[a],episode_rets[a]) #hack, where non-matches are -inf
                if mem_ind[a] + n_reps < mem_size:
                    mem_ind[a] += n_reps
                    #print(mem_ind[a],len(last_used[a][last_used[a] >= 0.0])) 
                else:
                    mem_ind[a] = mem_size
                knn[a].fit(S[a][:mem_ind[a]])
                episode_states[a] = []
                episode_rets[a] = []
                episode_matches[a] = []
        episode_rewards = []
        episode_actions = []
        Ret = 0
    if i >0 and i % refresh == 0:
        plt.clf()
        #plt.plot(last_used[0])
        r_hist.append(cumr/episodes)
        plt.plot(r_hist)
        plt.pause(.1)
        print(i,'reward per episode: ',cumr/episodes,'hit %: ',total_hits/i, 'time: ', time.clock()-cur_time)
        episodes = 0.0
        cumr = 0.0
        cur_time = time.clock()
