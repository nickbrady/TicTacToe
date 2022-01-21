"""
A simple example for Reinforcement Learning using table lookup Q-learning method.
An agent "o" is on the left of a 1 dimensional world, the treasure is on the rightmost location.
Run this program and to see how the agent will improve its strategy of finding the treasure.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""

import numpy as np
import pandas as pd
import time
from copy import copy
import random

import sys
sys.path.insert(0, '/Users/nicholasbrady/Documents/School/Academic/West Research/Projects/')
from PythonHelperFunctions import *
plot_parameters()
csfont = {'fontname':'Serif'}
matplotlib.rc('xtick', top=True, bottom=True, direction='in')
matplotlib.rc('ytick', left=True, right=True, direction='in')
plt.rc("axes.spines", top=True, right=True)
matplotlib.rc('axes', edgecolor='k')

np.random.seed(2)  # reproducible


N_STATES = 6   # the length of the 1 dimensional world
ACTIONS = ['left', 'right']     # available actions
EPSILON = 0.9   # greedy police
ALPHA = 0.1     # learning rate
GAMMA = 0.9    # discount factor
MAX_EPISODES = 2000   # maximum episodes
FRESH_TIME = 0.0    # fresh time for one move


def build_q_table(n_states, actions):
    table = pd.DataFrame(
        np.zeros((n_states, len(actions))),     # q_table initial values
        columns=actions,    # actions's name
    )
    # print(table)    # show table
    return table


def choose_action(state, q_table):
    # This is how to choose an action
    state_actions = q_table.iloc[state, :]
    if (np.random.uniform() > EPSILON) or ((state_actions == 0).all()):  # act non-greedy or state-action have no value
        action_name = np.random.choice(ACTIONS)
    else:   # act greedy
        action_name = state_actions.idxmax()    # replace argmax to idxmax as argmax means a different function in newer version of pandas
    return action_name


def get_env_feedback(S, A):
    # This is how agent will interact with the environment
    if A == 'right':    # move right
        if S == N_STATES - 2:   # terminate
            S_ = 'terminal'
            R = 1
        else:
            S_ = S + 1
            R = 0
    else:   # move left
        R = 0
        if S == 0:
            S_ = S  # reach the wall
        else:
            S_ = S - 1
    return S_, R


def update_env(S, episode, step_counter):
    # This is how environment be updated
    env_list = ['-']*(N_STATES-1) + ['T']   # '---------T' our environment
    if S == 'terminal':
        interaction = 'Episode %s: total_steps = %s' % (episode+1, step_counter)
        print('\r{}'.format(interaction), end='')
        time.sleep(0.)
        print('\r                                ', end='')
    else:
        env_list[S] = 'o'
        interaction = ''.join(env_list)
        print('\r{}'.format(interaction), end='')
        time.sleep(FRESH_TIME)

def update_env(S, episode, step_counter):
    if S == 'terminal':
        interaction = 'Episode %s' % (episode+1)
        print('\r{}'.format(interaction), end='')


def rl():
    # main part of RL loop
    q_table = build_q_table(N_STATES, ACTIONS)
    for episode in range(MAX_EPISODES):
        step_counter = 0
        S = 0
        is_terminated = False
        update_env(S, episode, step_counter)
        while not is_terminated:

            A = choose_action(S, q_table)
            S_, R = get_env_feedback(S, A)  # take action & get next state and reward
            q_predict = q_table.loc[S, A]
            if S_ != 'terminal':
                q_target = R + GAMMA * q_table.iloc[S_, :].max()   # next state is not terminal
            else:
                q_target = R     # next state is terminal
                is_terminated = True    # terminate this episode

            q_table.loc[S, A] += ALPHA * (q_target - q_predict)  # update
            S = S_  # move to next state

            update_env(S, episode, step_counter+1)
            step_counter += 1

        # display(q_table)
    return q_table


if __name__ == "__main__":
    q_table = rl()
    print('\r\nQ-table:\n')
    display(q_table)


# In[1]:
MAX_EPISODES = 2000

q_table = build_q_table(N_STATES, ACTIONS)

step_results = []
for episode in range(MAX_EPISODES):
    step_counter = 0
    S = 0
    is_terminated = False
    update_env(S, episode, step_counter)
    while not is_terminated:

        A = choose_action(S, q_table)
        S_, R = get_env_feedback(S, A)  # take action & get next state and reward
        q_predict = q_table.loc[S, A]
        if S_ != 'terminal':
            q_target = R + GAMMA * q_table.iloc[S_, :].max()   # next state is not terminal
        else:
            q_target = R     # next state is terminal
            is_terminated = True    # terminate this episode

        q_table.loc[S, A] += ALPHA * (q_target - q_predict)  # update
        S = S_  # move to next state

        update_env(S, episode, step_counter+1)
        step_counter += 1

    step_results.append(step_counter - 5)
    # display(q_table)
display(q_table)

# In[5]:
plt.semilogy(step_results[:100])


# In[2]:

MAX_EPISODES = 1000

q_table_target_training = build_q_table(N_STATES, ACTIONS)
q_table_target = copy(q_table_target_updated)

for episode in range(MAX_EPISODES+1):
    step_counter = 0
    S = 0
    is_terminated = False
    update_env(S, episode, step_counter)
    while not is_terminated:

        A = choose_action(S, q_table_target)
        S_, R = get_env_feedback(S, A)  # take action & get next state and reward
        q_predict = q_table_target_training.loc[S, A]
        # only update the target_update qtable
        if S_ != 'terminal':
            q_target = R + GAMMA * q_table_target_training.iloc[S_, :].max()   # next state is not terminal
        else:
            q_target = R     # next state is terminal
            is_terminated = True    # terminate this episode

        q_table_target_training.loc[S, A] += ALPHA * (q_target - q_predict)  # update
        S = S_  # move to next state


        update_env(S, episode, step_counter+1)
        step_counter += 1

    if episode % 100 == 0:
        display(q_table_target)
        q_table_target = copy(q_table_target_training)
        print()

# In[3]:
''' Double Q-Learning '''

MAX_EPISODES = 4000

q_table_A = build_q_table(N_STATES, ACTIONS)
q_table_B = copy(q_table_A)


for episode in range(MAX_EPISODES+1):
    step_counter = 0
    S = 0
    is_terminated = False
    update_env(S, episode, step_counter)

    while not is_terminated:
        q_table_C = (q_table_A + q_table_B)/2

        A = choose_action(S, q_table_C)
        S_, R = get_env_feedback(S, A)  # take action & get next state and reward

        if random.uniform(0,1) < 0.5:   # randomly choose with q-table gets updated
            q_update = q_table_A
            q_target = q_table_B
        else:
            q_update = q_table_B
            q_target = q_table_A

        if S_ == 'terminal':
            is_terminated = True
            q_update.loc[S, A] = q_update.loc[S, A] + ALPHA * (R - q_update.loc[S, A])
            S = S_
            update_env(S, episode, step_counter+1)
            step_counter += 1
            continue    # skip the rest below

        if q_update.loc[S_, 'left'] == q_update.loc[S_, 'right']:
            A_ = random.randint(0, 1)
        else:
            A_ = q_update.iloc[S_, :].argmax()

        if A_ == 0:
            A_ = 'left'
        else:
            A_ = 'right'

        q_update.loc[S, A] = q_update.loc[S, A] + ALPHA * (R + GAMMA * q_target.loc[S_, A_] - q_update.loc[S, A])
        S = S_  # move to next state


        update_env(S, episode, step_counter+1)
        step_counter += 1

    # if episode % 10 == 0:
display(q_table_A)
display(q_table_B)
print()
