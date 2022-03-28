"""
<<<<<<< HEAD
A simple example for Reinforcement Learning using table lookup Q-learning method.
An agent "o" is on the left of a 1 dimensional world, the treasure is on the rightmost location.
Run this program and to see how the agent will improve its strategy of finding the treasure.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/
=======
A simple example for Reinforcement Learning using the Q-learning method.
An agent "o" is on the left of a 1 dimensional world, the treasure is on the rightmost location.
Run this program and to see how the agent will improve its strategy of finding the treasure.

View more on the tutorial page: https://morvanzhou.github.io/tutorials/
>>>>>>> NewBranch
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
MAX_EPISODES = 1100

q_table = build_q_table(N_STATES, ACTIONS)

left_qtable, right_qtable = [np.zeros((MAX_EPISODES, N_STATES)), np.zeros((MAX_EPISODES, N_STATES))]

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

    left_qtable[episode, :] = q_table['left'].values
    right_qtable[episode, :] = q_table['right'].values

# In[2]:
ax, fig = axes(fig_number=1, rows=1, columns=2, row_height=4, column_width=6*4/5)

colors = ['b', 'g', 'r', 'purple', 'k']

for state in range(5):
    ax[1].plot(left_qtable[:,state], 'o', color=colors[state])
    ax[2].plot(right_qtable[:,state], 'o', color=colors[state], label = 'State {}'.format(state + 1))

ax[1].set_ylim(ax[2].get_ylim())
ax[2].set_xlim(-5, 205)
ax[1].set_xlim(-25, 1025)

<<<<<<< HEAD
fig.text(x=0.5, y=0.95, s='Q Table Values', ha = 'center', fontsize=20, fontweight='bold')
=======
fig.text(x=0.5, y=0.99, s='Q Table Values', ha = 'center', fontsize=20, fontweight='bold')
>>>>>>> NewBranch
ax[1].set_title('Left')
ax[2].set_title('Right')

ax[1].set_ylabel('Value')
ax[1].set_xlabel('Training Episode')
ax[2].set_xlabel('Training Episode')

handles, labels = ax[2].get_legend_handles_labels()
order = [4,3,2,1,0]
ax[2].legend([handles[idx] for idx in order],[labels[idx] for idx in order], fontsize=14)
ax[2].set_yticklabels([])

fig.tight_layout()

fig.savefig('TreasureHunt/QTable_Left_Right.png', format='png', dpi=300, bbox_inches = "tight")


# In[3]:
''' Double Q-Learning '''

MAX_EPISODES = 2000

q_table_A = build_q_table(N_STATES, ACTIONS)
q_table_B = copy(q_table_A)

left_qtable_A, right_qtable_A = [np.zeros((MAX_EPISODES, N_STATES)), np.zeros((MAX_EPISODES, N_STATES))]
left_qtable_B, right_qtable_B = [np.zeros((MAX_EPISODES, N_STATES)), np.zeros((MAX_EPISODES, N_STATES))]

for episode in range(MAX_EPISODES):
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

    left_qtable_A[episode, :]  = q_table_A['left'].values
    right_qtable_A[episode, :] = q_table_A['right'].values
    left_qtable_B[episode, :]  = q_table_B['left'].values
    right_qtable_B[episode, :] = q_table_B['right'].values

# In[4]:

ax, fig = axes(rows = 2, columns=2, row_height=4, column_width=6*4/5)

for state in range(5):
    ax[1].plot(left_qtable_A[:,state], 'o', color=colors[state])
    ax[2].plot(right_qtable_A[:,state], 'o', color=colors[state], label = 'State {}'.format(state + 1))

    ax[3].plot(left_qtable_B[:,state], 'o', color=colors[state])
    ax[4].plot(right_qtable_B[:,state], 'o', color=colors[state], label = 'State {}'.format(state + 1))

ax[1].set_ylim(ax[2].get_ylim())
ax[2].set_xlim(-10, 250)
ax[1].set_xlim(-50, 2050)
ax[4].set_xlim(-10, 250)
ax[3].set_xlim(-50, 2050)

ax[1].set_ylim(ax[2].get_ylim())
ax[3].set_ylim(ax[4].get_ylim())

fig.text(x=0.5, y=0.97, s='Q Table A', ha = 'center', fontsize=20, fontweight='bold')
ax[1].set_title('Left')
ax[2].set_title('Right')

fig.text(x=0.5, y=0.505, s='Q Table B', ha = 'center', fontsize=20, fontweight='bold')
ax[3].set_title('Left')
ax[4].set_title('Right')

ax[1].set_xticklabels([])
ax[2].set_xticklabels([])

ax[2].set_yticklabels([])
ax[4].set_yticklabels([])

ax[1].set_ylabel('Value')
ax[3].set_ylabel('Value')
ax[3].set_xlabel('Training Episode')
ax[4].set_xlabel('Training Episode')

handles, labels = ax[2].get_legend_handles_labels()
order = [4,3,2,1,0]
ax[2].legend([handles[idx] for idx in order],[labels[idx] for idx in order], fontsize=14)
ax[4].legend([handles[idx] for idx in order],[labels[idx] for idx in order], fontsize=14)

fig.tight_layout()

fig.savefig('TreasureHunt/DoubleQTable_Left_Right.png', format='png', dpi=300, bbox_inches = "tight")
<<<<<<< HEAD
=======

# In[5]:
'''
    Inspired from a YouTube video: https://www.youtube.com/watch?v=SX08NT55YhA (~10:50), we investigate how the q-table evolves if instead of always starting the agent at the left-most state (S = 0), we start the agent at a random state, not including the right-most state (S = random.choice([0, 1, 2, 3, 4])).

    The intuition was that starting the agent at random states was a good way to increase the exploration aspect of exploration vs exploitation. In the YouTube video, introducing the agent into random game states seemed to be a method to reduce overfitting and increase a more general approach to learning.

    Because this particular toy game requires the agent to pass through every possible game state in its search for the treasure, there is (or seems to be) no benefit (and perhaps even a hinderance) to using random starting states. The envidence being the underdeveloped Q-values, relative to those produced when the starting state is always S = 0.
'''

MAX_EPISODES = 1100

q_table = build_q_table(N_STATES, ACTIONS)
states = [0, 1, 2, 3, 4] # can't start at the furthest right state

left_qtable, right_qtable = [np.zeros((MAX_EPISODES, N_STATES)), np.zeros((MAX_EPISODES, N_STATES))]

for episode in range(MAX_EPISODES):
    step_counter = 0
    S = random.choice(states)
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

    left_qtable[episode, :] = q_table['left'].values
    right_qtable[episode, :] = q_table['right'].values

# In[6]:
ax, fig = axes(fig_number=1, rows=1, columns=2, row_height=4, column_width=6*4/5)

colors = ['b', 'g', 'r', 'purple', 'k']

for state in range(5):
    ax[1].plot(left_qtable[:,state], 'o', color=colors[state])
    ax[2].plot(right_qtable[:,state], 'o', color=colors[state], label = 'State {}'.format(state + 1))

ax[1].set_ylim(ax[2].get_ylim())
ax[2].set_xlim(-5, 205)
ax[1].set_xlim(-25, 1025)

fig.text(x=0.5, y=0.99, s='Q Table Values', ha = 'center', fontsize=20, fontweight='bold')
ax[1].set_title('Left')
ax[2].set_title('Right')

ax[1].set_ylabel('Value')
ax[1].set_xlabel('Training Episode')
ax[2].set_xlabel('Training Episode')

handles, labels = ax[2].get_legend_handles_labels()
order = [4,3,2,1,0]
ax[2].legend([handles[idx] for idx in order],[labels[idx] for idx in order], fontsize=14)
ax[2].set_yticklabels([])

fig.tight_layout()

# fig.savefig('TreasureHunt/QTable_Left_Right.png', format='png', dpi=300, bbox_inches = "tight")
>>>>>>> NewBranch
