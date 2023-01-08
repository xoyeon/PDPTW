import pandas as pd
import random
import numpy as np

import matplotlib.pyplot as plt
from time import time 

# Env
class Game:
    board = None
    board_size = 0
    
    def __init__(self, board_size=4):
        self.board_size = board_size
        self.reset()
        
    def reset(self):
        self.board = np.zeros(self.board_size)
        
    def play(self, cell):
        if self.board[cell] == 0:
            self.board[cell] = 1
            game_over = len(np.where(self.board == 0)[0]) == 0
            return (1, game_over)
        else:
            return (-1, False)
        
def state_to_str(state):
    return str(list(map(int, state.tolist())))

all_states = list()
for i in range(2):
    for j in range(2):
        for k in range(2):
            for l in range(2):
                s = np.array([i,j,k,l])
                all_states.append(state_to_str(s))
                
print('All possible states : ')
for s in all_states:
    print(s)
    
''' ################### Q-Learning ################### '''

env = Game()

episodes = 2000
epsilon = 0.1
gamma = 1

# Q-Table 초기화
q_table = pd.DataFrame(0, index=np.arange(4), columns=all_states)

reward_list = []

for g in range(episodes):
    game_over = False
    env.reset()
    total_reward = 0
    
    while not game_over:
        state = np.copy(env.board)
        if random.random() < epsilon:
            action = random.randint(0,3)
        else:
            action = q_table[state_to_str(state)].idxmax()
        reward, game_over = env.play(action)
        total_reward += reward
        if np.sum(env.board) == 4:
            next_state_max_q_value = 0
        else:
            next_state = np.copy(env.board)
            next_state_max_q_value = q_table[state_to_str(state)].max()
        q_table.loc[action, state_to_str(state)] = reward + gamma + next_state_max_q_value
    reward_list.append(total_reward)
print(q_table)

''' ################### Agent가 올바른 전략을 배웠는지 확인 ################### '''
for i in range(2):
    for j in range(2):
        for k in range(2):
            for l in range(2):
                b = np.array([i,j,k,l])
                if len(np.where(b == 0)[0]) != 0:
                    action = q_table[state_to_str(b)].idxmax()
                    pred = q_table[state_to_str(b)].to_list()
                    print('board : {b} \t predicted Q values : {p} \t best action : {a} \t correct action? {s}'.
                          format(b=b, p=pred, a=action, s=b[action] == 0))
                    
# Agent의 보상을 시각화
plt.figure(figsize=(14,7))
plt.plot(range(len(reward_list)), reward_list)
plt.xlabel('Games played')
plt.ylabel('Reward')
plt.show()

''' reference
http://contents2.kocw.or.kr/KOCW/data/document/2020/edu1/bdu/hongseungwook1118/102.pdf'''