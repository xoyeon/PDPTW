'''참고논문
Courier routing and assignment for food delivery service using reinforcement learning'''

import pandas as pd
import random
import numpy as np

import matplotlib.pyplot as plt
from time import time

seed = 1
np.random.seed(seed)

# 데이터 불러오기
orders = pd.read_csv('./data/orders.csv')
couriers = pd.read_csv('./data/couriers.csv')

# 주문 건 무작위 추출
select_order = orders.sample(n=1)
print("receive order : ", select_order)
rest_x = np.array(select_order['ox'])
rest_y = np.array(select_order['oy'])
placement_time = np.array(select_order['placement_time'])
ready_time = np.array(select_order['ready_time'])


# 근무중인 라이더
# rider = couriers[(couriers['on_time'] <= order[6]) & (couriers['off_time'] >= order[7])]
select_rider = couriers[(couriers['on_time'] <= 268) & (couriers['off_time'] >= 276)]
data_len = len(select_rider)
print("working couriers : ", data_len)

# 속성 값
name = np.array(select_rider['courier'])
rider_x = np.array(select_rider['x'])
rider_y = np.array(select_rider['y'])
on_time = np.array(select_rider['on_time'])
off_time = np.array(select_rider['off_time'])
delivered_num = np.random.uniform(low=3, high=25, size=data_len).astype(int)
trustiness = np.random.uniform(low=0.5, high=5.0, size=data_len).astype(float).round(2)

rider = np.stack((name, rider_x, rider_y, on_time, off_time, delivered_num, trustiness), axis=1)
print("rider info. : ",rider)


# 나의 state는 배달원(x,y), 음식점(x,y), 근무시간(on,off), 배달건수(delivery), 신뢰도(trustiness) --> 6가지
# 거리는 추후 고려--> 일단 모두 시간 내 배달할 수 있다고 가정,,
# 나의 action은 배달원 지정. 동,서,남,북,음식픽업 --> 거리를 고려하게 된다면 deliver도 추가

'''goal : 배달 처리량 증가 + 고객 만족
action = 배달원 지정
state = 배달원들과 음식점의 위치, 근무 시간... 등
reward = 배달 건 수 std 작게, trustiness 크게'''


# GridWorld Env
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