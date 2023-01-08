'''참고논문
Courier routing and assignment for food delivery service using reinforcement learning'''

import pandas as pd
import numpy as np

seed = 1
np.random.seed(seed)

# 데이터 불러오기
orders = pd.read_csv('./data/orders.csv')
couriers = pd.read_csv('./data/couriers.csv')

# 주문 건 무작위 추출
select_order = orders.sample(n=1)
print("receive order : ", select_order)
rest_x = select_order['ox']
rest_y = select_order['oy']
placement_time = select_order['placement_time']
ready_time = select_order['ready_time']

# 근무중인 라이더
# rider = couriers[(couriers['on_time'] <= order[6]) & (couriers['off_time'] >= order[7])]
select_rider = couriers[(couriers['on_time'] <= 268) & (couriers['off_time'] >= 276)]
rider_len = len(select_rider)
print("working couriers : ", rider_len)

# 속성 값
name = np.array(select_rider['courier'])
rider_x = np.array(select_rider['x'])
rider_y = np.array(select_rider['y'])
on_time = np.array(select_rider['on_time'])
off_time = np.array(select_rider['off_time'])
delivered_num = np.random.uniform(low=3, high=25, size=rider_len).astype(int)
trustiness = np.random.uniform(low=0.5, high=5.0, size=rider_len).astype(float).round(2)

rider = np.stack((name, rider_x, rider_y, on_time, off_time, delivered_num, trustiness), axis=1)
print("rider info. : ",rider)


# 나의 state는 배달원(x,y), 음식점(x,y), 근무시간(on,off), 배달건수(delivery), 신뢰도(trustiness) --> 6가지
# 거리는 추후 고려--> 일단 모두 시간 내 배달할 수 있다고 가정,,
# 나의 action은 배달원 지정. 동,서,남,북,음식픽업 --> 거리를 고려하게 된다면 deliver도 추가

'''goal : 배달 처리량 증가 + 고객 만족
action = 배달원 지정
state = 배달원들과 음식점의 위치, 근무 시간... 등
reward = 배달 건 수 std 작게, trustiness 크게'''

'''########################## ENVIRONMENT #############################'''
import os, sys, random, operator

class Environment:
    
    def __init__(self, Ny=5, Nx=5):  # 5*5 grid
        # Define state space
        self.Ny = Ny  # y grid size
        self.Nx = Nx  # x grid size6
        self.state_dim = (Ny, Nx)
        # Define action space
        self.action_dim = (5,)  # up, right, down, left, pick-up
        self.action_dict = {"up": 0, "right": 1, "down": 2, "left": 3} #, "pick-up": 4,}
        self.action_coords =[
            (rider_x[0],rider_y[0]),
            (rider_x[1],rider_y[1]),
            (rider_x[2],rider_y[2])]
        print("action_coords :", self.action_coords)
        # Define rewards table
        self.R = self._build_rewards()  # R(s,a) agent rewards
        # Check action space consistency
        # if len(self.action_dict.keys()) != len(self.action_coords):
        #     exit("err: inconsistent actions given")

    def reset(self):
        # Reset agent state to top-left grid corner
        self.state = (rest_x, rest_y)
        return self.state

    def step(self, action):
        # Evolve agent state
        state_next = (self.state[0] + self.action_coords[action][0],
                      self.state[1] + self.action_coords[action][1])
        # Collect reward
        reward = self.R[self.state + (action,)]
        # Terminate if we reach bottom-right grid corner
        done = (state_next[0] == self.Ny - 1) and (state_next[1] == self.Nx - 1)
        # Update state
        self.state = state_next
        return state_next, reward, done
    
    def allowed_actions(self):
        # Generate list of actions allowed depending on agent grid location
        actions_allowed = []
        y, x = self.state[0], self.state[1]
        print("self.stat : ", self.state[1])
        if (y > 0):  # no passing top-boundary
            actions_allowed.append(self.action_dict["up"])
        if (y < self.Ny - 1):  # no passing bottom-boundary
            actions_allowed.append(self.action_dict["down"])
        if (x > 0):  # no passing left-boundary
            actions_allowed.append(self.action_dict["left"])
        if (x < self.Nx - 1):  # no passing right-boundary
            actions_allowed.append(self.action_dict["right"])
        actions_allowed = np.array(actions_allowed, dtype=int)
        return actions_allowed

    def _build_rewards(self):
        # Define agent rewards R[s,a]
        r_goal = 100  # reward for arriving at terminal state (bottom-right corner)
        r_nongoal = -0.1  # penalty for not reaching terminal state
        R = r_nongoal * np.ones(self.state_dim + self.action_dim, dtype=float)  # R[s,a]
        R[self.Ny - 2, self.Nx - 1, self.action_dict["down"]] = r_goal  # arrive from above
        R[self.Ny - 1, self.Nx - 2, self.action_dict["right"]] = r_goal  # arrive from the left
        return R

'''########################## AGENT #############################'''
class Agent:
    
    def __init__(self, env):
        # Store state and action dimension 
        self.state_dim = env.state_dim
        self.action_dim = env.action_dim
        # Agent learning parameters
        self.epsilon = 1.0  # initial exploration probability
        self.epsilon_decay = 0.99  # epsilon decay after each episode
        self.beta = 0.99  # learning rate
        self.gamma = 0.99  # reward discount factor
        # Initialize Q[s,a] table
        self.Q = np.zeros(self.state_dim + self.action_dim, dtype=float)

    def get_action(self, env):
        # Epsilon-greedy agent policy
        if random.uniform(0, 1) < self.epsilon:
            # explore
            return np.random.choice(env.allowed_actions())
        else:
            # exploit on allowed actions
            state = env.state;
            actions_allowed = env.allowed_actions()
            Q_s = self.Q[state[0], state[1], actions_allowed]
            actions_greedy = actions_allowed[np.flatnonzero(Q_s == np.max(Q_s))]
            return np.random.choice(actions_greedy)

    def train(self, memory):
        # -----------------------------
        # Update:
        #
        # Q[s,a] <- Q[s,a] + beta * (R[s,a] + gamma * max(Q[s,:]) - Q[s,a])
        #
        #  R[s,a] = reward for taking action a from state s
        #  beta = learning rate
        #  gamma = discount factor
        # -----------------------------
        (state, action, state_next, reward, done) = memory
        sa = state + (action,)
        self.Q[sa] += self.beta * (reward + self.gamma*np.max(self.Q[state_next]) - self.Q[sa])

    def display_greedy_policy(self):
        # greedy policy = argmax[a'] Q[s,a']
        greedy_policy = np.zeros((self.state_dim[0], self.state_dim[1]), dtype=int)
        for x in range(self.state_dim[0]):
            for y in range(self.state_dim[1]):
                greedy_policy[y, x] = np.argmax(self.Q[y, x, :])
        print("\nGreedy policy(y, x):")
        print(greedy_policy)
        print()


'''########################## SETTINGS #############################'''
env = Environment(Ny=8, Nx=8)
agent = Agent(env)

'''########################## TRAIN AGENT #############################'''
print("\nTraining agent...\n")
N_episodes = 500
for episode in range(N_episodes):

    # Generate an episode
    iter_episode, reward_episode = 0, 0
    state = env.reset()  # starting state
    while True:
        action = agent.get_action(env)  # get action
        state_next, reward, done = env.step(action)  # evolve state by action
        agent.train((state, action, state_next, reward, done))  # train agent
        iter_episode += 1
        reward_episode += reward
        if done:
            break
        state = state_next  # transition to next state

    # Decay agent exploration parameter
    agent.epsilon = max(agent.epsilon * agent.epsilon_decay, 0.01)

    # Print
    if (episode == 0) or (episode + 1) % 10 == 0:
        print("[episode {}/{}] eps = {:.3F} -> iter = {}, rew = {:.1F}".format(
            episode + 1, N_episodes, agent.epsilon, iter_episode, reward_episode))

    # Print greedy policy
    if (episode == N_episodes - 1):
        agent.display_greedy_policy()
        for (key, val) in sorted(env.action_dict.items(), key=operator.itemgetter(1)):
            print(" action['{}'] = {}".format(key, val))
        print()