#!/usr/bin/env python
# coding: utf-8

# # 데이터 불러오기

# In[1]:


import pandas as pd
import numpy as np

orders = pd.read_csv('./data/orders.csv')
couriers = pd.read_csv('./data/couriers.csv')


# In[2]:


orders.head()


# In[3]:


couriers.head()


# # 주문 하나 -- 근무 라이더

# In[4]:


# 주문 건 무작위 추출
np.random.seed(1)
order = orders.sample(n=1)
order


# In[5]:


# 근무중인 라이더
# rider = couriers[(couriers['on_time'] <= order[6]) & (couriers['off_time'] >= order[7])]
rider = couriers[(couriers['on_time'] <= 268) & (couriers['off_time'] >= 276)]
rider


# # Q-learning
# - goal : 배달 처리량 증가 + 고객 만족
# 
# - action = 배달원 지정
# - state = 배달원들과 음식점의 위치, 근무 시간
# - reward = 배달 건 수 std 작게, trustiness 크게

# https://www.learndatasci.com/tutorials/reinforcement-q-learning-scratch-python-openai-gym/
# 
# https://blog.floydhub.com/an-introduction-to-q-learning-reinforcement-learning/
# 
# Breaking it down into steps, we get
# 
# - Initialize the Q-table by all zeros.
# - Start exploring actions: For each state, select any one among all possible actions for the current state (S).
# - Travel to the next state (S') as a result of that action (a).
# - For all possible actions from the state (S') select the one with the highest Q-value.
# - Update Q-table values using the equation.
# - Set the next state as the current state.
# - If goal state is reached, then end and repeat the process.

# In[6]:


# 나의 state는 배달원(x,y), 음식점(x,y), 근무시간(on,off) --> 6가지
# 나의 action은 배달원 지정. 동,서,남,북,음식픽업 --> 5가지


# # Create Grid World Env

# In[7]:


riders = len(rider)
on_time = rider['on_time']
off_time = rider['off_time']
delivery_std = rider['delivery']
trustiness = rider['trustiness']
riders_x = rider['x']
riders_y = rider['y']

restaurant_x = order['rx']
restaurant_y = order['ry']
placement_time = order['placement_time']
ready_time = order['ready_time']


# In[8]:


grid = np.zeros((5,5))
grid


# In[9]:


Y = .90  #discount value
for num in range(10): #number of times we will go through the whole grid
  for i in range(5):      #all the rows
    for j in range(5):    #all the columns
      
      up_grid = grid[i-1][j] if i > 0 else 0   #if going up takes us out of the grid then its value be 0
      down_grid = grid[i+1][j] if i < 4 else 0  #if going down takes us out of the grid then its value be 0
      left_grid = grid[i][j-1] if j > 0 else 0  #if going left takes us out of the grid then its value be 0
      right_grid = grid[i][j+1] if j < 4 else 0  #if going right takes us out of the grid then its value be 0

      all_dirs = [up_grid, down_grid, left_grid, right_grid]     

      value=0  
      if i==0 and j==1: # the position of A
        value = 10 + Y*grid[4][1]
      elif i==0 and j==3: # the position of B
        value = 5 + Y*grid[2][3]
      else:
        for direc in all_dirs:
          if direc != 0: 
            value += .25 * (0 + Y*direc)  #if we don't go out of the grid
          else:
            value += .25 * (-1 + Y*grid[i][j]) #if we go out of the grid
        
      grid[i][j] = value


# In[10]:


np.round(grid, 1)


# # Q-learning

# In[ ]:




