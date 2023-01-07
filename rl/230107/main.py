import pandas as pd
import numpy as np

import env
import agent

import random
import pickle
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
SEEDS = (12, 34, 56, 78, 90)

%matplotlib inline
plt.style.use('fivethirtyeight')
params = {
    'figure.figsize': (15, 8),
    'font.size': 24,
    'legend.fontsize': 20,
    'axes.titlesize': 28,
    'axes.labelsize': 24,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20
}
pylab.rcParams.update(params)
np.set_printoptions(suppress=True)

SEEDS = (12, 34, 56, 78, 90) # 차량수 실험에서만 순서 변경 # 원래는  (12, 34, 56, 78, 90)


ddpg_results = []
best_agent, best_eval_score = None, float('-inf')


with open("train_set.pickle" , 'rb') as f:
    train_data = pickle.load(f)

with open("test_set.pickle" , 'rb') as f:
    eval_data = pickle.load(f)
val_data = eval_data[:200]


print("Data set: train %d, val %d, test %d" %(len(train_data), len(val_data), len(eval_data)))


model_count =0
for i in range(5):

    seed = SEEDS[i]
    model_number ="Test_Z%d_C%s_S%d_seed%d" %(int(0.8*100), int(0.2*100), 5, seed)
    noise_std =0.12

    environment_settings = {
        'env_name': 'R', #RV, R, V
        'gamma': 0.99,
        'max_minutes': 120,
        'max_episodes': 300,
        'goal_mean_100_reward': 100
    }

    experiment_settings = {
        'zipf_param': 0.8, 
        'RSU_num': 36, 
        'vehicle_num':100, 
        'RSU_capa': 0.2, 
        'vehicle_capa': 0.0, 
        'content_num': 20, 
        'alpha':0.8, 
        'seg':5 
    }

    # 학습에 사용할 모델 세팅
    policy_model_fn = lambda nS, bounds: ddpg_lib.FCDP(nS, bounds, hidden_dims=(512,512))
    policy_max_grad_norm = float('inf')
    policy_optimizer_fn = lambda net, lr: optim.Adam(net.parameters(), lr=lr)
    policy_optimizer_lr = 0.0003 #0.0003

    value_model_fn = lambda nS, nA: ddpg_lib.FCQV(nS, nA, hidden_dims=(512,512))
    value_max_grad_norm = float('inf')
    value_optimizer_fn = lambda net, lr: optim.Adam(net.parameters(), lr=lr)
    value_optimizer_lr = 0.0003 # 0.0003

    training_strategy_fn = lambda bounds: ddpg_lib.NormalNoiseStrategy(bounds, exploration_noise_ratio=noise_std) #원래 0.1                                                     
    evaluation_strategy_fn = lambda bounds: ddpg_lib.GreedyStrategy(bounds)

    replay_buffer_fn = lambda: ddpg_lib.ReplayBuffer(max_size=10000, batch_size=256)
    n_warmup_batches = 5 #원래 5
    update_target_every_steps = 400 #원래 1
    tau = 0.005 # 원래 0.005
    
    env_name, gamma, max_minutes, max_episodes, goal_mean_100_reward = environment_settings.values()
    zipf_param, rsu_num, vehicle_num, rsu_capa, vehicle_capa, content_num, alpha, seg_num = experiment_settings.values()

    
    agent = ddpg_lib.DDPG(replay_buffer_fn,
                 policy_model_fn, 
                 policy_max_grad_norm, 
                 policy_optimizer_fn, 
                 policy_optimizer_lr,
                 value_model_fn, 
                 value_max_grad_norm, 
                 value_optimizer_fn, 
                 value_optimizer_lr, 
                 training_strategy_fn,
                 evaluation_strategy_fn,
                 n_warmup_batches,
                 update_target_every_steps,
                 tau,
                 model_number)

    # 환경 초기화
    urban_env = env.Environment(env_name, 
                            zipf_param, 
                            rsu_num, 
                            vehicle_num, 
                            rsu_capa, 
                            vehicle_capa, 
                            content_num, 
                            alpha, 
                            seg_num)

    # 환경 데이터 설정
    # urban_env.get_vehicle_data(train_data)
    
    result, final_eval_score, training_time, wallclock_time = agent.train(
        urban_env, seed, gamma, max_minutes, max_episodes, goal_mean_100_reward, 
        rsu_num, vehicle_num, content_num, zipf_param,
        train_data, val_data, eval_data)
    ddpg_results.append(result)
    if final_eval_score > best_eval_score:
        best_eval_score = final_eval_score
        best_agent = agent

ddpg_results = np.array(ddpg_results)




"""
참고: 
https://goodboychan.github.io/book/GDRL-chapter-12.html
"""