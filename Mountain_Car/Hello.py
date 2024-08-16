print("Bu program Kuzey T. Kale tarafından yazılmıştır. Bu porgramın çalışması için Python programının versiyonunun (sürümünün) en az 3.8.0 olması gerekmektedir. (Bilgisayarınızda bulunan Python programının versiyonunu (sürümünü) öğrenmek için Komut İstemcisine 'python --version' komutunu girebilirsiniz) Bu programın çalışması için olması gereken kütüphaneler şunlardır: gymnasium, numpy, matplotlib.pyplot, pickle") 

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle


def run(episodes , is_traing=True, render=False):

    env = gym.make("MountainCar-v0", render_mode="human" if render else None)

    pos_space = np.linspace(env.observation_space.low[0], env.observation_space.high[0], 20)
    vel_space = np.linspace(env.observation_space.low[1], env.observation_space.high[1], 20)

    if (is_traing):
        q = np.zeros((len(pos_space), len(vel_space), env.action_space.n)) # 20*20*3
    else:
        f = open('Hello.py.mountain_car.plk', 'rb')
        q = pickle.load(f)
        f.close()

    learning_rate_a = 0.9
    discount_factor_g = 0.9

    epsilon = 1
    epsilon_decay_rate = 2/episodes
    rng = np.random.default_rng()

    rewards_per_episode = np.zeros(episodes)

    for i in range(episodes):
        
        if is_traing:
            print(i, " | ", episodes)

        state = env.reset()[0]
        state_p = np.digitize(state[0], pos_space)
        state_v = np.digitize(state[1], vel_space)

        terminated = False

        rewards = 0

        while not terminated and rewards >- 1000:

            if is_traing and rng.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q[state_p, state_v, :])
            
            new_state,reward,terminated,_,_ = env.step(action)
            new_state_p = np.digitize(new_state[0], pos_space)
            new_state_v = np.digitize(new_state[1], vel_space)

            if is_traing:
                q[state_p, state_v, action] = q[state_p, state_v, action] + learning_rate_a * (
                    reward + discount_factor_g*np.max(q[new_state_p, new_state_v, :]) - q[state_p, state_v, action]
                )

            state = new_state
            state_p = new_state_p
            state_v = new_state_v

            rewards += reward

        epsilon = max(epsilon - epsilon_decay_rate, 0)

        rewards_per_episode[i] = rewards

    env.close()

    if is_traing:
        f = open('Hello.py.mountain_car.plk', 'wb')
        pickle.dump(q, f)
        f.close
    
    mean_rewards = np.zeros(episodes)
    for t in range(episodes):
        mean_rewards[t] = np.mean(rewards_per_episode[max(0, t-100):(t+1)])
    plt.plot(mean_rewards)
    plt.savefig(f'Hello.py.mountain_car.png')

    if is_traing == True:
        print(episodes, " | ", episodes)
        run(10, is_traing=False, render=True)
        


if __name__ == "__main__":


    def Traing():

        traing = input("Bu bir eğtim mi? [(T)rue/(F)alse]: ")

        if traing == 'T' or traing == 't':
            new_traing = True
            return new_traing
        else:
            if traing == 'F' or traing == 'f':
                new_traing = False
                return new_traing
            else:
                print("Anlaşılamadı lütfen tekrar girinz! ")
                Traing()
        
    new_traing = Traing()
    
    if new_traing == True:
        episodes = int(input("Kendimi kaç kez eğitiyim?/Sınavdan kaç kez geçeyim? (Lütfen sayı girin) (Genelde 1000 yazılır) : "))
        old_render = False
        print("Eğitim bittikten sonra eğtim modeli kaydedilir ve eğitim sonu modeli 10 defa çalıştırılır.")
        
    if new_traing == False:
        episodes = int(input("Kaç defa göstereyim? (Lütfen sayı girin) (Genelde 10 yazılır) : "))
        old_render = True
    
    devam = input("Programı başlatmak için 'B' tuşuna ve ardından enter tuşuna basın. (Başka bir tuşa ve ardından enter tuşuna basarsanız program sonlandırılır) : ")

    if devam == 'B' or devam == 'b':
        run(episodes, is_traing=new_traing, render=old_render)
    else:
        print("Program sonlandırıldı.")
        print("Güle güle...")