import flappy_bird_gymnasium
import gymnasium as gym
from flappy_bird import Agent
from flappy_bird import TopModel
# from utils import plotLearning
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
import datetime
import os

def resize_and_rgb2gray(image_data):
    # remove the ground from the image
    image_data = image_data[:, :408, :]
    # resize to 84x84 from a 288x512 image
    image_data = cv2.resize(image_data, (84, 84), interpolation=cv2.INTER_AREA)
    # convert to grayscale
    image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)
    # convert to numpy array
    image_data = np.array(image_data)
    return image_data

if __name__ == '__main__':
    env = gym.make('FlappyBird-v0', render_mode='rgb_array')
    env.metadata['render_fps'] = 99999
    
    # de modificat aicisa
    n_frames = 2_000_000
    frames = 0
    cat = 0

    top_model = TopModel()


    agent = Agent(type="train", gamma=0.97, epsilon_start=0.1, lr=0.0001, batch_size=32,
                  eps_end=0.0001, eps_dec=0.0005)

    sum_of_scores, history_episodes = [], []
    total_rewards = []
    
    while frames < n_frames:
        done = False
        ep_reward = 0
        score = 0
        observation, _ = env.reset()
        observation = np.transpose(np.array(env.render()), (1, 0, 2))
        observation = resize_and_rgb2gray(observation)
        frame_que = np.zeros((4, 84, 84))

        for j in range(4):
            frame_que[j] = observation
        
        observation = np.array(frame_que)

        while True:
            action = agent.choose_action(observation)
            observation_, reward, done, _, info = env.step(action)
            observation_ = np.transpose(np.array(env.render()), (1, 0, 2))
            observation_ = resize_and_rgb2gray(observation_)
            
            frame_que = np.roll(frame_que, -1, axis=0)
            frame_que[-1] = observation_
            observation_ = np.array(frame_que)

            frames += 1
            ep_reward += reward
            score = info['score']
            agent.store_transition(observation, action, reward, observation_, done)
            agent.learn(frames)

            observation = observation_

            if done or frames>=n_frames:
                break
        
        total_rewards.append(ep_reward)
        sum_of_scores.append(score)
        history_episodes.append(frames)

        # get hour
        hour = datetime.datetime.now().strftime("%H%M")
        if score > top_model.score:
            top_model.add_score(score)
            torch.save(agent.Q_eval.state_dict(), f"top_model/model{top_model.contor}_{score}_{hour}.pth")
            top_model.contor += 1

        if top_model.last_episode_score != -1:
            if score >= top_model.acceptable_score:
                if top_model.last_episode_score >= top_model.acceptable_score:
                    print("decreased epsilon")
                    agent.epsilon_start -= 0.15
                    if agent.epsilon_start - agent.eps_min < 0:
                        agent.epsilon_start = agent.eps_min
                top_model.last_episode_score = score
            elif score >= 10 and top_model.last_episode_score > 10:
                print("decreased epsilon smool")
                agent.epsilon_start -= 0.05
                if agent.epsilon_start - agent.eps_min < 0:
                    agent.epsilon_start = agent.eps_min
                top_model.last_episode_score = score
            else:
                top_model.last_episode_score = score


        if frames // 20_000 != cat:
            cat = frames // 20_000
            
            time = datetime.datetime.now().strftime("%m_%d_%H-%M-%S")
            plt.plot(history_episodes, sum_of_scores)
            plt.xlabel("Episodes")
            plt.ylabel("Score")
            #save plot
            os.mkdir(f"models/model_{time}")
            plt.savefig(f"models/model_{time}/score.png")
            plt.plot(history_episodes, total_rewards)
            plt.xlabel("Episodes")
            plt.ylabel("Reward")
            plt.savefig(f"models/model_{time}/reward.png")
            # save torch
            torch.save(agent.Q_eval.state_dict(), f"models/model_{time}/model_flappy.pth")


        print(f"Episode: {len(sum_of_scores)}, Score: {score}, Reward: {ep_reward}, Frames: {frames}, Epsilon: {agent.epsilon}")
        #save to file the history (history.txt)
        with open("history.txt", "a") as f:
            f.write(f"Episode: {len(sum_of_scores)}, Score: {score}, Reward: {ep_reward}, Frames: {frames}, Epsilon: {agent.epsilon}\n")

    
    # get time
    time = datetime.datetime.now().strftime("%m_%d_%H-%M-%S")
    plt.plot(history_episodes, sum_of_scores)
    plt.xlabel("Episodes")
    plt.ylabel("Score")
    #save plot
    os.mkdir(f"models/model_{time}")
    plt.savefig(f"models/model_{time}/score.png")
    plt.plot(history_episodes, total_rewards)
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.savefig(f"models/model_{time}/reward.png")
    # save torch
    torch.save(agent.Q_eval.state_dict(), f"models/model_{time}/model_flappy.pth")