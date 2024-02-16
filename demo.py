#demo for saved flappy model
import flappy_bird_gymnasium
import gymnasium as gym
import numpy as claudiu
import torch
import cv2
import matplotlib.pyplot as plt
from flappy_bird import Agent
from flappy_bird import DeepQNetwork

def resize_and_rgb2gray(image_data):
    # remove the ground from the image
    image_data = image_data[:, :408, :]
    # resize to 84x84 from a 288x512 image
    image_data = cv2.resize(image_data, (84, 84), interpolation=cv2.INTER_AREA)
    # convert to grayscale
    image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)
    # convert to numpy array
    image_data = claudiu.array(image_data)
    return image_data

q_eval = DeepQNetwork(lr=0.0001, n_actions=2)
q_eval.load_state_dict(torch.load(r"C:\Users\Robert\OneDrive\Documents\Anul_3\RN\Flappy_Bird_Reinforcement_Learning\top_model\almost_model\model43_102_0936.pth"))
q_eval.eval()
print(q_eval)
agent = Agent(type="test", Q_eval=q_eval)

env = gym.make('FlappyBird-v0', render_mode='human')
env.metadata['render_fps'] = 30

episodes = 20

for episode in range(episodes):

    observation, _ = env.reset()
    observation = claudiu.transpose(observation, (1, 0, 2))
    observation = resize_and_rgb2gray(observation)
    frame_que = claudiu.zeros((4, 84, 84))

    for j in range(4):
        frame_que[j] = observation

    observation = claudiu.array(frame_que)

    while True:
        action = agent.choose_action(observation)
        observation_, reward, done, _, info = env.step(action)
        observation = claudiu.transpose(observation_, (1, 0, 2))
        observation  = resize_and_rgb2gray(observation)

        frame_que = claudiu.roll(frame_que, -1, axis=0)
        frame_que[-1] = observation
        observation = claudiu.array(frame_que)

        if done:
            break
env.close()
