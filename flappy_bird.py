import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class TopModel():
    """
        stochez care e cel mai bun scor
        salvez top 10 scoruri si modele
    """
    def __init__(self, top_size=50):
        self.top_size = top_size
        self.top_scores = []
        self.score = 65
        self.contor = 0
        self.last_episode_score = -1
        self.acceptable_score = 25
    
    def add_score(self, score):
        if len(self.top_scores) < self.top_size:
            self.top_scores.append(score)



class DeepQNetwork(nn.Module):
    def __init__(self, lr, n_actions):
        # lr: learning rate
        # input_dims: input dimensions
        # fc1_dims: first fully connected layer dimensions
        # fc2_dims: second fully connected layer dimensions
        # n_actions: number of actions
        super(DeepQNetwork, self).__init__()
        self.n_actions = n_actions

        self.conv1 = nn.Conv2d(4, 32, 8, 4)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 64, 4, 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(64, 64, 3, 1)
        self.relu3 = nn.ReLU(inplace=True)
        self.fc4 = nn.Linear(3136, 512)
        self.relu4 = nn.ReLU(inplace=True)
        self.fc5 = nn.Linear(512, self.n_actions)

        # Optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        # Loss function
        self.criterion = nn.MSELoss()
        # Device
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # Send model to device
        self.to(self.device)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        out = self.relu3(out)
        out = out.view(out.size()[0], -1)
        out = self.fc4(out)
        out = self.relu4(out)
        out = self.fc5(out)

        return out


class CustomMemory():
    def __init__(self, capacity, input_dims):
        self.capacity = capacity
        self.mem_cntr = 0

        self.state_buffer = np.zeros((self.capacity, *input_dims), dtype=np.float32)
        self.new_state_buffer = np.zeros((self.capacity, *input_dims), dtype=np.float32)
        self.action_buffer = np.zeros(self.capacity, dtype=int)
        self.reward_buffer = np.zeros(self.capacity, dtype=np.float32)
        self.terminal_buffer = np.zeros(self.capacity, dtype=bool)

    def store_transition(self, state, action, reward, new_state, terminal):
        index = self.mem_cntr % self.capacity
        self.state_buffer[index] = state
        self.new_state_buffer[index] = new_state
        self.reward_buffer[index] = reward
        self.action_buffer[index] = action
        self.terminal_buffer[index] = terminal

        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_capacity = min(self.mem_cntr, self.capacity)

        batch = np.random.choice(max_capacity, batch_size, replace=False)

        state_batch = self.state_buffer[batch]
        new_state_batch = self.new_state_buffer[batch]
        reward_batch = self.reward_buffer[batch]
        terminal_batch = self.terminal_buffer[batch]
        action_batch = self.action_buffer[batch]

        return state_batch, action_batch, reward_batch, new_state_batch, terminal_batch


class Agent():
    def __init__(self, type, gamma=0.95, epsilon_start=0.1, lr=0.0001, batch_size=32, n_actions=2, 
                 max_capacity=50_000, eps_end=0.01, eps_dec=0.5e-4,
                 n_steps=1_000_000, Q_eval=None, R_memory=None, target_copy_rate=1000):
        self.type = type
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon = epsilon_start
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.lr = lr
        self.action_space = [i for i in range(n_actions)]
        self.mem_size = max_capacity
        self.batch_size = batch_size
        self.n_steps = n_steps
        self.target_copy_rate = target_copy_rate
        self.epsilon_decrements = np.linspace(self.epsilon, self.eps_min, self.n_steps)

        if Q_eval is None:
            self.Q_eval = DeepQNetwork(self.lr, n_actions)
        else:
            self.Q_eval = Q_eval
        
        self.Q_target = DeepQNetwork(self.lr, n_actions)
        self.Q_target.load_state_dict(self.Q_eval.state_dict())
        
        if R_memory is None:
            self.R_memory = CustomMemory(self.mem_size, (4, 84, 84))
        else:
            self.R_memory = R_memory

    def store_transition(self, state, action, reward, new_state, terminal): 
        self.R_memory.store_transition(state, action, reward, new_state, terminal)

    def choose_action(self, observation):
        if self.type == "test" or np.random.random() > self.epsilon:
        # if self.type == "test" or self.type == "train":
            state = torch.tensor(observation, dtype=torch.float32).unsqueeze(0).to(self.Q_eval.device)
            actions = self.Q_eval.forward(state)
            action = torch.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)

        return action
    
    def learn(self, steps_done):
        if self.R_memory.mem_cntr < self.batch_size:
            return

        if steps_done % self.target_copy_rate == 0:
            self.Q_target.load_state_dict(self.Q_eval.state_dict())

        self.Q_eval.optimizer.zero_grad()

        batch_index = np.arange(self.batch_size, dtype=np.int64)

        state_batch, action_batch, reward_batch, new_state_batch, terminal_batch = self.R_memory.sample_buffer(self.batch_size)
        
        state_batch = torch.tensor(state_batch).to(self.Q_eval.device)
        new_state_batch = torch.tensor(new_state_batch).to(self.Q_eval.device)
        reward_batch = torch.tensor(reward_batch).to(self.Q_eval.device)
        terminal_batch = torch.tensor(terminal_batch).to(self.Q_eval.device)

        q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch]
        q_next = self.Q_target.forward(new_state_batch)
        q_next[terminal_batch] = 0.0

        q_target = reward_batch + self.gamma * torch.max(q_next, dim=1)[0]

        loss = self.Q_eval.criterion(q_target, q_eval).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()

        # self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min \
        #                else self.eps_min
        # self.epsilon = self.eps_min + (self.epsilon - self.eps_min) * np.exp(-1. * steps_done / self.n_steps)
        # self.epsilon = self.epsilon_decrements[steps_done]
        delta_epsilon = (self.epsilon_start - self.eps_min) / (self.n_steps )
        self.epsilon = max(self.eps_min, self.epsilon_start - delta_epsilon * steps_done)
