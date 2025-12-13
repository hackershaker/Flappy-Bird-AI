from collections import deque
from random import randrange
import random
import numpy as np
from torch import FloatTensor, no_grad, optim
import torch
from torch.nn import Linear, MSELoss, Module, ReLU, Sequential

from agents.base_agent import BaseAgent


class DQNNetwork(Module):
    """
    Deep Q Network 모델 정의
    입력 : state dimension
    출력 : action dimension
    """

    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.model = Sequential(
            Linear(state_dim, 128),
            ReLU(),
            Linear(128, 128),
            ReLU(),
            Linear(128, action_dim),
        )

    def forward(self, x):
        return self.model(x)


class DQNAgent(BaseAgent):
    def __init__(
        self,
        state_dim,
        action_dim,
        gamma=0.99,
        lr=0.0005,
        batch_size=64,
        memory_size=50000,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.05,
        target_update=1000,
        device="cpu",
    ):
        self.state_dim = state_dim  # 입력 차원 수
        self.action_dim = action_dim  # 출력 차원 수

        self.gamma = gamma  # 학습률 γ
        self.batch_size = batch_size  # 배치 사이즈
        self.device = device  # cpu,gpu 학습 방법 지정

        self.epsilon = epsilon  # ε-greedy 하이퍼파라미터
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.memory = deque(maxlen=memory_size)

        self.policy_net = DQNNetwork(state_dim, action_dim).to(device)
        self.target_net = DQNNetwork(state_dim, action_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # 타깃 네트워크는 추론 전용

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = MSELoss()

        self.learn_step = 0
        self.target_update = target_update

        print(">>> DQN init: state_dim=", state_dim, " action_dim=", action_dim)
        print(">>> policy model:", self.policy_net)

    def act(self, state):
        """
        행동 선택

        :param self: Description
        :param state: Description
        """
        if np.random.rand() < self.epsilon:
            return randrange(self.action_dim)

        state = FloatTensor(state).unsqueeze(0).to(self.device)

        with no_grad():
            q_values = self.policy_net(state)

        return q_values.argmax().item()
    
    def act_eval(self, state):
        """
        행동 평가, 항상 greedy
        
        :param self: Description
        :param state: Description
        """
        state = FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state)
        return q_values.argmax(dim=1).item()

    def memorize(self, state, action, reward, next_state, done):
        """
        데이터 샘플들 사이의 Temporal correlations를 해결하기 위한 buffer

        :param self: deep q learning agent class
        :param state: current state
        :param action: Description
        :param reward: Description
        :param next_state: Description
        :param done: Description
        """
        self.memory.append((state, action, reward, next_state, done))

    def learn(self):
        """
        학습 정의 함수

        :param self: Description
        """
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)

        state, action, reward, next_state, done = zip(*minibatch)

        states = torch.tensor(np.array(state), dtype=torch.float32, device=self.device)
        next_states = torch.tensor(
            np.array(next_state), dtype=torch.float32, device=self.device
        )
        actions = torch.tensor(np.array(action), dtype=torch.long, device=self.device)
        rewards = torch.tensor(
            np.array(reward), dtype=torch.float32, device=self.device
        )
        dones = torch.tensor(np.array(done), dtype=torch.float32, device=self.device)

        # Q(s,a) from policy_net
        q_values = self.policy_net(states)
        q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # target: r + γ max_a' Q_target(s', a')
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            target = rewards + (1 - dones) * self.gamma * next_q

        # loss
        loss = self.loss_fn(q_value, target)

        # optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # target network soft update
        self.learn_step += 1
        if self.learn_step % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def decaying_epsilon(self):
        # epsilon 감소
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            if self.epsilon < self.epsilon_min:
                self.epsilon = self.epsilon_min

    def save(self, path):
        torch.save(
            {
                "policy_net": self.policy_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "epsilon": self.epsilon,
            },
            path,
        )

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint["policy_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.epsilon = checkpoint.get("epsilon", self.epsilon)
        self.policy_net.eval()
        self.target_net.eval()
