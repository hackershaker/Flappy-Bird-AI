import time
import numpy as np
import torch
from agents.dqn_agent import DQNAgent
from flappy_bird_env import FlappyBirdEnv


class DQNPlay:
    def __init__(self, model_path="best_dqn_model.pth", device="cpu"):
        self.env = FlappyBirdEnv(render=True)

        self.agent = DQNAgent(self.env.state_dim, self.env.action_dim, device=device)

        # 학습된 모델 로드
        self.agent.load(model_path)

        # 탐험 안 함
        self.agent.epsilon = 0.0

        # 순수 플레이 -> eval 유지
        self.agent.policy_net.eval()
        self.agent.target_net.eval()

        self.device = device

    def run(self,delay=0.015):
        print("==DQN Play 시작==")

        state=self.env.reset()
        total_reward = 0
        done = False

        while not done:
            # convert state to tensor
            state_arr = np.array(state, dtype=np.float32)
            state_tensor = torch.from_numpy(state_arr).unsqueeze(0).to(self.device)

            # compute Q-value using policy_net
            with torch.no_grad():
                q_values= self.agent.policy_net(state_tensor)

            action = q_values.argmax().item()

            next_state, reward, done = self.env.step(action)
            
            state = next_state
            total_reward = reward

            self.env.render()
            time.sleep(delay)

        print(f"Game Over! Total Reward: {total_reward}")