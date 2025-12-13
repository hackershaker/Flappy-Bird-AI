class BaseAgent:
    def act(self, state):
        """agent가 state를 인자로 받아 보고 action을 선택하여 반환"""
        raise NotImplementedError

    def act_eval(self, state):
        pass

    def observe(self, state, action, reward, next_state, done):
        """
        transition을 받아 buffer에 저장하거나 즉시 학습하는 역할

        Q-learning → observe() 안에서 바로 Q 업데이트
        DQN → observe()에서 buffer에 저장
        PPO → observe()에서 trajectory에 저장
        """
        raise NotImplementedError

    def learn(self, state, action, reward, next_state, done):
        """state transition으로 학습"""
        raise NotImplementedError

    def learn_transition(self, state, action, reward, next_state, done):
        """구 Q-learning 방식 학습 구조"""
        raise NotImplementedError

    def save(self, path):
        """학습 결과 저장"""
        raise NotImplementedError

    def load(self, path):
        """학습 결과 불러오기"""
        raise NotImplementedError
