class BaseAgent:
    def act(self, state):
        """state를 받아 action 반환"""
        raise NotImplementedError

    def learn(self, state, action, reward, next_state, done):
        """state transition으로 학습"""
        raise NotImplementedError

    def save(self, path):
        """학습 결과 저장"""
        raise NotImplementedError

    def load(self, path):
        """학습 결과 불러오기"""
        raise NotImplementedError
