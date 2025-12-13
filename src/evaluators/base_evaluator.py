class BaseEvaluator:
    def evaluate(self, agent, env, episodes=10):
        raise NotImplementedError
