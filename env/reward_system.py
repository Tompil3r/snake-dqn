

class RewardSystem():
    def __init__(self, reward_functions):
        self.reward_functions = reward_functions

    def get_reward(self, game_code, *args):
        return self.reward_functions[game_code](*args)