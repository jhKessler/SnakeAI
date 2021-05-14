from src.game import Game
import torch

class SnakeEnv(object):


    reward_dict = {
        "death": -1,
        "none": 0,
        "eat": 1
    }


    def __init__(self, gamesize: int):
        self.gamesize = gamesize
        self.game = Game(size=self.gamesize)


    def getState(self) -> torch.tensor:
        return self.game.getGamestate()


    def step(self, action: str):
        """Executes a timestep with the chosen action.\n\n
                returns:
                    previous_state (torch.tensor): gamestate before action is taken \n
                    observation (torch.tensor): gamestate after action is taken \n
                    reward (int): number representing reward after action is taken \n
                    episode_over (boolean): boolean representing whether current game has terminated"""
        previous_state = self.getState()

        reward = self.game.inputMove(action)
        observation = self.getState()
        episode_over = reward == "death"
        if episode_over:
            self.reset()
        reward = SnakeEnv.reward_dict[reward]
        return previous_state, observation, reward, episode_over


    def reset(self):
        self.game.restart()


    def getScore(self) -> int:
        return self.game.getScore()
