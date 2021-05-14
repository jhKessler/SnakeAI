from src.board import Board
from src.snake import Snake
import torch

class Game(object):
    """ Object of Class Game representing an Instance of the snake Game"""


    @staticmethod
    def createSnake(x: int, y:int) -> Snake:
        return Snake(x, y)


    def __init__(self, size):
        """Creates an Instance of the Class Game"""
        self.x_start = self.y_start = size // 2
        self.snake = Game.createSnake(self.x_start, self.y_start)
        self.board = Board(size, self.snake.getPartPositions())


    def restart(self):
        """Resets the game"""
        self.snake = Game.createSnake(self.x_start, self.y_start)
        self.board.reset(self.snake.getPartPositions())


    def gameStep(self, move: str):
        """Executes a timestep of the game"""
        self.snake.changeDirection(move)
        self.snake.move()
        positions = self.snake.getPartPositions()
        event = self.board.validateMove(positions)
        if event == "eat":
            self.snake.addPart()
        elif event == "death":
            self.restart()
        return event


    def getGamestate(self) -> torch.tensor:
        """Get Gamestate"""
        return self.board.getState()


    def getScore(self) -> int:
        """Get score of snake"""
        return len(self.snake)


    def inputMove(self, move: str):
        """Handles input from AI"""
        return self.gameStep(move)
