import numpy as np
from random import randint

class Board(object):
    """A Class representing the Gameboard.
        Objects of type Snake are represented on the board as 1s,
        Objects of type Apple are represented on the board as 2s
        Empty Places are represented as 0s"""

    object_dict = {
        "empty": 0,
        "snake": 1,
        "snakehead": 2,
        "apple": 3
    }

    @staticmethod
    def randomCoordinates(size: int) -> tuple:
        """Returns random coordinates, takes size of game field as input"""
        return randint(0, size-1), randint(0, size-1)

    @staticmethod
    def createBoard(size) -> np.ndarray:
        """Creates an empty Board"""
        return np.ones((size, size))

    def __init__(self, size: int, positions: list):
        """Creates a board object"""
        assert 0 < size, "Boardsize must be greater than 0"
        self.size = size
        self.reset(positions)
        
    def reset(self, positions: list):
        """Resets the board to a starting state"""
        self.board = Board.createBoard(self.size)
        self.applyPositions(positions, True)

    def access(self, x: int, y: int) -> np.ndarray:
        """Returns the board value at a specified coordinate"""
        return self.board[y][x]

    def setState(self, x: int, y: int, obj: str):
        """Sets the board to a specific object at a specified position"""
        obj = obj.lower()
        assert obj in Board.object_dict.keys(), f"Unknown Object \"{obj}\""
        self.board[y][x] = Board.object_dict[obj]

    def addApple(self):
        """Adds a new apple to the playing field"""
        random_x, random_y = Board.randomCoordinates(self.size)   
        # try until we find an empty field     
        while self.access(random_x, random_y) != 0:
            random_x, random_y = Board.randomCoordinates(self.size)
        self.setState(random_x, random_y, "apple")

    def getState(self) -> np.ndarray:
        """Returns the current state of the board"""
        return self.board

    def appleEaten(self) -> bool:
        """Checks if there is an Apple on the Board"""
        return np.max(self.board) <= Board.object_dict["snakehead"]

    def checkOverlap(self, positions: list) -> bool:
        """Checks if the snake is overlapping with itself"""
        return not (len(set(positions)) == len(positions))

    def outOfField(self, positions: list) -> bool:
        """Checks if snake is out of field"""
        for part_x, part_y in positions:
            if part_x < 0 or self.size <= part_x:
                return True
            if part_y < 0 or self.size <= part_y:
                return True
        return False

    def applyPositions(self, positions: list, apple_eaten: bool):
        """Applies validated positions to gameboard"""
        # reset snake parts on board
        self.board[self.board == Board.object_dict["snake"]] = Board.object_dict["empty"]
        self.board[self.board == Board.object_dict["snakehead"]] = Board.object_dict["empty"]
        # update snake positions
        for i, p in enumerate(positions):
            x, y = p
            self.setState(x, y, "snake" if i > 0 else "snakehead")
        # replace apple if eaten
        if apple_eaten:
            self.addApple()

    def validateMove(self, positions: list) -> str:
        """Validates snake move and returns corresponding game event"""
        output = "none"
        if self.checkOverlap(positions) or self.outOfField(positions):
            return "death"
        eaten = self.appleEaten()
        if eaten:
            output = "eat"
        self.applyPositions(positions, eaten)
        return output
        