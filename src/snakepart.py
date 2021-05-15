class SnakePart(object):
    """Class that represents a single part of the snake"""


    @staticmethod
    def calculateMove(x: int, y: int, direction: str, backwards=False) -> tuple:
        """Calculates next position corresponding to x, y and current velocity"""
        value = 1 if not backwards else -1
        if direction == "left":
            x -= value
        elif direction == "right":
            x += value
        elif direction == "up":
            y -= value
        elif direction == "down":
            y += value
        return x, y


    def __init__(self, x: int, y: int, direction: str):
        """Creates an Instance of the SnakePart class"""
        self.x, self.y = x, y
        self.direction = direction


    def setDirection(self, new_direction: str):
        """Sets the direction of a snakepart, returns old direction for convenience"""
        previous = self.direction
        self.direction = new_direction
        return previous


    def move(self):
        """Moves Snakepart"""
        self.x, self.y = SnakePart.calculateMove(self.x, self.y, self.direction)
    

    def getCoordinates(self) -> tuple:
        return self.x, self.y
        