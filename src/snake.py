from src.snakepart import SnakePart

class Snake(object):
    """Object of class Snake"""

    def __len__(self):
        return len(self.parts)

    def __init__(self, x: int, y: int):
        """Creates a Snake Object"""
        start_direction = "right"
        self.parts = [SnakePart(x, y, start_direction)]


    def changeDirection(self, new_move: str):
        """updates the directions of all paths"""
        previous_part_move = new_move
        for part in self.parts:
            # set direction of part to that of its previous part
            # setDirection also returns the parts current direction
            # which will be the next direction of the following part
            previous_part_move = part.setDirection(previous_part_move)


    def move(self):
        """Moves the snakeparts 1 step into their current direction"""
        for part in self.parts:
            part.move()


    def step(self, new_move: str):
        """Makes the Snake move"""
        self.changeDirection(new_move)
        self.move()
        

    def getPartPositions(self) -> list:
        """Returns positions of every Snakepart"""
        return [part.getCoordinates() for part in self.parts]


    def addPart(self):
        """Adds a new snakepart to the end of the snake"""
        lastPart = self.parts[-1]
        last_x, last_y = lastPart.getCoordinates()
        last_direction = lastPart.direction
        new_x, new_y = SnakePart.calculateMove(last_x, last_y, last_direction, backwards=True)
        self.parts.append(SnakePart(new_x, new_y, last_direction))

