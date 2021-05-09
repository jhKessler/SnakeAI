from src.game import Game
import pygame

visible = True
gamesize = 20
tile_size = 50
control_type = "PLAYER" # "PLAYER"

color_dict = {
    0: (0, 0, 0), # black for empty field
    1: (52, 235, 52), # green for snakepart
    2: (52, 235, 52), # green for snakehead
    3: (235, 52, 52) # red for apple
}

if __name__ == "__main__":
    game = Game(size=gamesize)

    if visible:
        pygame.init()
        screen = pygame.display.set_mode((gamesize*tile_size, gamesize*tile_size))

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                # key is pressed
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        game.inputMove("up")
                    elif event.key == pygame.K_DOWN:
                        game.inputMove("down")
                    elif event.key == pygame.K_LEFT:
                        game.inputMove("left")
                    elif event.key == pygame.K_RIGHT:
                        game.inputMove("right")
                board = game.getGamestate()
                print(board)
                for y_coord, row in enumerate(board):
                    for x_coord, element in enumerate(row):
                        color = color_dict[element]
                        rect_x, rect_y = x_coord * tile_size, y_coord * tile_size
                        pygame.draw.rect(screen, color, [rect_x, rect_y, tile_size, tile_size])

                pygame.display.update()


