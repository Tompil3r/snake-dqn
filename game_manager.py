from env import SnakeGame
from gui import GUI


class GameManager():
    def __init__(self):
        self.gui = GUI()
        self.snake_game = SnakeGame()

    
    def initialize(self):
        while True:
            event, values = self.gui.read()
            if event is None:
                break
            elif event == ' ':
                self.start()


    def start(self):
        while True:
            self.snake_game.randomize_apple()

            self.gui.draw_background(self.snake_game.width, self.snake_game.height)
            self.gui.draw_snake(self.snake_game.snake)
            self.gui.draw_apple(self.snake_game.apple)

            event, values = self.gui.read()                
            if event is None:
                break
            elif event == 'a':
                pass
            elif event == 'w':
                pass
            elif event == 'd':
                pass


game_manager = GameManager()
game_manager.initialize()
