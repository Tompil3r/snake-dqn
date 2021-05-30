from snake_env import SnakeEnv
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame


class GUI():
    WINDOW_WIDTH = 680
    WINDOW_HEIGHT = 650

    SQUARE_LEN = 40

    BACKGROUND_COLOR = (74, 117, 44)
    BOARD_COLORS = [(116, 217, 72), (142, 204, 58)]
    SNAKE_HEAD_COLOR = (63, 72, 204)
    SNAKE_BODY_COLOR = (0, 162, 232)
    APPLE_COLOR = (237, 28, 35)

    GAME_DELAY = 130


    def __init__(self, **kwargs):
        self.init_variables(**kwargs)
        pygame.init()


    def init_variables(self, **kwargs):
        self.left_key = kwargs.get('left_key', pygame.K_LEFT)
        self.right_key = kwargs.get('right_key', pygame.K_RIGHT)
        self.up_key = kwargs.get('up_key', pygame.K_UP)
        self.down_key = kwargs.get('down_key', pygame.K_DOWN)
        self.start_key = kwargs.get('start_key', pygame.K_SPACE)

        self.do_animation = kwargs.get('do_animation', True)
        self.animation_update_rate = kwargs.get('animation_update_rate', 30)

        self.best_score = kwargs.get('best_score', 0)
        self.curr_score = kwargs.get('curr_score', 0)


    def init_window(self):
        self.window = pygame.display.set_mode((GUI.WINDOW_WIDTH, GUI.WINDOW_HEIGHT))
        pygame.display.set_caption('Snake Game')


    def load_game(self, snake_env):
        self.init_window()

        while True:
            self.wait_to_start(snake_env)
            self.start(snake_env)
            snake_env.reset()


    def wait_to_start(self, snake_env):
        game_started = False

        while not game_started:
            self.draw_game(snake_env)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit(code=0)
                
                if event.type == pygame.KEYDOWN:
                    if event.key == self.start_key:
                        game_started = True


    def start(self, snake_env):
        game_running = True

        while game_running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit(code=0)
                
                elif event.type == pygame.KEYDOWN:
                    if self.handle_keys_input(event.key):
                        snake_env.step()

            self.draw_game(snake_env)
            game_status = snake_env.step()
            pygame.time.delay(GUI.GAME_DELAY)

            if game_status != 0:
                game_running = False


    def handle_keys_input(self, key):
        if key == self.left_key:
            snake_env.update_dir(SnakeEnv.LEFT_DIR)
            return True
        elif key == self.up_key:
            snake_env.update_dir(SnakeEnv.UP_DIR)
            return True
        elif key == self.right_key:
            snake_env.update_dir(SnakeEnv.RIGHT_DIR)
            return True
        elif key == self.down_key:
            snake_env.update_dir(SnakeEnv.DOWN_DIR)
            return True
        
        return False


    def draw_background(self, board_width, board_height):
        for row in range(board_height):
            for column in range(board_width):
                pygame.draw.rect(self.window, GUI.BOARD_COLORS[(row+column)%2],
                [column*GUI.SQUARE_LEN, row*GUI.SQUARE_LEN, GUI.SQUARE_LEN, GUI.SQUARE_LEN])

    
    def snake_animation(self):
        delay_per_update = GUI.GAME_DELAY // self.animation_update_rate
        movement_deltas = snake_env.get_snake_movment_delta(movement_rate=1/self.animation_update_rate)


        

    def draw_snake(self, snake, offset=(0, 0)):
        if snake is None:
            return

        color = GUI.SNAKE_HEAD_COLOR
        offset_x, offset_y = offset

        for snake_point in snake:
            row, column = snake_point
            pygame.draw.rect(self.window, color, [column*GUI.SQUARE_LEN + offset_x, row*GUI.SQUARE_LEN + offset_y, GUI.SQUARE_LEN, GUI.SQUARE_LEN])

            color = GUI.SNAKE_BODY_COLOR
    

    def draw_apple(self, apple):
        if apple is None:
            return

        row, column = apple
        pygame.draw.rect(self.window, GUI.APPLE_COLOR, [column*GUI.SQUARE_LEN, row*GUI.SQUARE_LEN, GUI.SQUARE_LEN, GUI.SQUARE_LEN])
    

    def draw_game(self, snake_env):
        self.draw_background(SnakeEnv.BOARD_WIDTH, SnakeEnv.BOARD_HEIGHT)
        self.draw_snake(snake_env.snake)
        self.draw_apple(snake_env.apple)
        pygame.display.update()


snake_env = SnakeEnv()

gui = GUI()
gui.load_game(snake_env)