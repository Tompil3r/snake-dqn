from snake_env import SnakeEnv
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
import time


class GUI():
    WINDOW_WIDTH = 680
    WINDOW_HEIGHT = 650

    BOARD_HEIGHT = 600
    TEXT_HEIGHT = 50
    TEXT_X_GAP = 30

    SQUARE_LEN = 40

    BACKGROUND_COLOR = (74, 117, 44)
    BOARD_COLORS = [(116, 217, 72), (142, 204, 58)]
    SNAKE_HEAD_COLOR = (63, 72, 204)
    SNAKE_BODY_COLOR = (0, 162, 232)
    APPLE_COLOR = (237, 28, 35)


    def __init__(self, **kwargs):
        pygame.init()
        pygame.font.init()
        self.init_variables(**kwargs)


    def init_variables(self, **kwargs):
        self.left_key = kwargs.get('left_key', pygame.K_LEFT)
        self.right_key = kwargs.get('right_key', pygame.K_RIGHT)
        self.up_key = kwargs.get('up_key', pygame.K_UP)
        self.down_key = kwargs.get('down_key', pygame.K_DOWN)
        self.start_key = kwargs.get('start_key', pygame.K_SPACE)

        self.do_animation = kwargs.get('do_animation', False)
        self.animation_update_rate = kwargs.get('animation_update_rate', 30)
        self.game_delay = 120

        if self.do_animation:
            self.game_delay -= 30

        self.best_score = kwargs.get('best_score', 0)
        self.curr_score = kwargs.get('curr_score', 0)

        self.font = pygame.font.SysFont('Ariel', 30)


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

        self.update_scores()
        
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
                        if self.do_animation:
                            self.snake_animation(snake_env)

                        game_status = snake_env.step()
                        game_running = self.update_game(game_status)

            if self.do_animation:
                self.snake_animation(snake_env)

            else:
                pygame.time.delay(self.game_delay)

            game_status = snake_env.step()
            game_running = self.update_game(game_status)
            self.draw_game(snake_env)


    def update_game(self, game_status):
        if game_status == SnakeEnv.GAME_OVER_CODE:
            if self.curr_score > self.best_score:
                self.best_score = self.curr_score
            self.curr_score = 0
            return False
        elif game_status == SnakeEnv.WON_GAME_CODE:
            self.curr_score += 1                
            if self.curr_score > self.best_score:
                self.best_score = self.curr_score
            
            self.update_scores()
            return False
        elif game_status == SnakeEnv.APPLE_EATEN_CODE:
            self.curr_score += 1
            self.update_scores()
            return True
        elif game_status == SnakeEnv.NOTHING_CHANGED_CODE:
            return True



    def handle_keys_input(self, key):
        if key == self.left_key:
            return snake_env.update_dir(SnakeEnv.LEFT_DIR)
        elif key == self.up_key:
            return snake_env.update_dir(SnakeEnv.UP_DIR)
        elif key == self.right_key:
            return snake_env.update_dir(SnakeEnv.RIGHT_DIR)
        elif key == self.down_key:
            return snake_env.update_dir(SnakeEnv.DOWN_DIR)
        
        return False


    def draw_background(self, board_width, board_height):
        for row in range(board_height):
            for column in range(board_width):
                pygame.draw.rect(self.window, GUI.BOARD_COLORS[(row+column)%2],
                [column*GUI.SQUARE_LEN, row*GUI.SQUARE_LEN, GUI.SQUARE_LEN, GUI.SQUARE_LEN])


    def update_scores(self):
        white_color = (255, 255, 255)

        best_score_text = self.font.render(f'Best Score: {self.best_score}', False, white_color)
        curr_score_text = self.font.render(f'Current Score: {self.curr_score}', False, white_color)

        text_size = curr_score_text.get_rect()
        text_y_gap = int(GUI.TEXT_HEIGHT / 2 - text_size.height / 2)

        pygame.draw.rect(self.window, GUI.BACKGROUND_COLOR, [0, GUI.BOARD_HEIGHT, GUI.WINDOW_WIDTH, GUI.TEXT_HEIGHT])
        self.window.blit(best_score_text, (GUI.TEXT_X_GAP, GUI.BOARD_HEIGHT + text_y_gap))
        self.window.blit(curr_score_text, (GUI.WINDOW_WIDTH - GUI.TEXT_X_GAP - text_size.width, GUI.BOARD_HEIGHT + text_y_gap))

        pygame.display.update()

    
    def snake_animation(self, snake_env):
        delay_per_update = self.game_delay // self.animation_update_rate
        movement_delta = 1 / self.animation_update_rate

        env_copy = snake_env.copy()
        snake_dirs = env_copy.get_snake_dirs(ratio=movement_delta)

        for idx in range(self.animation_update_rate):
            env_copy.move_snake_with_dirs(snake_dirs)
            self.draw_game(env_copy)
            pygame.time.delay(delay_per_update)

        

    def draw_snake(self, snake):
        if snake is None:
            return

        color = GUI.SNAKE_HEAD_COLOR

        for snake_point in snake:
            row, column = snake_point
            pygame.draw.rect(self.window, color, [int(column*GUI.SQUARE_LEN), int(row*GUI.SQUARE_LEN), GUI.SQUARE_LEN, GUI.SQUARE_LEN])

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

gui = GUI(do_animation=True, animation_update_rate=30)
gui.load_game(snake_env)