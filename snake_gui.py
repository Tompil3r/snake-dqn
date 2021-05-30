from snake_env import SnakeEnv
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame


class GUI():
    def __init__(self):
        self.init_properties()
        pygame.init()


    def init_properties(self):
        self.window_width = 680
        self.window_height = 650

        self.square_len = 40

        self.background_color = (74, 117, 44)
        self.board_colors = [(116, 217, 72), (142, 204, 58)]
        self.snake_head_color = (63, 72, 204)
        self.snake_body_color = (0, 162, 232)
        self.apple_color = (237, 28, 35)

        self.game_delay = 130
        self.snake_animation_rate = 30

        self.animation = True

        self.best_score = 0
        self.curr_score = 0


    def init_window(self):
        self.window = pygame.display.set_mode((self.window_width, self.window_height))
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
                    if event.key == pygame.K_SPACE:
                        game_started = True


    def start(self, snake_env):
        game_running = True

        while game_running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit(code=0)
                
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:
                        snake_env.update_dir((0, -1))
                        if self.animation:
                            self.snake_animation(snake_env)
                        else:
                            snake_env.step()
                    elif event.key == pygame.K_UP:
                        snake_env.update_dir((-1, 0))
                        if self.animation:
                            self.snake_animation(snake_env)
                        else:
                            snake_env.step()
                    elif event.key == pygame.K_RIGHT:
                        snake_env.update_dir((0, 1))
                        if self.animation:
                            self.snake_animation(snake_env)
                        else:
                            snake_env.step()
                    elif event.key == pygame.K_DOWN:
                        snake_env.update_dir((1, 0))
                        if self.animation:
                            self.snake_animation(snake_env)
                        else:
                            snake_env.step()
            
            if self.animation:
                self.snake_animation(snake_env)
            else:
                game_status = snake_env.step()
                pygame.time.delay(self.game_delay)
                self.draw_game(snake_env)
            
                if game_status != 0:
                    game_running = False


    def draw_background(self, board_width, board_height):
        for row in range(board_height):
            for column in range(board_width):
                pygame.draw.rect(self.window, self.board_colors[(row+column)%2],
                [column*self.square_len, row*self.square_len, self.square_len, self.square_len])

    
    def snake_animation(self, snake_env):
        delay_per_update = self.game_delay / self.snake_animation_rate
        moving_rate = 1 / self.snake_animation_rate

        for idx in range(self.snake_animation_rate):
            snake_env.move_snake(moving_rate)
            self.draw_game(snake_env)
            pygame.time.delay(int(delay_per_update))

        

    def draw_snake(self, snake, offset=(0, 0)):
        if snake is None:
            return

        color = self.snake_head_color
        offset_x, offset_y = offset

        for snake_point in snake:
            row, column = snake_point
            pygame.draw.rect(self.window, color, [column*self.square_len + offset_x, row*self.square_len + offset_y, self.square_len, self.square_len])

            color = self.snake_body_color
    

    def draw_apple(self, apple):
        if apple is None:
            return

        row, column = apple
        pygame.draw.rect(self.window, self.apple_color, [column*self.square_len, row*self.square_len, self.square_len, self.square_len])
    

    def draw_game(self, snake_env):
        self.draw_background(snake_env.board_width, snake_env.board_height)
        self.draw_snake(snake_env.snake)
        self.draw_apple(snake_env.apple)
        pygame.display.update()


snake_env = SnakeEnv()

gui = GUI()
gui.load_game(snake_env)