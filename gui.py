import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
import time


class SnakeGUI():
    def __init__(self, env, include_timer=True):
        self.window_width = 680
        self.window_height = 650

        self.board_height = 600
        self.text_height = 50
        self.text_x_gap = 30

        self.square_len = 40

        self.background_color = (74, 117, 44)
        self.board_colors = [(116, 217, 72), (142, 204, 58)]
        self.snake_head_color = (63, 72, 204)
        self.snake_body_color = (0, 162, 232)
        self.apple_color = (237, 28, 35)
        self.scores_color = (255, 255, 255)

        self.right_key = pygame.K_RIGHT
        self.left_key = pygame.K_LEFT
        self.up_key = pygame.K_UP
        self.down_key = pygame.K_DOWN
        self.switch_mode_key = pygame.K_SPACE

        self.timer = None
        self.include_timer = include_timer

        self.update_delay_sec = .120
        self.max_user_actions_per_update = 2

        pygame.init()
        pygame.font.init()
        self.font = pygame.font.SysFont('Ariel', 30)

        self.window = None
        self.env = env

    
    def render(self, mode, user_control):
        if self.include_timer and self.timer is not None:
            time_diff = time.perf_counter() - self.timer
        
            if time_diff < self.update_delay_sec:
                pygame.time.delay(int(self.update_delay_sec*1000 - time_diff*1000))
        
        if self.window is None:
            self.window = pygame.display.set_mode((self.window_width, self.window_height))
        
        self.draw_background(self.env.width, self.env.height)
        self.draw_snake(self.env.snake)
        self.draw_apple(self.env.apple)
        self.draw_scores(self.env.best_score, self.env.curr_score)
        pygame.display.update()


        user_actions = []
        switch_mode = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.display.quit()
                pygame.quit()
                quit(code=0)
        
            elif user_control and event.type == pygame.KEYDOWN and len(user_actions) < self.max_user_actions_per_update:
                if event.key == self.right_key:
                    user_actions.append(self.env.action_right)
                elif event.key == self.left_key:
                    user_actions.append(self.env.action_left)
                elif event.key == self.up_key:
                    user_actions.append(self.env.action_up)
                elif event.key == self.down_key:
                    user_actions.append(self.env.action_down)
                elif event.key == self.switch_mode_key:
                    switch_mode = True
                
            elif event.type == pygame.KEYDOWN:
                if event.key == self.switch_mode_key:
                    switch_mode = True


        if self.include_timer:
            self.timer = time.perf_counter()
        
        if user_control:
            return user_actions, switch_mode
        
        return None, switch_mode


    
    def draw_background(self, board_width, board_height):
        assert self.window is not None, 'self.draw_background (SnakeGUI): window must not be None'
        
        for row in range(board_height):
            for column in range(board_width):
                pygame.draw.rect(self.window, self.board_colors[(row+column)%2],
                [column*self.square_len, row*self.square_len, self.square_len, self.square_len])


    def draw_scores(self, best_score, curr_score):
        assert self.window is not None, 'self.draw_scores (SnakeGUI): window must not be None'

        best_score_text = self.font.render(f'Best Score: {best_score}', False, self.scores_color)
        curr_score_text = self.font.render(f'Current Score: {curr_score}', False, self.scores_color)

        text_size = curr_score_text.get_rect()
        text_y_gap = int(self.text_height / 2 - text_size.height / 2)

        pygame.draw.rect(self.window, self.background_color, [0, self.board_height, self.window_width, self.text_height])
        self.window.blit(best_score_text, (self.text_x_gap, self.board_height + text_y_gap))
        self.window.blit(curr_score_text, (self.window_width - self.text_x_gap - text_size.width, self.board_height + text_y_gap))

 
    def draw_snake(self, snake):
        assert snake is not None, 'self.draw_snake (SnakeGUI): snake must not be None'
        assert self.window is not None, 'self.draw_snake (SnakeGUI): window must not be None'

        color = self.snake_head_color        

        for point in snake:
            x_draw, y_draw = point.column*self.square_len, point.row*self.square_len
            pygame.draw.rect(self.window, color, [int(x_draw), int(y_draw), self.square_len, self.square_len])
            
            color = self.snake_body_color
    

    def draw_apple(self, apple):
        assert apple is not None, 'self.draw_apple (SnakeGUI): apple must not be None'
        assert self.window is not None, 'self.draw_apple (SnakeGUI): window must not be None'

        pygame.draw.rect(self.window, self.apple_color, [apple.column*self.square_len, apple.row*self.square_len, self.square_len,
        self.square_len])


    def reset(self):
        self.timer = None


    def close(self):
        pygame.display.quit()
        pygame.quit()
        self.window = None
        self.timer = None
