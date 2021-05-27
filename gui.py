from datetime import time
import PySimpleGUI as sg
from env import SnakeGame


class GUI(sg.Window):
    def __init__(self):
        self.init_properties()
        self.init_layout()

        super().__init__(title='Snake', layout=[[self.best_score_label, self.curr_score_label], [self.canvas]],
        size=(self.window_width, self.window_height), margins=(0, 0), background_color=self.background_color, return_keyboard_events=True)
        self.finalize()

    
    def init_properties(self):
        self.window_width = 680
        self.window_height = 650

        self.board_width = self.window_width # 40 * 17
        self.board_height = 600 # 40 * 15

        self.board_x = 0
        self.board_y = 50

        self.background_color = '#4a752c'
        self.board_colors = ['#74d948', '#8ecc3a']
        self.snake_colors = ['#3f48cc', '#00a2e8']
        self.apple_color = '#ed1c23'

        self.font = ('Arial', 16)

        self.board_square_length = 40


    def init_layout(self):
        self.best_score_label = sg.Text('Best Score:', size=(26, 1), font=self.font, background_color=self.background_color,
        justification='left')
        
        self.curr_score_label = sg.Text('Current Score:',  size=(26, 1), font=self.font, background_color=self.background_color,
        justification='right')

        self.canvas = sg.Graph(
            canvas_size=(self.board_width, self.board_height),
            graph_bottom_left=(self.board_x, self.board_height),
            graph_top_right=(self.board_width, self.board_y),
            key="graph"
        )

    
    def set_best_score(self, score):
        self.best_score_label.update(value=f'Best Score: {score}')
    

    def set_curr_score(self, score):
        self.curr_score_label.update(value=f'Current Score: {score}')

    
    def draw_background(self, width, height):
        for row in range(height):
            for column in range(width):
                start_x, start_y = column*self.board_square_length, row*self.board_square_length
                
                self.canvas.draw_rectangle((start_x, start_y), (start_x+self.board_square_length, start_y+self.board_square_length),
                fill_color=self.board_colors[(row+column)%2])
    

    def draw_snake(self, snake):
        head, *body = snake
        head_row, head_column = head
        head_color, body_color = self.snake_colors

        head_x, head_y = head_column*self.board_square_length, head_row*self.board_square_length
        self.canvas.draw_rectangle((head_x, head_y), (head_x+self.board_square_length, head_y+self.board_square_length), fill_color=head_color)

        for body_part in body:
            body_row, body_column = body_part
            body_x, body_y = body_column*self.board_square_length, body_row*self.board_square_length

            self.canvas.draw_rectangle((body_x, body_y), (body_x+self.board_square_length, body_y+self.board_square_length), fill_color=body_color)

    
    def draw_apple(self, apple):
        apple_row, apple_column = apple

        apple_x, apple_y = apple_column*self.board_square_length, apple_row*self.board_square_length
        self.canvas.draw_rectangle((apple_x, apple_y), (apple_x+self.board_square_length, apple_y+self.board_square_length),
        fill_color=self.apple_color)