import random
import numpy as np


class SnakeGame():
    def __init__(self):
        self.width = 17
        self.height = 15
        self.snake_starting_length = 3
        self.snake = []
        self.apple = None
        self.dir = (0, 1)
        
        self.empty_value = 0
        self.body_value = 1
        self.head_value = 2
        self.apple_value = 3

        self.init_snake()
    

    def init_snake(self):
        row = self.height // 2
        column  = self.snake_starting_length - 1

        for idx in range(self.snake_starting_length):
            self.snake.append([row, column])
            column -= 1
    

    def is_snake_occupying(self, point, include_head=True):
        head, *body = self.snake

        if include_head:
            if head == point:
                return True
    
        for body_part in body:
            if body_part == point:
                return True
        
        return False
    

    def is_board_full(self):
        board_matrix = self.get_board_matrix()        
        
        return board_matrix.count_nonzero((board_matrix==self.empty_value) | (board_matrix==self.apple_value)) == 0
    

    def randomize_apple(self):
        point_occupied = True

        while point_occupied:
            point = [random.randint(0, self.height - 1), random.randint(0, self.width - 1)]
            
            if not self.is_snake_occupying(point):
                self.apple = point
                return


    def get_board_matrix(self):
        # empty = 0, body part = 1, head = 2, apple = 3
        board_matrix = np.full(shape=(self.height, self.width), fill_value=self.empty_value)

        head, *body = self.snake
        board_matrix[head] = self.head_value

        for body_part in body:
            board_matrix[body_part] = self.body_value
        
        if self.apple is not None:
            board_matrix[self.apple] = self.apple_value
        
        return board_matrix
    

    def in_bounds(self, point):
        row, column = point

        return (0 <= row and row <= self.height - 1) and (0 <= column and column <= self.width - 1)


    def is_game_over(self):
        head, *body = self.snake
        return (not self.in_bounds(head) or self.is_snake_occupying(head, include_head=False))
    

    def move(self):
        move_to = self.snake[0].copy()
        self.snake[0][0] += self.dir[0]
        self.snake[0][1] += self.dir[1]

        for idx in range(1, len(self.snake)):
            self.snake[idx], move_to = move_to, self.snake[idx]
        
        self.snake.append(move_to)


