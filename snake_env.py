import random
import numpy as np


class SnakeEnv():
    def __init__(self):
        self.init_variables()
        self.init_snake()
        self.randomize_apple()
    

    def init_variables(self):
        self.board_width = 17
        self.board_height = 15
        
        self.snake_starting_len = 3
        self.snake_head_start = (self.board_height//2, self.board_width//2)
        self.max_spawning_attempts = 100
        self.snake = []
        self.apple = None
        self.last_tail = None
        self.dir = (0, 1)

        self.empty_value = 0
        self.body_value = 1
        self.head_value = 2
        self.apple_value = 3
    

    def init_snake(self):
        spawn_dir = self.get_opposite_dir(self.dir)

        last_point = self.copy_point(self.snake_head_start)
        self.snake.append(last_point)

        spawning_attempts = 0

        for idx in range(self.snake_starting_len - 1):
            point = self.add_dir(last_point, spawn_dir)

            while (not self.in_bounds(point) or self.is_occupied_by_snake(point)):
                spawning_attempts += 1
                if spawning_attempts == self.max_spawning_attempts:
                    raise Exception('Unable to spawn snake')

                spawn_dir = self.get_random_dir()
                point = self.add_dir(last_point, spawn_dir)
            
            last_point = point
            self.snake.append(point)


    def in_bounds(self, point):
        row, column = point 
        return (0 <= row and row < self.board_height) and (0 <= column and column < self.board_width)


    def add_dir(self, point, dir):
        row, column = point
        delta_row, delta_column = dir
        return (row + delta_row, column + delta_column)
    

    def get_random_direction_value(self):
        return (-1, 1)[random.randint(0, 1)]


    def get_random_dir(self, axis=None):
        if axis is None:
            dir = [0, 0]
            dir[random.randint(0, 1)] = self.get_random_direction_value()
            return tuple(dir)
        elif axis == 0:
            return (self.get_random_direction_value(), 0)
        elif axis == 1:
            return (0, self.get_random_direction_value())
    

    def get_opposite_dir(self, dir):
        return tuple([-value for value in dir])


    def get_axis(self, dir):
        if abs(dir[0]) == 1:
            return 0
        elif abs(dir[1]) == 1:
            return 1
        return None
    

    def get_perpendicular_axis(self, axis):
        if axis == 0:
            return 1
        elif axis == 1:
            return 0
        return None
    

    def is_occupied_by_snake(self, point, include_head=True):
        head, *body = self.snake

        if include_head:
            if point == head:
                return True
        
        for body_point in body:
            if point == body_point:
                return True
        
        return False


    def copy_point(self, point):
        return tuple([value for value in point])

    
    def get_game_matrix(self):
        game_matrix = np.full(shape=(self.board_height, self.board_width), fill_value=self.empty_value)

        head, *body = self.snake
        game_matrix[head] = self.head_value

        for body_point in body:
            game_matrix[body_point] = self.body_value
        
        if self.apple is not None:
            game_matrix[self.apple] = self.apple_value
        
        return game_matrix
    

    def is_board_full(self):
        game_matrix = self.get_game_matrix()
        return np.count_nonzero((game_matrix==self.empty_value) | (game_matrix==self.apple_value)) == 0
    

    def randomize_apple(self):
        while (self.apple is None or self.is_occupied_by_snake(self.apple)):
            self.apple = (random.randint(0, self.board_height - 1), random.randint(0, self.board_width - 1))


    def move_snake(self):
        head, *body = self.snake
        last_point = head

        snake = [self.add_dir(head, self.dir)]

        for idx in range(len(body)):
            snake.append(last_point)
            last_point = body[idx]
        
        self.snake = snake
        self.last_tail = last_point
    

    def is_eating_apple(self):
        return (self.snake[0] == self.apple)


    def is_game_over(self):
        return (not self.in_bounds(self.snake[0]) or self.is_occupied_by_snake(self.snake[0], include_head=False))


    def spawn_tail(self):
        self.snake.append(self.last_tail)
        self.last_tail = None
    

    def step(self):
        self.move_snake()

        if self.is_eating_apple():
            self.apple = None
            self.spawn_tail()
            self.randomize_apple()
        
        if self.is_game_over():
            return -1
        elif self.is_board_full():
            return 1
        return 0


    def update_dir(self, dir):
        if self.dir != self.get_opposite_dir(dir):
            self.dir = dir


    def reset(self):
        self.__init__()