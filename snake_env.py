import random
from typing import MutableMapping
import numpy as np


class SnakeEnv():
    BOARD_WIDTH = 17
    BOARD_HEIGHT = 15

    SNAKE_STARTING_LEN = 3
    SNAKE_HEAD_START = (BOARD_HEIGHT//2, BOARD_WIDTH//2)
    MAX_SPAWNING_ATTEMPTS = 100

    HEAD_INDEX = 0

    UP_DIR = (-1, 0)
    DOWN_DIR = (1, 0)
    RIGHT_DIR = (0, 1)
    LEFT_DIR = (0, -1)

    EMPTY_VALUE = 0
    BODY_VALUE = 1
    HEAD_VALUE = 2
    APPLE_VALUE = 3

    GAME_OVER_CODE = 0
    NOTHING_CHANGED_CODE = 1
    APPLE_EATEN_CODE = 2
    WON_GAME_CODE = 3


    def __init__(self, init=True):
        if init:
            self.init_variables()
            self.init_snake()
            self.randomize_apple()
        

    def init_variables(self):
        self.snake = []
        self.apple = None
        self.last_tail = None
        self.dir = (0, 1)


    def copy(self):
        snake_env = SnakeEnv(init=False)

        snake_env.snake = [self.copy_point(snake_point) for snake_point in self.snake]
        snake_env.apple = self.copy_point(self.apple)
        snake_env.last_tail = self.copy_point(self.last_tail)
        snake_env.dir = self.copy_point(self.dir)

        return snake_env


    def init_snake(self):
        spawn_dir = self.get_opposite_dir(self.dir)

        last_point = self.copy_point(SnakeEnv.SNAKE_HEAD_START)
        self.snake.append(last_point)

        spawning_attempts = 0

        for idx in range(SnakeEnv.SNAKE_STARTING_LEN - 1):
            point = self.add_point(last_point, spawn_dir)

            while (not self.in_bounds(point) or self.is_occupied_by_snake(point)):
                spawning_attempts += 1
                if spawning_attempts == SnakeEnv.MAX_SPAWNING_ATTEMPTS:
                    raise Exception('Unable to spawn snake')

                spawn_dir = self.get_random_dir()
                point = self.add_point(last_point, spawn_dir)
            
            last_point = point
            self.snake.append(point)


    def in_bounds(self, point):
        row, column = point 
        return (0 <= row and row < SnakeEnv.BOARD_HEIGHT) and (0 <= column and column < SnakeEnv.BOARD_WIDTH)


    def add_point(self, point, dir):
        row, column = point
        delta_row, delta_column = dir
        return (row + delta_row, column + delta_column)
    

    def sub_point(self, point, dir):
        row, column = point
        delta_row, delta_column = dir
        return (row - delta_row, column - delta_column)
    

    def mul_point_by_num(self, point, multiplier):
        row, column = point
        return (row*multiplier, column*multiplier)


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
        if point is None:
            return None
        return tuple(list(point))

    
    def get_game_matrix(self):
        game_matrix = np.full(shape=(SnakeEnv.BOARD_HEIGHT, SnakeEnv.BOARD_WIDTH), fill_value=SnakeEnv.EMPTY_VALUE)

        head, *body = self.snake
        game_matrix[head] = SnakeEnv.HEAD_VALUE

        for body_point in body:
            game_matrix[body_point] = SnakeEnv.BODY_VALUE
        
        if self.apple is not None:
            game_matrix[self.apple] = SnakeEnv.APPLE_VALUE
        
        return game_matrix
    

    def is_board_full(self):
        game_matrix = self.get_game_matrix()
        return np.count_nonzero((game_matrix==SnakeEnv.EMPTY_VALUE) | (game_matrix==SnakeEnv.APPLE_VALUE)) == 0
    

    def randomize_apple(self):
        while (self.apple is None or self.is_occupied_by_snake(self.apple)):
            self.apple = (random.randint(0, SnakeEnv.BOARD_HEIGHT - 1), random.randint(0, SnakeEnv.BOARD_WIDTH - 1))


    def move_snake(self):
        self.snake, self.last_tail = self.get_next_snake()
    

    def move_snake_with_dirs(self, dirs):
        if len(dirs) != len(self.snake):
            return
        
        for idx in range(len(self.snake)):
            self.snake[idx] = self.add_point(self.snake[idx], dirs[idx])


    
    def get_next_snake(self):
        last_point = self.snake[SnakeEnv.HEAD_INDEX]

        snake = [self.add_point(self.snake[SnakeEnv.HEAD_INDEX], self.dir)]

        for idx in range(len(self.snake)):
            if idx == SnakeEnv.HEAD_INDEX:
                continue

            snake.append(last_point)
            last_point = self.snake[idx]
        
        return (snake, last_point)
    

    def get_snake_dirs(self, ratio=1):
        next_snake, last_tail = self.get_next_snake()
        dirs = []

        for next_point, curr_point in zip(next_snake, self.snake):
            dirs.append(self.mul_point_by_num(self.sub_point(next_point, curr_point), ratio))
        
        return dirs


    def is_eating_apple(self):
        return (self.snake[SnakeEnv.HEAD_INDEX] == self.apple)


    def is_game_over(self):
        return (not self.in_bounds(self.snake[SnakeEnv.HEAD_INDEX]) or self.is_occupied_by_snake(self.snake[SnakeEnv.HEAD_INDEX], include_head=False))


    def spawn_tail(self):
        self.snake.append(self.last_tail)
        self.last_tail = None
    

    def step(self):
        self.move_snake()

        if self.is_eating_apple():
            self.apple = None
            self.spawn_tail()

            if self.is_board_full():
                return SnakeEnv.WON_GAME_CODE

            self.randomize_apple()
            return SnakeEnv.APPLE_EATEN_CODE
        
        if self.is_game_over():
            return SnakeEnv.GAME_OVER_CODE
        
        return SnakeEnv.NOTHING_CHANGED_CODE


    def update_dir(self, dir):
        if self.dir == dir or self.dir == self.get_opposite_dir(dir):
            return False

        self.dir = dir
        return True

    def reset(self):
        self.__init__()