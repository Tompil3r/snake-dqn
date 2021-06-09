from observation_space import ObservationSpace
from action_space import ActionSpace
from collections import namedtuple, deque
import numpy as np


Point = namedtuple('Point', ['row', 'column'])



class SnakeEnv():

    def __init__(self):
        # board attributes
        self.height = 15
        self.width = 17
        
        # snake attributes
        self.snake = None
        self.apple = None
        self.last_tail = None
        self.dir = None
        self.snake_starting_len = 3
        self.head_starting_point = Point(self.height//2, self.board//2)
        
        # functional attributes
        self.max_spawning_attempts = 100
        self.head_index = 0

        # movement attributes
        self.right_dir = 0
        self.left_dir = 1
        self.up_dir = 2
        self.down_dir = 3

        self.action_map = {self.right_dir:Point(0, 1), self.left_dir:Point(0, -1), self.up_dir:Point(-1, 0), self.down_dir:Point(1, 0)}

        # state attributes
        self.state = None
        self.empty_value = 0
        self.body_value = 1
        self.head_value = 2
        self.apple_value = 3

        # score attributes
        self.best_score = 0
        self.curr_score = 0

        # other properties
        self.action_space = ActionSpace(4)
        self.observation_space = ObservationSpace(shape=(self.height, self.width), dtype=int)

        # game codes
        self.normal_move_code = 0
        self.eating_apple_code = 1
        self.winning_game_code = 2
        self.losing_game_code = 3

        # rewards
        # normal_move_code is implemented with the function: reward = 1/(r+1) - where r is the distance
        # between the head of the snake to the apple
        self.reward_map = {self.normal_move_code:None, self.eating_apple_code:3, self.winning_game_code:1000, self.losing_game_code:-1000}
   

    def get_reward(self, game_code):
        if game_code == self.normal_move_code:
            assert self.snake is not None, 'self.get_reward (SnakeEnv): snake must not be none'
            assert self.apple is not None, 'self.get_reward (SnakeEnv): apple must not be none'

            diff = self.sub_points(self.snake[self.head], self.apple)
            distance = np.sqrt(np.square(diff.row) + np.square(diff.column))
            return 1/(distance + 1)
        try:
            return self.reward_map[game_code]
        except KeyError:
            raise Exception('self.get_reward (SnakeEnv): invalid game code')

    
    def get_dir(self, action):
        return self.action_map[action]


    def negate_point(self, point):
        return Point(-point.row, -point.column)
    

    def add_points(self, point_a, point_b):
        return Point(point_a.row + point_b.row, point_a.column + point_b.column)
    

    def sub_points(self, point_a, point_b):
        return Point(point_a.row - point_b.row, point_a.column - point_b.column)


    def copy_point(self, point):
        return Point(point.row, point.column)


    def in_bounds(self, point):
        return (point.row >= 0 and point.row < self.height) and (point.column >= 0 and point.column < self.width)
    

    def is_snake_occupying(self, point, include_head=True):
        assert self.snake is not None, 'self.is_snake_occupying (SnakeEnv): snake must not be None'

        for idx,_point in enumerate(self.snake):
            if include_head and idx == self.head_index:
                if point == _point:
                    return True
            
            else:
                if point == _point:
                    return True
        
        return False


    def randomize_apple(self):
        assert self.apple is None, 'self.randomize_apple (SnakeEnv): apple is not None'

        while self.apple is None or self.is_snake_occupying(self.apple):
            self.apple = Point(np.random.randint(0, self.height), np.random.randint(0, self.width))

    
    def get_state(self):
        state = np.full(shape=self.observation_space.shape, fill_value=self.empty_value, dtype=self.observation_space.dtype)

        if self.snake is not None:
            for idx,point in enumerate(self.snake):
                if idx == self.head_index:
                    state[point.row, point.column] = self.head_value
                else:
                    state[point.row, point.column] = self.body_valu
        
        if self.apple is not None:
            state[self.apple.row, self.apple.column] = self.apple_value
        
        return state


    def init_dir(self):
        self.dir = self.get_dir(self.right_dir)


    def init_snake(self):
        self.snake = deque()

        assert self.dir is not None, 'self.init_snake (SnakeEnv): dir must not be None'

        spawn_dir = self.negate_point(self.dir)
        last_point = self.copy_point(self.head_starting_point)
        self.snake.append(last_point)

        spawning_attemps = 0

        for idx in range(self.snake_starting_len - 1):
            point = self.add_points(last_point, spawn_dir)

            while not self.in_bounds(point) or self.is_snake_occupying(point):
                spawning_attemps += 1
                
                if spawning_attemps >= self.max_spawning_attempts:
                    raise Exception('self.init_snake (SnakeEnv): unable to spawn snake')
                
                spawn_dir = self.action_map[self.action_space.sample()]
                point = self.add_points(last_point, spawn_dir)

            self.snake.append(point)
            last_point = point
        

    def is_board_full(self):
        assert self.state is not None, 'self.is_board_full (SnakeEnv): state must no be None'
        return np.count_nonzero((self.state==self.empty_value) | (self.state==self.apple_value)) == 0
    

    def update_dir(self, action):
        try:
            if action is None:
                return

            dir = self.action_map[action]

            if dir != self.dir and dir != self.negate_point(self.dir):
                self.dir = self.copy_point(dir)

        except KeyError:
            raise Exception('self.update_dir (SnakeEnv): invalid action')


    def get_next_snake(self):
        snake = deque()

        last_point = self.snake[self.head_index]
        snake.append(self.add_points(last_point, self.dir))

        for idx in range(len(self.snake)):
            if idx == self.head_index:
                continue
            
            snake.append(last_point)
            last_point = self.snake[idx]
        
        return snake, last_point
    

    def is_eating_apple(self):
        return self.snake[self.head_index] == self.apple
    

    def is_game_over(self):
        return not self.in_bounds(self.snake[self.head_index]) or self.is_snake_occupying(self.snake[self.head], include_head=False)
    

    def spawn_tail(self):
        assert self.last_tail is not None, 'self.spawn_tail (SnakeEnv): last tail is None - invalid'
        self.snake.append(self.last_tail)
        self.last_tail = None
    

    def step(self, action):
        reward = 0
        done = False
        info = {}
        
        event_idx = 0

        self.update_dir(action)
        self.snake, self.last_tail = self.get_next_snake()

        if self.is_eating_apple():
            self.curr_score += 1
            info[f'Event{event_idx}':'Apple Eaten']
            event_idx += 1
            
            reward += self.get_reward(self.eating_apple_code)
                       
            self.apple = None
            self.spawn_tail()

            self.state = self.get_state()

            if self.is_board_full():
                done = True
                info[f'Event{event_idx}':'Game Won']
                event_idx += 1
                reward += self.get_reward(self.winning_game_code)
            else:
                self.randomize_apple()

        else:
            if self.is_game_over():
                done = True
                info[f'Event{event_idx}':'Game Lost']
                event_idx += 1
                reward += self.get_reward(self.losing_game_code)
            
            else:
                reward += self.get_reward(self.normal_move_code)
        
        return np.copy(self.state), reward, done, info




            

            
            
    

