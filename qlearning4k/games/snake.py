__author__ = "Fariz Rahman"

import numpy as np
from .game import Game


actions = {0:'left', 1:'right', 2:'up', 3:'down', 4:'idle'}
forbidden_moves = [(0, 1), (1, 0), (2, 3), (3, 2)]

class Snake(Game):

    def __init__(self, grid_size=10, snake_length=3):
        self.grid_size = grid_size
        self.snake_length = snake_length
        self.reset()
        self.state_changed = True

    @property
    def name(self):
        return "Snake"
    @property
    def nb_actions(self):
        return 5

    def play(self, action):
        assert action in range(5), "Invalid action."
        self.scored = False
        self.move_snake(action)
        if self.fruit == self.snake[0]:
            self.scored = True
            self.grow()
            self.drop_fruit()
        elif self.self_bite() or self.hit_border():
            self.game_over = True

    def grow(self):
        end = self.snake[-1]
        seg = self.snake[-2]  # segment just before end
        if end[0] == seg[0] - 1:
            # grow to left
            p = (end[0] - 1, end[1])
        elif end[0] == seg[0] + 1:
            # grow to rght
            p = (end[0] + 1, end[1])
        elif end[1] == seg[1] - 1:
            # grow up
            p = (end[0], end[1] - 1)
        else:
            p = (end[0], end[1] + 1)
        self.snake.append(p)

    def drop_fruit(self):
        if len(self.snake) >= (self.grid_size - 2) ** 2:
            self.fruit = (-1, -1)
            pass
        while True:
            fruit = np.random.randint(1, self.grid_size - 1, 2)
            fruit = (fruit[0], fruit[1])
            if fruit in self.snake:
                continue
            else:
                self.fruit = fruit
                break

    def move_snake(self, action):
        if action == 4 or (action, self.previous_action) in forbidden_moves:
            action = self.previous_action
        else:
            self.previous_action = action
        head = self.snake[0]
        if action == 0:
            p = (head[0] - 1, head[1])
        elif action == 1:
            p = (head[0] + 1, head[1])
        elif action == 2:
            p = (head[0], head[1] - 1)
        elif action == 3:
            p = (head[0], head[1] + 1)
        self.snake.insert(0, p)
        self.snake.pop()

    def get_state(self):
        canvas = np.ones((self.grid_size, ) * 2)
        canvas[1:-1, 1:-1] = 0.
        for seg in self.snake:
            canvas[seg[0], seg[1]] = 1.
        canvas[self.fruit[0], self.fruit[1]] = .5
        return canvas

    def get_score(self):
        if self.game_over:
            score = -1
        elif self.scored:
            score = len(self.snake)
        else:
            score = 0
        return score

    def reset(self):
        grid_size = self.grid_size
        snake_length = self.snake_length
        head_x = (grid_size - snake_length) // 2
        self.snake = [(x, grid_size // 2) for x in range (head_x, head_x + snake_length)]
        self.game_over = False
        self.scored = False
        self.drop_fruit()
        if np.random.randint(2) == 0:
            self.previous_action = 0
        else:
            self.previous_action = 1
            self.snake.reverse()
        self.border = []
        for z in range(grid_size):
            self.border += [(z, 0), (z, grid_size - 1), (0, z), (grid_size - 1, z)]

    def left(self):
        self.play(0)

    def right(self):
        self.play(1)

    def up(self):
        self.play(2)

    def down(self):
        self.play(3)

    def idle(self):
        self.play(4)

    def self_bite(self):
        return len(self.snake) > len(set(self.snake))

    def hit_border(self):
        return self.snake[0] in self.border or self.snake[-1] in self.border

    def is_over(self):
        return self.self_bite() or self.hit_border()

    def is_won(self):
        return len(self.snake) > self.snake_length
