import random
from collections import deque
from typing import Sequence

RENDER_ENABLED = False# Toggle this to enable/disable visualization

if RENDER_ENABLED:
    import pygame

class Vector:
    def __init__(self, x: int = 0, y: int = 0):
        self.x = x
        self.y = y

    def __str__(self):
        return f'Vector({self.x}, {self.y})'

    def __add__(self, other: 'Vector') -> 'Vector':
        return Vector(self.x + other.x, self.y + other.y)

    def within(self, scope: 'Vector') -> bool:
        return self.x < scope.x and self.x >= 0 and self.y < scope.y and self.y >= 0

    def __eq__(self, other: 'Vector') -> bool:
        return self.x == other.x and self.y == other.y

    @classmethod
    def random_within(cls, scope: 'Vector') -> 'Vector':
        return Vector(random.randint(0, scope.x - 1), random.randint(0, scope.y - 1))

class Snake:
    def __init__(self, *, game: 'SnakeGame'):
        self.game = game
        self.score = 0
        self.v = Vector(0, 0)
        self.body = deque()
        self.body.append(Vector.random_within(self.game.grid))

    def move(self):
        self.p = self.p + self.v

    @property
    def cross_own_tail(self):
        try:
            self.body.index(self.p, 1)
            return True
        except ValueError:
            return False

    @property
    def p(self):
        return self.body[0]

    @p.setter
    def p(self, value):
        self.body.appendleft(value)
        self.body.pop()

    def add_score(self):
        self.score += 1
        tail = self.body.pop()
        self.body.append(tail)
        self.body.append(tail)

class Food:
    def __init__(self, game: 'SnakeGame'):
        self.game = game
        self.p = Vector.random_within(self.game.grid)

class SnakeGame:
    def __init__(self, xsize: int = 30, ysize: int = 30, scale: int = 15):
        self.grid = Vector(xsize, ysize)
        self.scale = scale
        if RENDER_ENABLED:
            pygame.init()
            self.screen = pygame.display.set_mode((xsize * scale, ysize * scale))
            self.clock = pygame.time.Clock()
            self.color_snake_head = (0, 255, 0)
            self.color_food = (255, 0, 0)

    def __del__(self):
        if RENDER_ENABLED:
            pygame.quit()

    def block(self, obj):
        return (obj.x * self.scale, obj.y * self.scale, self.scale, self.scale)

    def update_display(self, snake: Snake, food: Food):
        if RENDER_ENABLED:
            self.screen.fill((0, 0, 0))
            pygame.draw.rect(self.screen, self.color_food, self.block(food.p))
            for segment in snake.body:
                pygame.draw.rect(self.screen, self.color_snake_head, self.block(segment))
            pygame.display.update()
            self.clock.tick(10)

def get_input_state(snake: Snake, food: Food, grid: Vector) -> Sequence[int]:
    """Creates input state for AI agent based on current game state"""
    head = snake.p
    food_pos = food.p

    FoodTable = [0, 0, 0, 0]
    ObstacleTable = [0, 0, 0, 0]

    if food_pos.y < head.y:
        FoodTable[0] = 1
    elif food_pos.y > head.y:
        FoodTable[1] = 1
    if food_pos.x > head.x:
        FoodTable[2] = 1
    elif food_pos.x < head.x:
        FoodTable[3] = 1

    directions = {
        0: Vector(head.x, head.y - 1),
        1: Vector(head.x, head.y + 1),
        2: Vector(head.x + 1, head.y),
        3: Vector(head.x - 1, head.y),
    }
    
    for i, direction in directions.items():
        if not direction.within(grid) or direction in snake.body:
            ObstacleTable[i] = 1

    return FoodTable + ObstacleTable


