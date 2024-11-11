import random
from collections import deque
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
        return self.x <= scope.x and self.x >= 0 and self.y <= scope.y and self.y >= 0

    def __eq__(self, other: 'Vector') -> bool:
        return self.x == other.x and self.y == other.y

    @classmethod
    def random_within(cls, scope: 'Vector') -> 'Vector':
        return Vector(random.randint(0, scope.x - 1), random.randint(0, scope.y - 1))


class SnakeGame:
    def __init__(self, xsize: int = 30, ysize: int = 30, scale: int = 15):
        self.grid = Vector(xsize, ysize)
        self.scale = scale
        pygame.init()
        self.screen = pygame.display.set_mode((xsize * scale, ysize * scale))
        self.clock = pygame.time.Clock()

        self.color_snake_head = (0, 255, 0)
        self.color_food = (255, 0, 0)

    def __del__(self):
        pygame.quit()

    def block(self, obj):
        return (obj.x * self.scale, obj.y * self.scale, self.scale, self.scale)

    def run(self):
        running = True
        snake = Snake(game=self)
        food = Food(game=self)

        while running:

            # handle pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:
                        snake.v = Vector(-1, 0)
                    if event.key == pygame.K_RIGHT:
                        snake.v = Vector(1, 0)
                    if event.key == pygame.K_UP:
                        snake.v = Vector(0, -1)
                    if event.key == pygame.K_DOWN:
                        snake.v = Vector(0, 1)

            # wipe screen
            self.screen.fill('black')

            # update game state
            snake.move()
            if not snake.p.within(self.grid):
                running = False
            if snake.cross_own_tail:
                running = False
            if snake.p == food.p:
                snake.add_score()
                food = Food(game=self)

            # render game
            for i, p in enumerate(snake.body):
                pygame.draw.rect(self.screen,
                                 (0, max(128, 255 - i * 8), 0),
                                 self.block(p))
            pygame.draw.rect(self.screen, self.color_food, self.block(food.p))

            # render screen
            pygame.display.flip()

            # progress time
            self.clock.tick(10)

        print(f'Score: {snake.score}')


class Food:
    def __init__(self, game: SnakeGame):
        self.game = game
        self.p = Vector.random_within(self.game.grid)


class Snake:
    def __init__(self, *, game: SnakeGame):
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

    def debug(self):
        print('===')
        for i in self.body:
            print(str(i))


if __name__ == '__main__':
    game = SnakeGame()
    game.run()
