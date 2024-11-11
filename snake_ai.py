# snake_ai.py
import numpy as np
from snake import Snake, SnakeGame, Food, Vector
import random
from collections import deque
import pygame
# hej
class NeuralNetwork:
    def __init__(self, input_size: int = 24, hidden_size: int = 16, output_size: int = 4):
        self.weights1 = np.random.randn(input_size, hidden_size) / np.sqrt(input_size)
        self.weights2 = np.random.randn(hidden_size, output_size) / np.sqrt(hidden_size)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.hidden = np.tanh(np.dot(x, self.weights1))
        return np.tanh(np.dot(self.hidden, self.weights2))
    
    def mutate(self, rate: float = 0.1, scale: float = 0.2):
        mask1 = np.random.random(self.weights1.shape) < rate
        mask2 = np.random.random(self.weights2.shape) < rate
        self.weights1 += mask1 * np.random.randn(*self.weights1.shape) * scale
        self.weights2 += mask2 * np.random.randn(*self.weights2.shape) * scale

class AISnake(Snake):
    def __init__(self, game: SnakeGame):
        super().__init__(game=game)
        self.brain = NeuralNetwork()
        self.lifetime = 0
        self.moves_without_food = 0
        self.max_moves_without_food = 100
    
    def get_state(self, food: Food) -> np.ndarray:
        directions = [
            Vector(-1, 0),  # Left
            Vector(-1, -1), # Top-left
            Vector(0, -1),  # Up
            Vector(1, -1),  # Top-right
            Vector(1, 0),   # Right
            Vector(1, 1),   # Bottom-right
            Vector(0, 1),   # Down
            Vector(-1, 1)   # Bottom-left
        ]
        
        state = []
        head = self.p
        
        for d in directions:
            # Distance to wall
            pos = head
            distance = 0
            while pos.within(self.game.grid):
                pos = pos + d
                distance += 1
            state.append(1.0 / distance)
            
            # Distance to food in this direction
            pos = head
            distance = 0
            food_found = False
            while pos.within(self.game.grid) and not food_found:
                pos = pos + d
                distance += 1
                if pos == food.p:
                    state.append(1.0 / distance)
                    food_found = True
            if not food_found:
                state.append(0)
                
            # Distance to own body in this direction
            pos = head
            distance = 0
            body_found = False
            while pos.within(self.game.grid) and not body_found:
                pos = pos + d
                distance += 1
                if any(b == pos for b in list(self.body)[1:]):
                    state.append(1.0 / distance)
                    body_found = True
            if not body_found:
                state.append(0)
        
        return np.array(state)
    
    def think(self, food: Food):
        state = self.get_state(food)
        output = self.brain.forward(state)
        
        direction_idx = np.argmax(output)
        if direction_idx == 0:   # Left
            self.v = Vector(-1, 0)
        elif direction_idx == 1: # Up
            self.v = Vector(0, -1)
        elif direction_idx == 2: # Right
            self.v = Vector(1, 0)
        elif direction_idx == 3: # Down
            self.v = Vector(0, 1)

class GeneticAlgorithm:
    def __init__(self, game: SnakeGame, population_size: int = 50):
        self.game = game
        self.population_size = population_size
        self.population = [AISnake(game) for _ in range(population_size)]
        self.generation = 0
        self.best_score = 0
        
    def calculate_fitness(self, snake: AISnake) -> float:
        return snake.score * 100 + snake.lifetime * 0.1
        
    def select_parent(self) -> AISnake:
        tournament_size = 5
        tournament = random.sample(self.population, tournament_size)
        return max(tournament, key=self.calculate_fitness)
    
    def create_offspring(self, parent1: AISnake, parent2: AISnake) -> AISnake:
        child = AISnake(self.game)
        
        for i in range(len(parent1.brain.weights1)):
            if random.random() < 0.5:
                child.brain.weights1[i] = parent1.brain.weights1[i]
            else:
                child.brain.weights1[i] = parent2.brain.weights1[i]
                
        for i in range(len(parent1.brain.weights2)):
            if random.random() < 0.5:
                child.brain.weights2[i] = parent1.brain.weights2[i]
            else:
                child.brain.weights2[i] = parent2.brain.weights2[i]
        
        child.brain.mutate()
        return child
    
    def evolve(self):
        self.population.sort(key=self.calculate_fitness, reverse=True)
        best_snake = self.population[0]
        self.best_score = max(self.best_score, best_snake.score)
        
        new_population = []
        new_population.append(AISnake(self.game))
        new_population[0].brain = self.population[0].brain
        
        while len(new_population) < self.population_size:
            parent1 = self.select_parent()
            parent2 = self.select_parent()
            child = self.create_offspring(parent1, parent2)
            new_population.append(child)
            
        self.population = new_population
        self.generation += 1

class AISnakeGame(SnakeGame):
    def __init__(self, xsize: int = 30, ysize: int = 30, scale: int = 15):
        super().__init__(xsize, ysize, scale)
        self.ga = GeneticAlgorithm(self)
        
    def run_training(self, generations: int = 100):
        for generation in range(generations):
            for snake in self.ga.population:
                running = True
                food = Food(game=self)
                
                while running:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            return
                    
                    snake.think(food)
                    snake.move()
                    snake.lifetime += 1
                    snake.moves_without_food += 1
                    
                    if not snake.p.within(self.grid):
                        running = False
                    if snake.cross_own_tail:
                        running = False
                    if snake.moves_without_food >= snake.max_moves_without_food:
                        running = False
                    if snake.p == food.p:
                        snake.add_score()
                        snake.moves_without_food = 0
                        food = Food(game=self)
                    
                    self.screen.fill('black')
                    for i, p in enumerate(snake.body):
                        pygame.draw.rect(self.screen,
                                     (0, max(128, 255 - i * 8), 0),
                                     self.block(p))
                    pygame.draw.rect(self.screen, self.color_food, self.block(food.p))
                    pygame.display.flip()
                    self.clock.tick(30)
            
            self.ga.evolve()
            print(f"Generation {self.ga.generation}: Best Score = {self.ga.best_score}")

if __name__ == '__main__':
    game = AISnakeGame()
    game.run_training()