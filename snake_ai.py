# snake_ai.py
import numpy as np
from snake import Snake, SnakeGame, Food, Vector
import random
from collections import deque
import pygame

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
        self.avg_fitness = 0
        
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
        
        # Calculate average fitness
        total_fitness = sum(self.calculate_fitness(snake) for snake in self.population)
        self.avg_fitness = total_fitness / len(self.population)
        
        new_population = []
        new_population.append(AISnake(self.game))
        new_population[0].brain = self.population[0].brain  # Keep the best performer
        
        while len(new_population) < self.population_size:
            parent1 = self.select_parent()
            parent2 = self.select_parent()
            child = self.create_offspring(parent1, parent2)
            new_population.append(child)
            
        self.population = new_population
        self.generation += 1

class AISnakeGame(SnakeGame):
    def __init__(self, xsize: int = 30, ysize: int = 30, scale: int = 20):
        # Calculate total window size including stats panel
        self.stats_width = 300  # Width of stats panel
        total_width = (xsize * scale) + self.stats_width
        total_height = ysize * scale
        
        # Initialize pygame but override screen creation
        self.grid = Vector(xsize, ysize)
        self.scale = scale
        pygame.init()
        self.screen = pygame.display.set_mode((total_width, total_height))
        self.clock = pygame.time.Clock()
        
        # Create separate surface for game area
        self.game_surface = pygame.Surface((xsize * scale, ysize * scale))
        
        self.color_snake_head = (0, 255, 0)
        self.color_food = (255, 0, 0)
        
        # Stats initialization
        self.ga = GeneticAlgorithm(self)
        self.generation_stats = []
        self.current_gen_scores = []
        self.font = pygame.font.Font(None, 24)
        self.stats_surface = pygame.Surface((self.stats_width, total_height))
        
        # History tracking for plotting
        self.score_history = []
        self.max_history_points = 50
        
    def draw_stats(self, current_snake):
        # Clear stats surface
        self.stats_surface.fill((40, 40, 40))
        
        # Draw separator line
        pygame.draw.line(self.stats_surface, (200, 200, 200), (0, 0), (0, self.grid.y * self.scale), 2)
        
        # Calculate stats
        if self.current_gen_scores:
            avg_score = sum(self.current_gen_scores) / len(self.current_gen_scores)
            max_score = max(self.current_gen_scores)
        else:
            avg_score = 0
            max_score = 0
            
        # Create stats text
        stats_texts = [
            ("Training Stats", (255, 255, 0)),  # Title in yellow
            ("", None),  # Spacing
            (f"Generation: {self.ga.generation}", (255, 255, 255)),
            (f"Population: {len(self.current_gen_scores)}/{self.ga.population_size}", (255, 255, 255)),
            (f"Best Score Ever: {self.ga.best_score}", (0, 255, 0)),  # Green for best score
            ("", None),  # Spacing
            ("Current Generation", (255, 255, 0)),  # Subtitle
            (f"Avg Score: {avg_score:.1f}", (255, 255, 255)),
            (f"Best Score: {max_score}", (255, 255, 255)),
            ("", None),  # Spacing
            ("Current Snake", (255, 255, 0)),  # Subtitle
            (f"Score: {current_snake.score}", (255, 255, 255)),
            (f"Lifetime: {current_snake.lifetime}", (255, 255, 255)),
            (f"Moves without food: {current_snake.moves_without_food}", (255, 255, 255))
        ]
        
        # Draw stats text
        y_offset = 20
        for text, color in stats_texts:
            if color is not None:  # Skip if this is just spacing
                text_surface = self.font.render(text, True, color)
                self.stats_surface.blit(text_surface, (20, y_offset))
            y_offset += 25
            
        # Store and draw score history
        if self.current_gen_scores:
            self.score_history.append(avg_score)
            if len(self.score_history) > self.max_history_points:
                self.score_history = self.score_history[-self.max_history_points:]
            
            # Draw score history graph
            if len(self.score_history) > 1:
                graph_rect = pygame.Rect(20, y_offset + 30, self.stats_width - 40, 100)
                pygame.draw.rect(self.stats_surface, (60, 60, 60), graph_rect)
                
                max_score_history = max(max(self.score_history), 1)  # Avoid division by zero
                points = []
                for i, score in enumerate(self.score_history):
                    x = graph_rect.left + (i * graph_rect.width // (self.max_history_points - 1))
                    y = graph_rect.bottom - (score * graph_rect.height // max_score_history)
                    points.append((x, y))
                
                if len(points) > 1:
                    pygame.draw.lines(self.stats_surface, (0, 255, 0), False, points, 2)
                
                # Draw graph labels
                label = self.font.render("Score History", True, (255, 255, 0))
                self.stats_surface.blit(label, (20, y_offset))
        
    def run_training(self, generations: int = 100):
        for generation in range(generations):
            self.current_gen_scores = []
            
            for snake_idx, snake in enumerate(self.ga.population):
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
                    
                    # Clear game surface
                    self.game_surface.fill('black')
                    
                    # Draw snake and food on game surface
                    for i, p in enumerate(snake.body):
                        pygame.draw.rect(self.game_surface,
                                     (0, max(128, 255 - i * 8), 0),
                                     self.block(p))
                    pygame.draw.rect(self.game_surface, self.color_food, self.block(food.p))
                    
                    # Update stats
                    self.current_gen_scores = [s.score for s in self.ga.population[:snake_idx+1]]
                    self.draw_stats(snake)
                    
                    # Draw everything to main screen
                    self.screen.blit(self.game_surface, (0, 0))
                    self.screen.blit(self.stats_surface, (self.grid.x * self.scale, 0))
                    
                    pygame.display.flip()
                    self.clock.tick(30)
                
                self.current_gen_scores.append(snake.score)
            
            self.ga.evolve()
            print(f"Generation {self.ga.generation}: Best Score = {self.ga.best_score}")

if __name__ == '__main__':
    game = AISnakeGame()
    game.run_training()