import random
from collections import deque
from typing import Sequence, Tuple
import numpy as np
from math import tanh
from scipy.special import softmax  # For normalized output probabilities
import pygame

# Umiddelbart er banen større end forventet, og slangen dør ikke hvis den løber ud fra de 30x30. Kig på dette!!!!!!!

# --------------------
# Spilkode (uændret fra dit originale spil)
# --------------------
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
        pygame.init()
        self.screen = pygame.display.set_mode((xsize * scale, ysize * scale))
        self.clock = pygame.time.Clock()
        self.color_snake_head = (0, 255, 0)
        self.color_food = (255, 0, 0)

    def __del__(self):
        pygame.quit()

    def block(self, obj):
        return (obj.x * self.scale, obj.y * self.scale, self.scale, self.scale)

# --------------------
# Hjælpefunktioner til DNN
# --------------------
def update_input_tables(snake, food, grid):
    head = snake.p
    food_pos = food.p

    # Initialize tables
    FoodTable = [0, 0, 0, 0]  # N, S, E, W
    ObstacleTable = [0, 0, 0, 0]  # N, S, E, W

    # Food direction (baseret på kvadrant)
    if food_pos.y < head.y:
        FoodTable[0] = 1  # North
    elif food_pos.y > head.y:
        FoodTable[1] = 1  # South
    if food_pos.x > head.x:
        FoodTable[2] = 1  # East
    elif food_pos.x < head.x:
        FoodTable[3] = 1  # West

    # Forhindringer (vægge og krop)
    directions = {
        0: Vector(head.x, head.y - 1),  # North
        1: Vector(head.x, head.y + 1),  # South
        2: Vector(head.x + 1, head.y),  # East
        3: Vector(head.x - 1, head.y),  # West
    }
    for i, direction in directions.items():
        if not direction.within(grid) or direction in snake.body:
            ObstacleTable[i] = 1

    # Debugging
    #print(f"FoodTable: {FoodTable}, ObstacleTable: {ObstacleTable}")

    return FoodTable + ObstacleTable  # Returner én samlet observation



def fitness_function(agent, steps, food_count, efficiency_score):
    # Belønning for mad
    food_reward = food_count * 1000 + max(0, (500 - steps // food_count)) if food_count > 0 else 0
    
    # Effektivitet (belønning for at reducere afstand)
    efficiency_reward = efficiency_score * 2  # Positiv belønning for at reducere afstand

    # Straf for død
    death_penalty = -1000 if food_count == 0 else 0  # Straf for død uden at spise mad

    # Straf for ineffektivitet (fx mange unødvendige skridt)
    inefficiency_penalty = -steps * 0.5  # Straf for at tage for mange skridt

    # Straf for langsom opførsel (fx lang tid før at spise mad)
    efficiency_score_penalty = -efficiency_score * 10  # Straf for lav effektivitet

    # Samlet fitness
    fitness = food_reward + efficiency_reward + death_penalty + inefficiency_penalty + efficiency_score_penalty
    return fitness





# --------------------
# Neural net og agent-logik
# --------------------
class SimpleModel:
    def __init__(self, *, dims: Tuple[int, ...]):
        assert len(dims) >= 2, 'Error: dims must be two or higher.'
        self.dims = dims
        self.DNA = []
        for i, dim in enumerate(dims):
            if i < len(dims) - 1:
                # Initialize weights with better scaling
                self.DNA.append(np.random.randn(dim, dims[i + 1]) * np.sqrt(2.0 / dim))

    def update(self, obs: Sequence, temperature: float = 1.0) -> np.ndarray:
        """
        Process the observation through the neural network and return action probabilities.
        
        Args:
            obs: Input observation (8 values: 4 for food direction, 4 for obstacles)
            temperature: Controls exploration vs exploitation (higher = more random)
        
        Returns:
            Action probabilities for four possible directions
        """
        x = np.array(obs, dtype=np.float32)
        
        # Add batch dimension if needed
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
            
        # Forward pass through each layer
        for i, layer in enumerate(self.DNA):
            # Linear transformation
            x = x @ layer
            
            # Apply activation function (tanh for hidden layers)
            if i < len(self.DNA) - 1:
                x = np.tanh(x)
        
        # Apply temperature scaling
        x = x / temperature
        
        # Convert to probabilities using softmax
        probs = softmax(x, axis=-1)
        
        return probs.flatten()  # Return flattened array for single sample

    def action(self, obs: Sequence) -> int:
        """
        Choose an action based on the observation.
        
        Args:
            obs: Current game state observation
            
        Returns:
            Integer representing the chosen direction (0: North, 1: South, 2: East, 3: West)
        """
        # Get action probabilities
        probs = self.update(obs)
        
        # Get current valid moves (where obstacles are not present)
        obstacles = obs[4:]  # Last 4 values are obstacles
        valid_moves = np.where(np.array(obstacles) == 0)[0]
        
        if len(valid_moves) > 0:
            # Filter probabilities for valid moves only
            valid_probs = probs[valid_moves]
            
            # Normalize probabilities for valid moves
            valid_probs = valid_probs / valid_probs.sum()
            
            # Choose the valid move with highest probability
            valid_action_idx = np.argmax(valid_probs)
            return valid_moves[valid_action_idx]
        else:
            # If no valid moves, return the direction with highest probability
            # (though the snake will likely die in this case)
            return np.argmax(probs)


    def mutate(self, mutation_rate) -> None:
        for layer in self.DNA:
            if random.random() < mutation_rate:
                mutation_mask = np.random.rand(*layer.shape) < 0.1  # 10% chance for mutation per vægt
                gaussian_noise = np.random.normal(0, 0.01, layer.shape)  # Mindre ændringer (mean=0, std=0.01)
                layer += mutation_mask * gaussian_noise



    def __add__(self, other):
        baby_DNA = []
        for mom_layer, dad_layer in zip(self.DNA, other.DNA):
        # Brug vægtet gennemsnit (f.eks. 50% fra begge)
            blend_factor = random.uniform(0.4, 0.6)  # Juster sandsynligheden for at blande vægtene
            baby_layer = blend_factor * mom_layer + (1 - blend_factor) * dad_layer
            baby_DNA.append(baby_layer)
    
        baby = type(self)(dims=self.dims)
        baby.DNA = baby_DNA
        return baby



# --------------------
# Simulering og træning
# --------------------
def simulate_game(agent, game):
    snake = Snake(game=game)
    food = Food(game=game)
    steps = 0
    food_count = 0
    efficiency_score = 0

    while steps < 500:  # Max steps
        # Generér observation baseret på slangens position og omgivelser
        obs = update_input_tables(snake, food, game.grid)

        # Debugging for at vise observationen (kan fjernes, når du er tilfreds med resultatet)
        #print(f"Observation: {obs}")

        # Få agentens handling baseret på observationen
        action = agent.action(obs)

        # Oversæt handlingen til en bevægelsesvektor
        snake.v = [Vector(0, -1), Vector(0, 1), Vector(1, 0), Vector(-1, 0)][action]

        # Flyt slangen
        snake.move()


        # Tjek for kollisioner med vægge eller egen krop
        if not snake.p.within(game.grid) or snake.cross_own_tail:
            break

        # Tjek for mad
        if snake.p == food.p:
            snake.add_score()
            food = Food(game=game)  # Generér ny mad
            food_count += 1

        steps += 1

    return steps, food_count, efficiency_score


# Breed de 50% bedste agenter
# Breed de 50% bedste agenter
def train_agents(agents, generations, mutation_rate):
    for generation in range(generations):
        fitness_scores = []
        food_counts = []  # For at gemme, hvor meget mad de spiser
        for agent in agents:
            game = SnakeGame()
            steps, food_count, efficiency_score = simulate_game(agent, game)
            fitness_scores.append(fitness_function(agent, steps, food_count, efficiency_score))
            food_counts.append(food_count)

        best_fitness = max(fitness_scores)
        best_food = max(food_counts)
        print(f'Generation {generation + 1}: Best fitness = {best_fitness}, Max food eaten = {best_food}')

        # Sort agents by fitness
        sorted_agents = [agent for _, agent in sorted(zip(fitness_scores, agents), key=lambda pair: pair[0], reverse=True)]
        top_agents = sorted_agents[:len(sorted_agents) // 2]

        # Gem den bedste agent til at vise spillet
        best_agent = top_agents[0]

        # Vis spillet med den bedste agent fra denne generation
        show_game_with_best_agent(best_agent)

        # Breed the top agents
        new_agents = []
        while len(new_agents) < len(agents):
            parent1, parent2 = random.sample(top_agents, 2)
            child = parent1 + parent2
            child.mutate(mutation_rate)
            new_agents.append(child)

        agents = new_agents  # Replace old population with new population


# Det ligner den stopper med at træne efter vi har set den første agent. Tjek dette!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
def show_game_with_best_agent(agent):
    game = SnakeGame()
    snake = Snake(game=game)
    food = Food(game=game)
    steps = 0
    food_count = 0
    efficiency_score = 0

    running = True
    while running:
        # Håndter pygame begivenheder
        for event in pygame.event.get():
            if event.type == pygame.QUIT:  # Hvis brugeren lukker vinduet
                running = False

        # Generér observation baseret på slangens position og omgivelser
        obs = update_input_tables(snake, food, game.grid)

        # Få agentens handling baseret på observationen
        action = agent.action(obs)

        # Oversæt handlingen til en bevægelsesvektor
        snake.v = [Vector(0, -1), Vector(0, 1), Vector(1, 0), Vector(-1, 0)][action]

        # Flyt slangen
        snake.move()

        # Tjek for kollisioner med vægge eller egen krop
        if not snake.p.within(game.grid) or snake.cross_own_tail:
            running = False

        # Tjek for mad
        if snake.p == food.p:
            snake.add_score()
            food = Food(game=game)  # Generér ny mad
            food_count += 1

        # Opdater skærmen
        game.screen.fill((0, 0, 0))  # Ryd skærmen
        pygame.draw.rect(game.screen, game.color_food, game.block(food.p))
        for segment in snake.body:
            pygame.draw.rect(game.screen, game.color_snake_head, game.block(segment))

        pygame.display.update()  # Opdater skærmen

        # Hastighed på spillet (kan justeres)
        game.clock.tick(10)

        steps += 1

    pygame.quit()





# --------------------
# Start træning
# --------------------
if __name__ == "__main__":
    population_size = 500  # Brug færre agenter for at teste
    generations = 50  # Øg antallet af generationer for at give træningen mere tid
    mutation_rate = 0.1

    agents = [SimpleModel(dims=(8, 12, 4)) for _ in range(population_size)]  
    train_agents(agents, generations, mutation_rate)
