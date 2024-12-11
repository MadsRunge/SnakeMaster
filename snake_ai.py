import numpy as np
from typing import Tuple, List
from snake_game import SnakeGame, Snake, Food, Vector, get_input_state
import pygame
import time
import argparse
import random

class SimpleModel:
    def __init__(self, input_size: int = 8, hidden_size: int = 12, output_size: int = 4):
        self.hidden_weights = np.random.uniform(-1, 1, (input_size, hidden_size))
        self.output_weights = np.random.uniform(-1, 1, (hidden_size, output_size))
        self.games_played = 0

    def get_action(self, observations: List[int]) -> int:
        x = np.array(observations, dtype=np.float32)
        hidden = np.tanh(x @ self.hidden_weights)
        output = hidden @ self.output_weights

        exp_output = np.exp(output)
        probabilities = exp_output / exp_output.sum()

        return np.argmax(probabilities)

    def mutate(self, mutation_rate: float = 0.05, mutation_scale: float = 0.1):
        # Muter vægte med lille sandsynlighed og små ændringer
        for weights in [self.hidden_weights, self.output_weights]:
            for w_idx in np.ndindex(weights.shape):
                if random.random() < mutation_rate:
                    # Tilføj lille gaussian støj i stedet for at erstatte helt.
                    weights[w_idx] += np.random.normal(0, mutation_scale)

def uniform_crossover(parent1: SimpleModel, parent2: SimpleModel) -> SimpleModel:
    child = SimpleModel()

    # Uniform crossover for hidden_weights
    mask = np.random.rand(*parent1.hidden_weights.shape) < 0.5
    child.hidden_weights = np.where(mask, parent1.hidden_weights, parent2.hidden_weights)
    
    # Uniform crossover for output_weights
    mask = np.random.rand(*parent1.output_weights.shape) < 0.5
    child.output_weights = np.where(mask, parent1.output_weights, parent2.output_weights)
    
    return child

def single_point_crossover(parent1: SimpleModel, parent2: SimpleModel) -> SimpleModel:
    child = SimpleModel()
    # Single point for hidden_weights
    h_size = parent1.hidden_weights.size
    cut = random.randint(1, h_size-1)
    h_flat1 = parent1.hidden_weights.flatten()
    h_flat2 = parent2.hidden_weights.flatten()
    child_h = np.concatenate([h_flat1[:cut], h_flat2[cut:]])
    child.hidden_weights = child_h.reshape(parent1.hidden_weights.shape)

    # Single point for output_weights
    o_size = parent1.output_weights.size
    cut = random.randint(1, o_size-1)
    o_flat1 = parent1.output_weights.flatten()
    o_flat2 = parent2.output_weights.flatten()
    child_o = np.concatenate([o_flat1[:cut], o_flat2[cut:]])
    child.output_weights = child_o.reshape(parent1.output_weights.shape)

    return child

def evaluate_fitness(model: SimpleModel, game: SnakeGame, initial_max_steps: int = 200) -> Tuple[float, int]:
    """
    Forbedret fitnessfunktion:
    - Start med initial_max_steps
    - For hvert mad stykke spist, tilføj 50 ekstra steps
    - Beløn hurtig madspisning: Jo færre steps mellem mad, jo højere bonus
    - Hvis > 50 steps uden mad -> stop simulation for at undgå "stalling"
    """

    model.games_played += 1
    snake = Snake(game=game)
    food = Food(game=game)

    food_eaten = 0
    total_steps = 0
    steps_since_last_food = 0
    max_steps = initial_max_steps

    while total_steps < max_steps:
        state = get_input_state(snake, food, game.grid)
        action = model.get_action(state)
        snake.v = [Vector(0, -1), Vector(0, 1), Vector(1, 0), Vector(-1, 0)][action]
        snake.move()
        
        if not snake.p.within(game.grid) or snake.cross_own_tail:
            break  # Død

        steps_since_last_food += 1

        if snake.p == food.p:
            food_eaten += 1
            # Forlæng spilletid pr. mad
            max_steps += 50
            # Bonus for hurtig spisning: jo færre steps siden sidst, jo større bonus
            # Eksempel: bonus = max(0, 50 - steps_since_last_food)
            # Justér efter behov
            steps_since_last_food = 0
            food = Food(game=game)
            
        # Hvis slangen ikke spiser noget mad i f.eks. 50 steps, stop
        if steps_since_last_food > 50:
            break

        total_steps += 1

    # Basis fitness
    fitness = food_eaten * 1000 + total_steps

    # Mulighed for yderligere finjustering:
    # Giv en lille straf, hvis ingen mad er spist
    if food_eaten == 0:
        fitness *= 0.5

    return fitness, food_eaten

def visualize_game(model: SimpleModel, game: SnakeGame, max_steps: int = 300, 
                   generation: int = 0, fps: int = 20) -> Tuple[float, int]:
    pygame.init()
    pygame.font.init()
    screen = pygame.display.set_mode((game.grid.x * game.scale, game.grid.y * game.scale))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont('arial', 16)
    
    snake = Snake(game=game)
    food = Food(game=game)
    steps = 0
    food_eaten = 0
    
    running = True
    while running and steps < max_steps:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    fps = 100
                if event.key == pygame.K_BACKSPACE:
                    fps = 20

        state = get_input_state(snake, food, game.grid)
        action = model.get_action(state)
        snake.v = [Vector(0, -1), Vector(0, 1), Vector(1, 0), Vector(-1, 0)][action]
        snake.move()

        if not snake.p.within(game.grid) or snake.cross_own_tail:
            pygame.quit()
            return evaluate_fitness(model, game, steps)
            
        if snake.p == food.p:
            food_eaten += 1
            snake.add_score()
            food = Food(game=game)
            
        steps += 1
        
        screen.fill((0, 0, 0))
        
        pygame.draw.rect(screen, (255, 0, 0), game.block(food.p))
        for segment in snake.body:
            pygame.draw.rect(screen, (0, 255, 0), game.block(segment))
            
        info_texts = [
            f'Generation: {generation}',
            f'Steps: {steps}',
            f'Food: {food_eaten}',
            f'FPS: {fps} (SPACE: hurtig, BACKSPACE: normal)'
        ]
        
        for i, text in enumerate(info_texts):
            text_surface = font.render(text, True, (255, 255, 255))
            screen.blit(text_surface, (10, 10 + i * 20))
        
        pygame.display.update()
        clock.tick(fps)
    
    pygame.quit()
    return food_eaten * 1000 + steps, food_eaten 

def tournament_selection(population: List[SimpleModel], fitness_scores: List[float], k: int = 4) -> SimpleModel:
    """
    Tournament selection:
    - Vælg k tilfældige individer fra populationen
    - Returnér den bedste af disse
    """
    selected_indices = np.random.choice(len(population), size=k, replace=False)
    best_idx = selected_indices[np.argmax([fitness_scores[i] for i in selected_indices])]
    return population[best_idx]

def train_population(population_size: int = 200, generations: int = 100, 
                     mutation_rate: float = 0.05, mutation_scale: float = 0.1,
                     elite_size: int = 10, initial_max_steps: int = 200,
                     visual: bool = False):
    game = SnakeGame()
    population = [SimpleModel() for _ in range(population_size)]
    best_fitness_ever = 0
    best_model_ever = None
    best_food_ever = 0
    
    for generation in range(generations):
        fitness_scores = []
        food_scores = []
        
        print(f"Evaluerer generation {generation + 1}")
        for model in population:
            game_test = SnakeGame()
            fitness, food_count = evaluate_fitness(model, game_test, initial_max_steps)
            fitness_scores.append(fitness)
            food_scores.append(food_count)
        
        best_idx = np.argmax(fitness_scores)
        best_fitness = fitness_scores[best_idx]
        best_model = population[best_idx]
        
        if best_fitness > best_fitness_ever:
            best_fitness_ever = best_fitness
            best_model_ever = best_model
            best_food_ever = max(food_scores)
        
        if visual:
            print(f"\nViser den bedste agent fra generation {generation + 1}...")
            game_viz = SnakeGame()
            viz_fitness, viz_food = visualize_game(best_model, game_viz, initial_max_steps, generation + 1)
        
        avg_fitness = sum(fitness_scores) / len(fitness_scores)
        avg_food = sum(food_scores) / len(food_scores)
        best_food_current = max(food_scores)
        
        print(f"Generation {generation + 1}:")
        print(f"  Bedste fitness: {best_fitness:.1f}")
        print(f"  Gennemsnit fitness: {avg_fitness:.1f}")
        print(f"  Bedste mad i denne gen: {best_food_current}")
        print(f"  Gennemsnit mad: {avg_food:.1f}")
        print(f"  Bedste mad nogensinde: {best_food_ever}")
        print(f"  Bedste fitness nogensinde: {best_fitness_ever:.1f}")
        print("-" * 40)
        
        # Sorter populationen efter fitness
        sorted_pairs = sorted(zip(fitness_scores, population), key=lambda pair: pair[0], reverse=True)
        population = [x for _, x in sorted_pairs]
        
        next_population = []
        # Elite
        next_population.extend(population[:elite_size])
        
        # Resten via tournament selection + crossover + mutation
        while len(next_population) < population_size:
            parent1 = tournament_selection(population, fitness_scores, k=4)
            parent2 = tournament_selection(population, fitness_scores, k=4)
            child = single_point_crossover(parent1, parent2)  # eller single_point_crossover(parent1, parent2)
            child.mutate(mutation_rate, mutation_scale)
            next_population.append(child)
        
        population = next_population
    
    return best_model_ever

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Snake AI')
    parser.add_argument('--visual', action='store_true', help='Vis træningen visuelt')
    args = parser.parse_args()
    
    # Træningsparametre
    POPULATION_SIZE = 400
    GENERATIONS = 100
    MUTATION_RATE = 0.05
    MUTATION_SCALE = 0.1
    ELITE_SIZE = 10
    INITIAL_MAX_STEPS = 200
    
    best_model = train_population(
        population_size=POPULATION_SIZE,
        generations=GENERATIONS,
        mutation_rate=MUTATION_RATE,
        mutation_scale=MUTATION_SCALE,
        elite_size=ELITE_SIZE,
        initial_max_steps=INITIAL_MAX_STEPS,
        visual=args.visual
    )
