import numpy as np
from typing import Tuple, List
from snake_game import SnakeGame, Snake, Food, Vector, get_input_state
import pygame
import time

class SimpleModel:
    """
    En simpel neural network model der styrer slangen.
    Input: 8 neuroner (4 for mad position, 4 for forhindringer)
    Output: 4 neuroner (op, ned, højre, venstre)
    """
    def __init__(self, input_size: int = 8, hidden_size: int = 12, output_size: int = 4):
        # Initialiserer vægte med små tilfældige værdier
        self.hidden_weights = np.random.randn(input_size, hidden_size) * 0.1
        self.output_weights = np.random.randn(hidden_size, output_size) * 0.1
        self.games_played = 0

    def get_action(self, observations: List[int]) -> int:
        """
        Bestemmer slangens næste bevægelse baseret på spil-tilstanden.
        Returnerer: 0 (op), 1 (ned), 2 (højre), eller 3 (venstre)
        """
        # Konverterer input til numpy array
        x = np.array(observations, dtype=np.float32)
        
        # Forward pass gennem netværket
        hidden = np.maximum(0, x @ self.hidden_weights)  # ReLU aktivering
        output = hidden @ self.output_weights
        
        # Find gyldige bevægelser (hvor der ikke er forhindringer)
        obstacles = observations[4:]  # De sidste 4 værdier er forhindringer
        valid_moves = [i for i, obs in enumerate(obstacles) if obs == 0]
        
        if valid_moves:
            # Vælg den bedste gyldige bevægelse
            valid_outputs = [output[i] for i in valid_moves]
            return valid_moves[np.argmax(valid_outputs)]
        return np.argmax(output)  # Hvis ingen gyldige træk, vælg bedste output

    def mutate(self, mutation_rate: float = 0.1, mutation_scale: float = 0.1):
        """
        Muterer modellens vægte med en given sandsynlighed og styrke.
        """
        # Mutation af hidden layer vægte
        mask = np.random.random(self.hidden_weights.shape) < mutation_rate
        self.hidden_weights += mask * np.random.randn(*self.hidden_weights.shape) * mutation_scale
        
        # Mutation af output layer vægte
        mask = np.random.random(self.output_weights.shape) < mutation_rate
        self.output_weights += mask * np.random.randn(*self.output_weights.shape) * mutation_scale

def crossover(parent1: SimpleModel, parent2: SimpleModel) -> SimpleModel:
    """
    Laver en ny model ved at kombinere to forældres vægte med vægtet gennemsnit.
    """
    child = SimpleModel()
    weight = 0.5  # 50/50 vægtning mellem forældrene
    
    # Vægtet gennemsnit af vægtene
    child.hidden_weights = parent1.hidden_weights * weight + parent2.hidden_weights * (1-weight)
    child.output_weights = parent1.output_weights * weight + parent2.output_weights * (1-weight)
    
    return child

def visualize_game(model: SimpleModel, game: SnakeGame, max_steps: int = 300, 
                  generation: int = 0, fps: int = 20) -> Tuple[float, int]:
    """
    Kører et spil med visualisering og returnerer fitness og mad spist.
    """
    snake = Snake(game=game)
    food = Food(game=game)
    steps = 0
    food_eaten = 0
    clock = pygame.time.Clock()
    
    # Opsæt tekst
    pygame.font.init()
    font = pygame.font.SysFont('arial', 16)
    
    while steps < max_steps:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return steps + (food_eaten * 200), food_eaten
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:  # Tryk space for at øge hastighed
                    fps = 100
                if event.key == pygame.K_BACKSPACE:  # Tryk backspace for normal hastighed
                    fps = 20
                
        state = get_input_state(snake, food, game.grid)
        action = model.get_action(state)
        snake.v = [Vector(0, -1), Vector(0, 1), Vector(1, 0), Vector(-1, 0)][action]
        snake.move()
        
        if not snake.p.within(game.grid) or snake.cross_own_tail:
            break
            
        if snake.p == food.p:
            snake.add_score()
            food = Food(game=game)
            food_eaten += 1
            
        steps += 1
        
        # Opdater display
        game.screen.fill((0, 0, 0))
        
        # Tegn mad og slange
        pygame.draw.rect(game.screen, (255, 0, 0), game.block(food.p))
        for segment in snake.body:
            pygame.draw.rect(game.screen, (0, 255, 0), game.block(segment))
            
        # Vis information
        info_texts = [
            f'Generation: {generation}',
            f'Steps: {steps}',
            f'Food: {food_eaten}',
            f'FPS: {fps} (SPACE: hurtig, BACKSPACE: normal)'
        ]
        
        for i, text in enumerate(info_texts):
            text_surface = font.render(text, True, (255, 255, 255))
            game.screen.blit(text_surface, (10, 10 + i * 20))
        
        pygame.display.update()
        clock.tick(fps)
        
    return steps + (food_eaten * 200), food_eaten

def train_population_visual(population_size: int = 200, generations: int = 100, 
                          mutation_rate: float = 0.1, mutation_scale: float = 0.1,
                          elite_size: int = 10, max_steps: int = 300):
    """
    Træner population med visualisering af den bedste agent i hver generation.
    """
    # Initialiser spil
    game = SnakeGame()
    population = [SimpleModel() for _ in range(population_size)]
    best_fitness_ever = 0
    best_model_ever = None
    best_food_ever = 0
    
    try:
        for generation in range(generations):
            fitness_scores = []
            food_scores = []
            
            # Evaluer alle modeller uden visualisering først
            for i, model in enumerate(population):
                print(f"\rEvaluerer model {i+1}/{population_size}", end="")
                fitness, food_count = visualize_game(model, game, max_steps, generation + 1)
                fitness_scores.append(fitness)
                food_scores.append(food_count)
            print()  # Ny linje efter progress
            
            # Find den bedste model i denne generation
            best_idx = np.argmax(fitness_scores)
            best_fitness = fitness_scores[best_idx]
            best_model = population[best_idx]
            
            # Gem den bedste model nogensinde
            if best_fitness > best_fitness_ever:
                best_fitness_ever = best_fitness
                best_model_ever = best_model
                best_food_ever = max(food_scores)
            
            # Print statistik
            avg_fitness = sum(fitness_scores) / len(fitness_scores)
            avg_food = sum(food_scores) / len(food_scores)
            print(f"\nGeneration {generation + 1}:")
            print(f"  Bedste fitness: {best_fitness:.1f}")
            print(f"  Gennemsnit fitness: {avg_fitness:.1f}")
            print(f"  Bedste mad i denne gen: {max(food_scores)}")
            print(f"  Gennemsnit mad: {avg_food:.1f}")
            print(f"  Bedste mad nogensinde: {best_food_ever}")
            print(f"  Bedste fitness nogensinde: {best_fitness_ever:.1f}")
            print("-" * 40)
            
            # Sorter og lav næste generation
            sorted_pairs = sorted(zip(fitness_scores, population), key=lambda pair: pair[0], reverse=True)
            population = [x for _, x in sorted_pairs]
            
            # Lav næste generation
            next_population = []
            next_population.extend(population[:elite_size])
            
            while len(next_population) < population_size:
                weights = np.arange(len(population), 0, -1)
                parents = np.random.choice(population, size=2, p=weights/weights.sum())
                child = crossover(parents[0], parents[1])
                child.mutate(mutation_rate, mutation_scale)
                next_population.append(child)
                
            population = next_population
            
    except KeyboardInterrupt:
        print("\nTræning afbrudt af bruger")
        pygame.quit()
    
    return best_model_ever

if __name__ == "__main__":
    # Træningsparametre - samme som i din originale kode
    POPULATION_SIZE = 200    # Antal modeller i hver generation
    GENERATIONS = 100       # Antal generationer der trænes
    MUTATION_RATE = 0.1     # Sandsynlighed for mutation (0-1)
    MUTATION_SCALE = 0.1    # Hvor meget mutation ændrer vægtene
    ELITE_SIZE = 10         # Antal af de bedste modeller der føres direkte videre
    MAX_STEPS = 300        # Maksimalt antal skridt per spil
    
    # Start træning med visualisering
    best_model = train_population_visual(
        population_size=POPULATION_SIZE,
        generations=GENERATIONS,
        mutation_rate=MUTATION_RATE,
        mutation_scale=MUTATION_SCALE,
        elite_size=ELITE_SIZE,
        max_steps=MAX_STEPS
    )