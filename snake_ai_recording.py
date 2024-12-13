import numpy as np
from typing import Tuple, List
from snake_game import SnakeGame, Snake, Food, Vector, get_input_state
import pygame
import time
import argparse
import random
import os
import subprocess
from pathlib import Path

class RecordingManager:
    def __init__(self, run_name: str = "training_run_recording"):
        self.base_dir = Path(run_name)
        self.recordings_dir = self.base_dir / "recordings"
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.recordings_dir.mkdir(exist_ok=True)
        
    def save_frame(self, surface: pygame.Surface, generation: int, frame: int):
        filename = f"gen_{generation:03d}_frame_{frame:04d}.png"
        pygame.image.save(surface, str(self.recordings_dir / filename))
        
    def verify_files(self):
        files = sorted(os.listdir(self.recordings_dir))
        if not files:
            print("Ingen filer fundet i recordings mappen!")
            return
        print(f"Første fil: {files[0]}")
        print(f"Sidste fil: {files[-1]}")
        print(f"Antal filer: {len(files)}")
    
    def create_video(self, output_file: str, fps: int = 30):
        self.verify_files()
        cmd = [
            'ffmpeg',
            '-y',
            '-framerate', str(fps),
            '-pattern_type', 'sequence',
            '-start_number', '1',
            '-i', str(self.recordings_dir / 'gen_%03d_frame_%04d.png'),
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            output_file
        ]
        print(f"Kører kommando: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print("FFmpeg fejl:")
            print(result.stderr)

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
        for weights in [self.hidden_weights, self.output_weights]:
            for w_idx in np.ndindex(weights.shape):
                if random.random() < mutation_rate:
                    weights[w_idx] += np.random.normal(0, mutation_scale)

def two_point_crossover(parent1: SimpleModel, parent2: SimpleModel) -> SimpleModel:
    child = SimpleModel()
    
    h_size = parent1.hidden_weights.size
    cut1 = random.randint(1, h_size-2)
    cut2 = random.randint(cut1+1, h_size-1)
    
    h_flat1 = parent1.hidden_weights.flatten()
    h_flat2 = parent2.hidden_weights.flatten()
    child_h = np.concatenate([h_flat1[:cut1], h_flat2[cut1:cut2], h_flat1[cut2:]])
    child.hidden_weights = child_h.reshape(parent1.hidden_weights.shape)
    
    o_size = parent1.output_weights.size
    cut1 = random.randint(1, o_size-2)
    cut2 = random.randint(cut1+1, o_size-1)
    
    o_flat1 = parent1.output_weights.flatten()
    o_flat2 = parent2.output_weights.flatten()
    child_o = np.concatenate([o_flat1[:cut1], o_flat2[cut1:cut2], o_flat1[cut2:]])
    child.output_weights = child_o.reshape(parent1.output_weights.shape)
    
    return child

def evaluate_fitness(model: SimpleModel, game: SnakeGame, generation: int = None, 
                    recording_manager: RecordingManager = None, initial_max_steps: int = 200) -> Tuple[float, int]:
    ABSOLUTE_MAX_STEPS = 2000 
    
    model.games_played += 1
    snake = Snake(game=game)
    food = Food(game=game)
    food_eaten = 0
    total_steps = 0
    steps_since_last_food = 0
    max_steps = min(initial_max_steps, ABSOLUTE_MAX_STEPS)
    frame = 0

    # Setup recording if needed
    if recording_manager and generation is not None:
        # Initialize pygame and font system
        if not pygame.get_init():
            pygame.init()
        if not pygame.font.get_init():
            pygame.font.init()
            
        screen = pygame.Surface((game.grid.x * game.scale, game.grid.y * game.scale))
        font = pygame.font.SysFont('arial', 16)

    while total_steps < max_steps and total_steps < ABSOLUTE_MAX_STEPS:
        state = get_input_state(snake, food, game.grid)
        action = model.get_action(state)
        snake.v = [Vector(0, -1), Vector(0, 1), Vector(1, 0), Vector(-1, 0)][action]
        snake.move()
        
        if not snake.p.within(game.grid) or snake.cross_own_tail:
            break

        steps_since_last_food += 1
        if snake.p == food.p:
            food_eaten += 1
            max_steps = min(max_steps + 30, ABSOLUTE_MAX_STEPS)
            steps_since_last_food = 0
            snake.add_score()
            food = Food(game=game)
            
        if steps_since_last_food > 50:
            break

        # Record frame if recording manager is provided
        if recording_manager and generation is not None:
            screen.fill((0, 0, 0))
            pygame.draw.rect(screen, (255, 0, 0), game.block(food.p))
            for segment in snake.body:
                pygame.draw.rect(screen, (0, 255, 0), game.block(segment))
            
            info_texts = [
                f"Generation: {generation + 1}",
                f"Steps: {total_steps}",
                f"Food: {food_eaten}"
            ]
            
            for i, text in enumerate(info_texts):
                text_surface = font.render(text, True, (255, 255, 255))
                screen.blit(text_surface, (10, 10 + i * 20))
            
            recording_manager.save_frame(screen, generation + 1, frame)
            frame += 1

        total_steps += 1

    base_fitness = food_eaten * 75
    efficiency_bonus = (food_eaten / total_steps) * 75 if total_steps > 0 and food_eaten > 0 else 0
    fitness = base_fitness + efficiency_bonus
    if food_eaten == 0:
        fitness *= 0.5

    return fitness, food_eaten

def tournament_selection(population: List[SimpleModel], fitness_scores: List[float], k: int = 6) -> SimpleModel:
    selected_indices = np.random.choice(len(population), size=k, replace=False)
    best_idx = selected_indices[np.argmax([fitness_scores[i] for i in selected_indices])]
    return population[best_idx]

def train_population(population_size: int = 200, generations: int = 100, 
                     mutation_rate: float = 0.05, mutation_scale: float = 0.1,
                     elite_size: int = 10, initial_max_steps: int = 200,
                     visual: bool = False):
    
    recording_manager = RecordingManager()
    game = SnakeGame()
    population = [SimpleModel() for _ in range(population_size)]
    best_fitness_ever = 0
    best_model_ever = None
    best_food_ever = 0
    
    try:
        for generation in range(generations):
            fitness_scores = []
            food_scores = []
            
            print(f"Evaluerer generation {generation + 1}")
            
            # Først evaluerer vi alle modeller uden optagelse
            for model in population:
                game_test = SnakeGame()
                fitness, food_count = evaluate_fitness(
                    model, game_test, None, None, initial_max_steps
                )
                fitness_scores.append(fitness)
                food_scores.append(food_count)
            
            # Find den bedste model fra denne generation
            best_idx = np.argmax(fitness_scores)
            best_fitness = fitness_scores[best_idx]
            best_model = population[best_idx]
            
            # Kør den bedste model igen med optagelse
            game_test = SnakeGame()
            _, _ = evaluate_fitness(
                best_model, 
                game_test, 
                generation,
                recording_manager, 
                initial_max_steps
            )
            
            # Opdater best ever tracking
            if best_fitness > best_fitness_ever:
                best_fitness_ever = best_fitness
                best_model_ever = best_model
                best_food_ever = max(food_scores)
            
            # Vis visuelt hvis flag er sat
            if visual:
                print(f"\nViser den bedste agent fra generation {generation + 1}...")
                game_viz = SnakeGame()
                viz_fitness, viz_food = visualize_game(best_model, game_viz, initial_max_steps, generation + 1)
            
            # Beregn og vis statistik
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
            
            # Sorter population efter fitness
            sorted_pairs = sorted(zip(fitness_scores, population), key=lambda pair: pair[0], reverse=True)
            population = [x for _, x in sorted_pairs]
            
            # Lav næste generation
            next_population = []
            next_population.extend(population[:elite_size])  # Elitism
            
            # Fyld resten af population med børn
            while len(next_population) < population_size:
                parent1 = tournament_selection(population, fitness_scores, k=4)
                parent2 = tournament_selection(population, fitness_scores, k=4)
                child = two_point_crossover(parent1, parent2)
                child.mutate(mutation_rate, mutation_scale)
                next_population.append(child)
            
            population = next_population
        
        print("Creating evolution video...")
        recording_manager.create_video(str(recording_manager.base_dir / "evolution.mp4"))
            
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        print("Creating evolution video from recorded frames...")
        recording_manager.create_video(str(recording_manager.base_dir / "evolution.mp4"))
    
    return best_model_ever

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Snake AI with Recording')
    parser.add_argument('--visual', action='store_true', help='Vis træningen visuelt')
    args = parser.parse_args()
    
    POPULATION_SIZE = 200
    GENERATIONS = 100
    MUTATION_RATE = 0.05
    MUTATION_SCALE = 0.1
    ELITE_SIZE = 70
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