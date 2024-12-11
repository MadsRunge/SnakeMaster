import numpy as np
from typing import Tuple, List
from snake_game import SnakeGame, Snake, Food, Vector, get_input_state
import pygame
import time
import argparse
import random
import matplotlib.pyplot as plt
from pathlib import Path
import json
import datetime
from dataclasses import dataclass

class SimpleModel:
    def __init__(self, input_size: int = 8, hidden_size: int = 12, output_size: int = 4):
        self.hidden_weights = np.random.uniform(-1, 1, (input_size, hidden_size))
        self.output_weights = np.random.uniform(-1, 1, (hidden_size, output_size))
        self.games_played = 0

    def save_weights(self, filepath: str):
        weights = {
            'hidden_weights': self.hidden_weights.tolist(),
            'output_weights': self.output_weights.tolist()
        }
        with open(filepath, 'w') as f:
            json.dump(weights, f)

    @classmethod
    def load_weights(cls, filepath: str) -> 'SimpleModel':
        model = cls()
        with open(filepath, 'r') as f:
            weights = json.load(f)
        model.hidden_weights = np.array(weights['hidden_weights'])
        model.output_weights = np.array(weights['output_weights'])
        return model

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

class TrainingManager:
    def __init__(self):
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.base_dir = Path(f"training_run_1")
        self.generations_dir = self.base_dir / "generations"
        self.best_ever_dir = self.base_dir / "best_ever"
        
        self.base_dir.mkdir(parents=True)
        self.generations_dir.mkdir()
        self.best_ever_dir.mkdir()
        
    def save_plot(self, plt):
        plt.savefig(self.base_dir / "training_progress.png")
        plt.close()
        
    def save_generation_best(self, model, generation: int, fitness: float, food_eaten: int, snake: Snake, food: Food, food_positions: List[Food], max_steps: int):
        gen_dir = self.generations_dir / f"generation_{generation}"
        gen_dir.mkdir()
        
        food_pos_list = [{"x": f.p.x, "y": f.p.y} for f in food_positions]
        random_state = list(random.getstate())  # Convert tuple to list
        
        game_state = {
            "generation": generation,
            "fitness": fitness,
            "food_eaten": food_eaten,
            "timestamp": datetime.datetime.now().isoformat(),
            "initial_state": {
                "snake_position": {"x": snake.p.x, "y": snake.p.y},
                "food_position": {"x": food.p.x, "y": food.p.y},
                "random_seed": random_state,  # Now a list
                "food_positions": food_pos_list,
                "max_steps": max_steps
            }
        }
        
        model.save_weights(str(gen_dir / "weights.json"))
        with open(gen_dir / "metadata.json", "w") as f:
            json.dump(game_state, f, indent=2)

    def save_best_ever(self, model, generation: int, fitness: float, food_eaten: int, snake: Snake, food: Food, food_positions: List[Food], max_steps: int):
        model.save_weights(str(self.best_ever_dir / "weights.json"))
        
        food_pos_list = [{"x": f.p.x, "y": f.p.y} for f in food_positions]
        random_state = list(random.getstate())  # Convert tuple to list
        
        metadata = {
            "generation": generation,
            "fitness": fitness,
            "food_eaten": food_eaten,
            "timestamp": datetime.datetime.now().isoformat(),
            "initial_state": {
                "snake_position": {"x": snake.p.x, "y": snake.p.y},
                "food_position": {"x": food.p.x, "y": food.p.y},
                "random_seed": random_state,  # Now a list
                "food_positions": food_pos_list,
                "max_steps": max_steps
            }
        }
        with open(self.best_ever_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

def evaluate_fitness(model: SimpleModel, game: SnakeGame, initial_max_steps: int = 200) -> Tuple[float, int, Snake, Food, List[Food], int]:
    ABSOLUTE_MAX_STEPS = 2000
    snake = Snake(game=game)
    food = Food(game=game)
    initial_snake = snake
    initial_food = food
    food_positions = [food]
    food_eaten = 0
    total_steps = 0
    steps_since_last_food = 0
    max_steps = min(initial_max_steps, ABSOLUTE_MAX_STEPS)

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
            food_positions.append(food)
            
        if steps_since_last_food > 50:
            break

        total_steps += 1

    base_fitness = food_eaten * 75
    efficiency_bonus = (food_eaten / total_steps) * 75 if total_steps > 0 and food_eaten > 0 else 0
    fitness = base_fitness + efficiency_bonus
    if food_eaten == 0:
        fitness *= 0.5

    return fitness, food_eaten, initial_snake, initial_food, food_positions, max_steps

def uniform_crossover(parent1: SimpleModel, parent2: SimpleModel) -> SimpleModel:
    child = SimpleModel()
    for weights_name in ['hidden_weights', 'output_weights']:
        parent1_weights = getattr(parent1, weights_name)
        parent2_weights = getattr(parent2, weights_name)
        mask = np.random.rand(*parent1_weights.shape) < 0.5
        setattr(child, weights_name, np.where(mask, parent1_weights, parent2_weights))
    return child

def single_point_crossover(parent1: SimpleModel, parent2: SimpleModel) -> SimpleModel:
    child = SimpleModel()
    
    # Hidden weights crossover
    h_size = parent1.hidden_weights.size
    cut = random.randint(1, h_size-1)
    h_flat1 = parent1.hidden_weights.flatten()
    h_flat2 = parent2.hidden_weights.flatten()
    child_h = np.concatenate([h_flat1[:cut], h_flat2[cut:]])
    child.hidden_weights = child_h.reshape(parent1.hidden_weights.shape)

    # Output weights crossover
    o_size = parent1.output_weights.size
    cut = random.randint(1, o_size-1)
    o_flat1 = parent1.output_weights.flatten()
    o_flat2 = parent2.output_weights.flatten()
    child_o = np.concatenate([o_flat1[:cut], o_flat2[cut:]])
    child.output_weights = child_o.reshape(parent1.output_weights.shape)
    
    return child

def tournament_selection(population: List[SimpleModel], fitness_scores: List[float], k: int = 6) -> SimpleModel:
    selected_indices = np.random.choice(len(population), size=k, replace=False)
    best_idx = selected_indices[np.argmax([fitness_scores[i] for i in selected_indices])]
    return population[best_idx]

def train_population(population_size: int = 200, generations: int = 100, 
                    mutation_rate: float = 0.05, mutation_scale: float = 0.1,
                    elite_size: int = 10, initial_max_steps: int = 200):
    
    manager = TrainingManager()
    game = SnakeGame()
    population = [SimpleModel() for _ in range(population_size)]
    best_fitness_ever = 0
    best_model_ever = None
    best_food_ever = 0
    
    max_food_history = []
    avg_food_history = []
    
    try:
        for generation in range(generations):
            fitness_scores = []
            food_scores = []
            best_states = None
            
            print(f"Evaluating generation {generation + 1}")
            for model in population:
                game_test = SnakeGame()
                fitness, food_count, initial_snake, initial_food, food_positions, max_steps = evaluate_fitness(model, game_test, initial_max_steps)
                fitness_scores.append(fitness)
                food_scores.append(food_count)
                
                if not best_states or fitness > max(fitness_scores[:-1]):
                    best_states = (initial_snake, initial_food, food_positions, max_steps)
            
            best_idx = np.argmax(fitness_scores)
            best_fitness = fitness_scores[best_idx]
            best_model = population[best_idx]
            
            initial_snake, initial_food, food_positions, max_steps = best_states
            manager.save_generation_best(best_model, generation, best_fitness, max(food_scores), 
                                      initial_snake, initial_food, food_positions, max_steps)
            
            if best_fitness > best_fitness_ever:
                best_fitness_ever = best_fitness
                best_model_ever = best_model
                best_food_ever = max(food_scores)
                manager.save_best_ever(best_model, generation, best_fitness, best_food_ever,
                                     initial_snake, initial_food, food_positions, max_steps)
            
            max_food = max(food_scores)
            avg_food = sum(food_scores) / len(food_scores)
            max_food_history.append(max_food)
            avg_food_history.append(avg_food)
            
            plt.figure(figsize=(10, 6))
            plt.plot(max_food_history, label='Best Food', color='green')
            plt.plot(avg_food_history, label='Average Food', color='blue')
            plt.xlabel('Generation')
            plt.ylabel('Food Eaten')
            plt.title('Training Progress')
            plt.legend()
            plt.grid(True)
            manager.save_plot(plt)
            
            avg_fitness = sum(fitness_scores) / len(fitness_scores)
            print(f"Generation {generation + 1}:")
            print(f"  Best fitness: {best_fitness:.1f}")
            print(f"  Average fitness: {avg_fitness:.1f}")
            print(f"  Best food in this gen: {max_food}")
            print(f"  Average food: {avg_food:.1f}")
            print(f"  Best food ever: {best_food_ever}")
            print(f"  Best fitness ever: {best_fitness_ever:.1f}")
            print("-" * 40)
            
            sorted_pairs = sorted(zip(fitness_scores, population), key=lambda pair: pair[0], reverse=True)
            population = [x for _, x in sorted_pairs]
            
            next_population = []
            next_population.extend(population[:elite_size])
            
            while len(next_population) < population_size:
                parent1 = tournament_selection(population, fitness_scores, k=6)
                parent2 = tournament_selection(population, fitness_scores, k=6)
                child = single_point_crossover(parent1, parent2)
                child.mutate(mutation_rate, mutation_scale)
                next_population.append(child)
            
            population = next_population
            
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        
    return best_model_ever, max_food_history, avg_food_history

if __name__ == "__main__":
    POPULATION_SIZE = 200
    GENERATIONS = 100
    MUTATION_RATE = 0.05
    MUTATION_SCALE = 0.1
    ELITE_SIZE = 50
    INITIAL_MAX_STEPS = 200
    
    best_model = train_population(
        population_size=POPULATION_SIZE,
        generations=GENERATIONS,
        mutation_rate=MUTATION_RATE,
        mutation_scale=MUTATION_SCALE,
        elite_size=ELITE_SIZE,
        initial_max_steps=INITIAL_MAX_STEPS
    )