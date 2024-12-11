import pygame
import json
import argparse
from pathlib import Path
from snake_game import SnakeGame, Snake, Food, Vector, get_input_state
from snake_ai_visual import SimpleModel
import random

class Replay:
    def __init__(self):
        pygame.init()
        pygame.font.init()
        self.game = SnakeGame()
        self.screen = pygame.display.set_mode((self.game.grid.x * self.game.scale, 
                                             self.game.grid.y * self.game.scale))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('arial', 16)
        self.fps = 20
        
    def load_agent(self, weights_path: Path, metadata_path: Path = None):
        model = SimpleModel.load_weights(str(weights_path))
        metadata = None
        if metadata_path and metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        return model, metadata
    
    def play_agent(self, model, metadata=None):
        if metadata and 'initial_state' in metadata:
            init_state = metadata['initial_state']
            snake = Snake(game=self.game)
            snake.p = Vector(init_state['snake_position']['x'], 
                           init_state['snake_position']['y'])
            food = Food(game=self.game)
            food.p = Vector(init_state['food_position']['x'], 
                          init_state['food_position']['y'])
            if 'random_seed' in init_state:
                # Convert list to proper random state tuple structure
                random_state = (init_state['random_seed'][0], 
                              tuple(init_state['random_seed'][1]), 
                              init_state['random_seed'][2])
                random.setstate(random_state)
            max_steps = init_state.get('max_steps', 1000)
        else:
            snake = Snake(game=self.game)
            food = Food(game=self.game)
            max_steps = 1000
            
        steps = 0
        food_eaten = 0
        running = True
        
        while running and steps < max_steps:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        self.fps = min(self.fps * 2, 120)
                    elif event.key == pygame.K_DOWN:
                        self.fps = max(self.fps // 2, 5)
                        
            state = get_input_state(snake, food, self.game.grid)
            action = model.get_action(state)
            snake.v = [Vector(0, -1), Vector(0, 1), Vector(1, 0), Vector(-1, 0)][action]
            snake.move()
            
            if not snake.p.within(self.game.grid) or snake.cross_own_tail:
                break
                
            if snake.p == food.p:
                food_eaten += 1
                snake.add_score()
                if metadata and 'initial_state' in metadata and 'food_positions' in metadata['initial_state'] and food_eaten < len(metadata['initial_state']['food_positions']):
                    next_food = metadata['initial_state']['food_positions'][food_eaten]
                    food.p = Vector(next_food['x'], next_food['y'])
                else:
                    food = Food(game=self.game)
            
            self.screen.fill((0, 0, 0))
            pygame.draw.rect(self.screen, (255, 0, 0), self.game.block(food.p))
            for segment in snake.body:
                pygame.draw.rect(self.screen, (0, 255, 0), self.game.block(segment))
            
            info_texts = [
                f"Steps: {steps}",
                f"Food: {food_eaten}",
                f"Speed: {self.fps} FPS (UP/DOWN to change)",
            ]
            if metadata:
                info_texts.extend([
                    f"Generation: {metadata['generation']}",
                    f"Original Fitness: {metadata['fitness']:.1f}",
                    f"Original Food: {metadata['food_eaten']}"
                ])
            
            for i, text in enumerate(info_texts):
                text_surface = self.font.render(text, True, (255, 255, 255))
                self.screen.blit(text_surface, (10, 10 + i * 20))
                
            pygame.display.flip()
            self.clock.tick(self.fps)
            steps += 1
        
        pygame.quit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Replay Snake AI Agent')
    parser.add_argument('--run', type=str, help='Training run folder path')
    parser.add_argument('--generation', type=int, help='Generation number to replay (optional)')
    args = parser.parse_args()
    
    run_path = Path(args.run)
    replay = Replay()
    
    if args.generation is not None:
        weights_path = run_path / "generations" / f"generation_{args.generation}" / "weights.json"
        metadata_path = run_path / "generations" / f"generation_{args.generation}" / "metadata.json"
    else:
        weights_path = run_path / "best_ever" / "weights.json"
        metadata_path = run_path / "best_ever" / "metadata.json"
    
    model, metadata = replay.load_agent(weights_path, metadata_path)
    replay.play_agent(model, metadata)