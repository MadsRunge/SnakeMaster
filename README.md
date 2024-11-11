# Snake Game with Genetic Algorithm

The Snake Game is a classic video game originating from the 8-bit era. The objective is to steer a snake around an arena to eat food (dots) and grow longer. Each time the snake eats a dot, the player scores a point, and the snake’s body extends by one segment. The game ends if the snake runs into the walls or its own tail.

This project contains a base version of the Snake Game implemented with Python and Pygame, alongside additional branches that support AI integration through a Genetic Algorithm (GA). The challenge is to replace the human player with a computer-controlled snake using GA to train a population of snake agents, aiming to create a snake that can navigate efficiently and maximize its score.

## Project Overview

### Structure

- **Vector Class**: Manages 2D coordinates for the snake and food, providing vector operations and methods to check positions.
- **SnakeGame Class**: Handles game setup and the main game loop, rendering the snake and food, handling player input, and updating game states.
- **Food Class**: Places food randomly on the grid within the game boundaries.
- **Snake Class**: Manages the snake’s movement, score, and length, as well as checking if the snake collides with itself.

### Controls

- Arrow keys control the snake’s direction:
  - **Up**: Move up
  - **Down**: Move down
  - **Left**: Move left
  - **Right**: Move right
- The game ends when the snake collides with a wall or its own body.

## Genetic Algorithm Integration

In this project, the goal is to replace human control with a Genetic Algorithm:
- **Objective**: Train a population of snakes to maximize their survival time and score by evolving better navigation strategies.
- **Implementation**: Modify the game to automatically steer the snake using a GA model, which will allow for autonomous learning and optimization.
