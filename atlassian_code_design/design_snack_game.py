from collections import deque
from typing import List, Tuple

class SnakeGame:
    def __init__(self, width: int, height: int, food: List[Tuple[int, int]]):
        self.width = width
        self.height = height
        self.food = food
        self.food_index = 0
        
        self.snake = deque([(0, 0)])  # Snake's body; head at front
        self.snake_positions = set([(0, 0)])  # Set for fast collision detection
        
        self.score = 0
        self.directions = {
            'U': (-1, 0),
            'D': (1, 0),
            'L': (0, -1),
            'R': (0, 1)
        }

    def move(self, direction: str) -> int:
        if direction not in self.directions:
            raise ValueError(f"Invalid direction: {direction}")

        current_head = self.snake[0]
        delta_row, delta_col = self.directions[direction]
        new_head = (current_head[0] + delta_row, current_head[1] + delta_col)

        # Check if out of bounds
        if not (0 <= new_head[0] < self.height and 0 <= new_head[1] < self.width):
            return -1  # Game over

        # Remove tail temporarily (to allow movement to previous tail location)
        tail = self.snake.pop()
        self.snake_positions.remove(tail)

        # Check if snake bites itself
        if new_head in self.snake_positions:
            return -1  # Game over

        # Add new head
        self.snake.appendleft(new_head)
        self.snake_positions.add(new_head)

        # Check for food
        if self.food_index < len(self.food) and new_head == tuple(self.food[self.food_index]):
            self.score += 1
            self.food_index += 1
            # Add tail back since snake grows
            self.snake.append(tail)
            self.snake_positions.add(tail)

        return self.score


# Unit Test Case
import unittest

class TestSnakeGame(unittest.TestCase):
    def test_snake_game(self):
        game = SnakeGame(3, 3, [(1, 2), (0, 1)])
        self.assertEqual(game.move("R"), 0)
        self.assertEqual(game.move("D"), 0)
        self.assertEqual(game.move("R"), 1)
        self.assertEqual(game.move("U"), 1)
        self.assertEqual(game.move("L"), 2)
        self.assertEqual(game.move("U"), -1)  # Game over

if __name__ == "__main__":
    unittest.main(argv=[''], exit=False)
55