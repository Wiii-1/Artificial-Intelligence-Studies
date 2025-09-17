# Rat and Cheese Maze Solver

This project implements a visual maze generator and solver using the A* search algorithm. The maze features a rat (start) and cheese (goal), with walls and paths, and uses a graphical interface for interaction.

## Prerequisites

Before running the script, ensure you have the following:

- Python 3.8 or newer
- Required Python packages:
  - `tkinter` (usually included with Python)
  - `Pillow` (for image handling)
  - `networkx` and `matplotlib` (only needed for [setTheory.py](setTheory.py))
- Image files in the same directory:
  - `rat.png`
  - `cheese.png`
  - `wall.png`

Install Pillow using pip:

```sh
pip install pillow
```

## How to Run

1. Place `Asearch.py`, `rat.png`, `cheese.png`, and `wall.png` in the same folder.
2. Run the script:

```sh
python Asearch.py
```

A window will open showing the maze. Use the buttons to generate a new maze or start the search.

## File Overview

### [Asearch.py](Asearch.py)

#### Classes

- **Node**
  - Represents a cell in the maze for A* search.
  - Attributes:
    - `position`: Tuple (row, col)
    - `g`: Cost from start node
    - `h`: Heuristic cost to goal
    - `f`: Total cost (`g + h`)
    - `parent`: Reference to previous node in path
  - Methods:
    - `__lt__`: Comparison for priority queue (heapq)

- **MazeGUI**
  - Handles the graphical interface and maze logic.
  - Key methods:
    - `__init__`: Sets up the GUI, loads images, and generates the initial maze.
    - `generate_random_maze`: Creates a random maze using recursive backtracking.
    - `generate_new_maze`: Finds valid start and goal positions, redraws the grid.
    - `load_images`: Loads and resizes images for rat, cheese, and wall.
    - `open_and_resize`: Helper for loading and resizing images.
    - `draw_grid`: Draws the maze grid and places images.
    - `on_click`: Handles mouse clicks to set start (Shift) or goal (Ctrl) positions.
    - `run_search`: Runs the A* search and draws the resulting path.
    - `draw_path`: Animates the solution path on the canvas.

#### Functions

- **heuristic(a, b)**
  - Calculates Manhattan distance between two positions.

- **a_star_search(grid_nodes, start_pos, goal_pos, canvas, cell_size, root)**
  - Implements the A* search algorithm.
  - Visualizes the search process on the canvas.

- **reconstruct_path(current_node)**
  - Builds the path from start to goal by following parent links.

## Usage

- Click "Generate New Maze" to create a new maze.
- Click "Start Search" to find the shortest path from rat to cheese.
- Click on a cell while holding **Shift** to set the rat's position.
- Click on a cell while holding **Ctrl** to set the cheese's position.

## License

See [LICENSE](LICENSE) for details.
