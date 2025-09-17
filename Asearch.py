import tkinter as tk
import heapq
import time
import os
import random
from PIL import Image, ImageTk

# Node class
class Node:
    def __init__(self, position):
        self.position = position
        self.g = float('inf')
        self.h = 0.0
        self.f = float('inf')
        self.parent = None

    def __lt__(self, other):
        return self.f < other.f

# Manhattan distance heuristic
def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def a_star_search(grid_nodes, start_pos, goal_pos, canvas, cell_size, root):
    if not start_pos or not goal_pos:
        return None

    start_node = grid_nodes.get(start_pos)
    goal_node = grid_nodes.get(goal_pos)
    
    if not start_node or not goal_node:
        return None

    start_node.g = 0
    start_node.f = heuristic(start_pos, goal_pos)

    open_set = [(start_node.f, start_node)]
    closed_set_positions = set()
    open_set_nodes = {start_node}

    while open_set:
        current_node_f, current_node = heapq.heappop(open_set)
        open_set_nodes.remove(current_node)

        if current_node.position != start_pos and current_node.position != goal_pos:
            r, c = current_node.position
            canvas.create_rectangle(c * cell_size, r * cell_size, 
                                    (c + 1) * cell_size, (r + 1) * cell_size, 
                                    fill="gray", outline="lightgray", tags="path_rects")
            root.update()
            time.sleep(0.01)
        
        if current_node.position == goal_pos:
            return reconstruct_path(current_node)

        closed_set_positions.add(current_node.position)

        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            neighbor_pos = (current_node.position[0] + dr, current_node.position[1] + dc)
            
            if neighbor_pos in grid_nodes and neighbor_pos not in closed_set_positions:
                neighbor_node = grid_nodes[neighbor_pos]
                tentative_g_score = current_node.g + 1

                if tentative_g_score < neighbor_node.g:
                    neighbor_node.parent = current_node
                    neighbor_node.g = tentative_g_score
                    neighbor_node.h = heuristic(neighbor_node.position, goal_pos)
                    neighbor_node.f = neighbor_node.g + neighbor_node.h

                    if neighbor_node not in open_set_nodes:
                        heapq.heappush(open_set, (neighbor_node.f, neighbor_node))
                        open_set_nodes.add(neighbor_node)

                        if neighbor_node.position != start_pos and neighbor_node.position != goal_pos:
                            r, c = neighbor_node.position
                            canvas.create_rectangle(c * cell_size, r * cell_size, 
                                                    (c + 1) * cell_size, (r + 1) * cell_size, 
                                                    fill="cyan", outline="lightblue", tags="path_rects")
                            root.update()
    
    return None

def reconstruct_path(current_node):
    path = []
    while current_node is not None:
        path.append(current_node.position)
        current_node = current_node.parent
    return path[::-1]

class MazeGUI:
    def __init__(self, root, rows, cols, cell_size):
        self.root = root
        self.rows = rows
        self.cols = cols
        self.cell_size = cell_size
        self.start_pos = None
        self.goal_pos = None
        
        self.canvas = tk.Canvas(root, width=cols * cell_size, height=rows * cell_size, bg="white")
        self.canvas.pack()
        
        button_frame = tk.Frame(root)
        button_frame.pack()
        
        self.generate_button = tk.Button(button_frame, text="Generate New Maze", command=self.generate_new_maze)
        self.generate_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        self.start_button = tk.Button(button_frame, text="Start Search", command=self.run_search)
        self.start_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        self.canvas.bind("<Button-1>", self.on_click)
        
        self.images = self.load_images()
        self.generate_new_maze()

    def generate_random_maze(self):
        # Initialize all cells as walls
        self.grid = [['#' for _ in range(self.cols)] for _ in range(self.rows)]

        def carve_passages_from(r, c):
            directions = [(0, 2), (0, -2), (2, 0), (-2, 0)]
            random.shuffle(directions)
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.rows and 0 <= nc < self.cols and self.grid[nr][nc] == '#':
                    self.grid[nr][nc] = '.'
                    self.grid[r + dr // 2][c + dc // 2] = '.'
                    carve_passages_from(nr, nc)

        # Start carving from a random odd cell
        start_r = random.randrange(1, self.rows, 2)
        start_c = random.randrange(1, self.cols, 2)
        self.grid[start_r][start_c] = '.'
        carve_passages_from(start_r, start_c)

        # Ensure start and goal are not walls
        if self.start_pos:
            self.grid[self.start_pos[0]][self.start_pos[1]] = '.'
        if self.goal_pos:
            self.grid[self.goal_pos[0]][self.goal_pos[1]] = '.'

    def generate_new_maze(self):
        self.generate_random_maze()
        # Find a valid open cell for the start position (rat)
        for r in range(self.rows):
            for c in range(self.cols):
                if self.grid[r][c] == '.':
                    self.start_pos = (r, c)
                    break
            if self.start_pos:
                break
        # Find a valid open cell for the goal position (cheese), far from start
        for r in reversed(range(self.rows)):
            for c in reversed(range(self.cols)):
                if self.grid[r][c] == '.' and (r, c) != self.start_pos:
                    self.goal_pos = (r, c)
                    break
            if self.goal_pos:
                break
        self.draw_grid()
        self.canvas.delete("path_rects")

    def load_images(self):
        try:
            images = {}
            images["rat"] = self.open_and_resize("rat.png")
            images["cheese"] = self.open_and_resize("cheese.png")
            images["wall"] = self.open_and_resize("wall.png")
            return images
        except Exception as e:
            print(f"Error loading images. Make sure 'rat.png', 'cheese.png', and 'wall.png' are in the same folder.")
            print(f"Error details: {e}")
            return {}

    def open_and_resize(self, filename):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        img_path = os.path.join(script_dir, filename)
        img = Image.open(img_path).resize((self.cell_size, self.cell_size), Image.Resampling.LANCZOS)
        return ImageTk.PhotoImage(img)

    def draw_grid(self):
        self.canvas.delete("all")
        for r in range(self.rows):
            for c in range(self.cols):
                self.canvas.create_rectangle(c * self.cell_size, r * self.cell_size, 
                                            (c + 1) * self.cell_size, (r + 1) * self.cell_size, 
                                            fill="white", outline="lightgray")
                
                if (r, c) == self.start_pos:
                    self.canvas.create_image(c * self.cell_size + self.cell_size // 2, 
                                             r * self.cell_size + self.cell_size // 2, 
                                             image=self.images.get("rat"))
                elif (r, c) == self.goal_pos:
                    self.canvas.create_image(c * self.cell_size + self.cell_size // 2, 
                                             r * self.cell_size + self.cell_size // 2, 
                                             image=self.images.get("cheese"))
                elif self.grid[r][c] == '#':
                    self.canvas.create_image(c * self.cell_size + self.cell_size // 2, 
                                             r * self.cell_size + self.cell_size // 2, 
                                             image=self.images.get("wall"))

    def on_click(self, event):
        self.canvas.delete("path_rects")
        self.draw_grid()
        col = event.x // self.cell_size
        row = event.y // self.cell_size
        
        if self.grid[row][col] == '#':
            print("Cannot place start or goal on a wall.")
            return

        if event.state & 0x1:  # Shift key
            self.start_pos = (row, col)
        elif event.state & 0x4: # Ctrl key
            self.goal_pos = (row, col)
        
        self.draw_grid()
    
    def run_search(self):
        self.canvas.delete("path_rects")
        self.draw_grid()
        
        grid_nodes = {}
        for r in range(self.rows):
            for c in range(self.cols):
                if self.grid[r][c] != '#':
                    grid_nodes[(r, c)] = Node((r, c))

        path = a_star_search(grid_nodes, self.start_pos, self.goal_pos, self.canvas, self.cell_size, self.root)
        
        if path:
            self.draw_path(path)
        else:
            print("No path found.")

    def draw_path(self, path):
        for r, c in path:
            if (r, c) != self.start_pos and (r, c) != self.goal_pos:
                self.canvas.create_rectangle(c * self.cell_size, r * self.cell_size, 
                                             (c + 1) * self.cell_size, (r + 1) * self.cell_size, 
                                             fill="yellow", outline="gold", tags="path_rects")
                self.root.update()
                time.sleep(0.05)

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Rat and Cheese Maze")
    
    rows, cols = 30, 40
    cell_size = 20
    
    app = MazeGUI(root, rows, cols, cell_size)
    root.mainloop()