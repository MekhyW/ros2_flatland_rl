import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

def create_simple_corridor_map(width=300, height=150, wall_thickness=10):
    """Create a simple corridor with obstacles - SMALLER VERSION"""
    # Start with white background (free space)
    map_array = np.ones((height, width), dtype=np.uint8) * 255
    
    # Add outer walls (black borders)
    map_array[:wall_thickness, :] = 0  # Top wall
    map_array[-wall_thickness:, :] = 0  # Bottom wall
    map_array[:, :wall_thickness] = 0  # Left wall
    map_array[:, -wall_thickness:] = 0  # Right wall
    
    return map_array

def create_maze_map(width=250, height=250, wall_thickness=8):
    """Create a simple maze layout - SMALLER VERSION"""
    map_array = np.ones((height, width), dtype=np.uint8) * 255
    
    # Outer walls
    map_array[:wall_thickness, :] = 0
    map_array[-wall_thickness:, :] = 0
    map_array[:, :wall_thickness] = 0
    map_array[:, -wall_thickness:] = 0
    
    # Internal walls - create a simple maze pattern (scaled down)
    # Horizontal walls
    map_array[50:50+wall_thickness, 25:150] = 0
    map_array[100:100+wall_thickness, 100:225] = 0
    map_array[175:175+wall_thickness, 25:200] = 0
    
    # Vertical walls
    map_array[25:100, 125:125+wall_thickness] = 0
    map_array[125:200, 75:75+wall_thickness] = 0
    map_array[75:175, 175:175+wall_thickness] = 0
    
    return map_array

def create_room_map(width=300, height=200, wall_thickness=10):
    """Create a multi-room environment - SMALLER VERSION"""
    map_array = np.ones((height, width), dtype=np.uint8) * 255
    
    # Outer walls
    map_array[:wall_thickness, :] = 0
    map_array[-wall_thickness:, :] = 0
    map_array[:, :wall_thickness] = 0
    map_array[:, -wall_thickness:] = 0
    
    # Room dividers
    # Vertical divider with doorway
    map_array[:, 150:150+wall_thickness] = 0
    map_array[75:125, 150:150+wall_thickness] = 255  # Doorway
    
    # Horizontal divider with doorway
    #map_array[100:100+wall_thickness, :150] = 0
    map_array[100:100+wall_thickness, 50:75] = 255  # Doorway
    
    # Add some obstacles in rooms (smaller)
    map_array[25:50, 50:75] = 0  # Obstacle in room 1
    map_array[140:165, 225:250] = 0  # Obstacle in room 2
    
    return map_array

def create_circular_obstacles_map(width=250, height=250):
    """Create map with circular obstacles - SMALLER VERSION"""
    # Create PIL image for easier shape drawing
    img = Image.new('L', (width, height), 255)  # White background
    draw = ImageDraw.Draw(img)
    
    # Draw border
    draw.rectangle([0, 0, width-1, height-1], outline=0, width=8)
    
    # Draw circular obstacles (smaller and fewer)
    circles = [
        (50, 50, 20),   # (center_x, center_y, radius)
        (100, 150, 18),
        (175, 75, 22),
        (200, 200, 15),
        (75, 175, 12),
    ]
    
    for x, y, r in circles:
        draw.ellipse([x-r, y-r, x+r, y+r], fill=0)
    
    return np.array(img)

def create_turn_corridor_map(width=200, height=200, wall_thickness=10):
    """Create a simple L-shaped turn corridor similar to turn.png"""
    map_array = np.ones((height, width), dtype=np.uint8) * 255
    
    # Create L-shaped corridor
    # Horizontal section (left side)
    map_array[:wall_thickness, :100] = 0  # Top wall
    map_array[50:, :100] = 0  # Fill bottom area
    map_array[40:50, :wall_thickness] = 0  # Left wall connection
    
    # Vertical section (right side)
    map_array[:100, 100:] = 0  # Fill right area
    map_array[:100, 100:100+wall_thickness] = 255  # Open corridor
    map_array[:wall_thickness, 100:] = 0  # Top wall extension
    map_array[:100, -wall_thickness:] = 0  # Right wall
    
    # Open areas for the corridor
    map_array[40:50, :100] = 255  # Horizontal corridor
    map_array[:100, 140:150] = 255  # Vertical corridor
    
    # Make the turn area
    map_array[40:50, 140:150] = 255  # Turn connection
    
    return map_array

def create_simple_obstacles_map(width=200, height=150, wall_thickness=8):
    """Create a simple map with a few rectangular obstacles"""
    map_array = np.ones((height, width), dtype=np.uint8) * 255
    
    # Outer walls
    map_array[:wall_thickness, :] = 0
    map_array[-wall_thickness:, :] = 0
    map_array[:, :wall_thickness] = 0
    map_array[:, -wall_thickness:] = 0
    
    # Add simple rectangular obstacles
    map_array[40:60, 50:70] = 0    # Obstacle 1
    map_array[90:110, 80:100] = 0  # Obstacle 2
    map_array[30:80, 130:140] = 0  # Obstacle 3 (vertical)
    
    return map_array

def visualize_and_save_map(map_array, filename, show_plot=True):
    """Save map and optionally display it"""
    # Save as PNG
    img = Image.fromarray(map_array, mode='L')
    img.save(f"{filename}.png")
    
    # Create YAML file with appropriate origin for smaller maps
    yaml_content = f"""image: {filename}.png
resolution: 0.04
origin: [-{map_array.shape[1]*0.04/2:.2f}, -{map_array.shape[0]*0.04/2:.2f}, 0.0]
negate: 0
occupied_thresh: 0.65
free_thresh: 0.196"""
    
    with open(f"{filename}.yaml", 'w') as f:
        f.write(yaml_content)
    
    if show_plot:
        plt.figure(figsize=(8, 6))
        plt.imshow(map_array, cmap='gray', origin='lower')
        plt.title(f"Generated Map: {filename}")
        plt.xlabel("X (pixels)")
        plt.ylabel("Y (pixels)")
        plt.colorbar(label="Occupancy (0=occupied, 255=free)")
        plt.show()
    
    print(f"Map saved as {filename}.png and {filename}.yaml")
    print(f"Map dimensions: {map_array.shape[1]} x {map_array.shape[0]} pixels")
    print(f"Real-world size: {map_array.shape[1]*0.04:.2f} x {map_array.shape[0]*0.04:.2f} meters")

# Example usage
if __name__ == "__main__":
    # Generate different types of maps - all smaller now
    
    # Simple corridor
    corridor_map = create_simple_corridor_map(300, 150)
    visualize_and_save_map(corridor_map, "corridor_map_small", show_plot=False)
    
    # Maze layout
    maze_map = create_maze_map(250, 250)
    visualize_and_save_map(maze_map, "maze_map_small", show_plot=False)
    
    # Multi-room environment
    room_map = create_room_map(300, 200)
    visualize_and_save_map(room_map, "room_map_small", show_plot=False)
    
    # Circular obstacles
    circular_map = create_circular_obstacles_map(250, 250)
    visualize_and_save_map(circular_map, "circular_map_small", show_plot=False)
    
    # Turn corridor (similar to turn.png)
    turn_map = create_turn_corridor_map(200, 200)
    visualize_and_save_map(turn_map, "turn_map_small", show_plot=False)
    
    # Simple obstacles map
    obstacles_map = create_simple_obstacles_map(200, 150)
    visualize_and_save_map(obstacles_map, "obstacles_map_small", show_plot=False)
    
    print("\nAll smaller maps generated successfully!")
    print("\nTo use these maps:")
    print("1. Copy the .png and .yaml files to your world/ directory")
    print("2. Update world/world.yaml to reference your new map")
    print("3. Update starting positions in world.yaml to fit the smaller maps")