import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

def create_simple_corridor_map(width=500, height=300, wall_thickness=20):
    """Create a simple corridor with obstacles"""
    # Start with white background (free space)
    map_array = np.ones((height, width), dtype=np.uint8) * 255
    
    # Add outer walls (black borders)
    map_array[:wall_thickness, :] = 0  # Top wall
    map_array[-wall_thickness:, :] = 0  # Bottom wall
    map_array[:, :wall_thickness] = 0  # Left wall
    map_array[:, -wall_thickness:] = 0  # Right wall
    
    return map_array

def create_maze_map(width=500, height=500, wall_thickness=10):
    """Create a simple maze layout"""
    map_array = np.ones((height, width), dtype=np.uint8) * 255
    
    # Outer walls
    map_array[:wall_thickness, :] = 0
    map_array[-wall_thickness:, :] = 0
    map_array[:, :wall_thickness] = 0
    map_array[:, -wall_thickness:] = 0
    
    # Internal walls - create a simple maze pattern
    # Horizontal walls
    map_array[100:100+wall_thickness, 50:300] = 0
    map_array[200:200+wall_thickness, 200:450] = 0
    map_array[350:350+wall_thickness, 50:400] = 0
    
    # Vertical walls
    map_array[50:200, 250:250+wall_thickness] = 0
    map_array[250:400, 150:150+wall_thickness] = 0
    map_array[150:350, 350:350+wall_thickness] = 0
    
    return map_array

def create_room_map(width=600, height=400, wall_thickness=15):
    """Create a multi-room environment"""
    map_array = np.ones((height, width), dtype=np.uint8) * 255
    
    # Outer walls
    map_array[:wall_thickness, :] = 0
    map_array[-wall_thickness:, :] = 0
    map_array[:, :wall_thickness] = 0
    map_array[:, -wall_thickness:] = 0
    
    # Room dividers
    # Vertical divider with doorway
    map_array[:, 300:300+wall_thickness] = 0
    map_array[150:250, 300:300+wall_thickness] = 255  # Doorway
    
    # Horizontal divider with doorway
    map_array[200:200+wall_thickness, :300] = 0
    map_array[200:200+wall_thickness, 100:150] = 255  # Doorway
    
    # Add some obstacles in rooms
    map_array[50:100, 100:150] = 0  # Obstacle in room 1
    map_array[280:330, 450:500] = 0  # Obstacle in room 2
    
    return map_array

def create_circular_obstacles_map(width=500, height=500):
    """Create map with circular obstacles"""
    # Create PIL image for easier shape drawing
    img = Image.new('L', (width, height), 255)  # White background
    draw = ImageDraw.Draw(img)
    
    # Draw border
    draw.rectangle([0, 0, width-1, height-1], outline=0, width=10)
    
    # Draw circular obstacles
    circles = [
        (100, 100, 40),  # (center_x, center_y, radius)
        (200, 300, 35),
        (350, 150, 45),
        (400, 400, 30),
        (150, 350, 25),
    ]
    
    for x, y, r in circles:
        draw.ellipse([x-r, y-r, x+r, y+r], fill=0)
    
    return np.array(img)

def visualize_and_save_map(map_array, filename, show_plot=True):
    """Save map and optionally display it"""
    # Save as PNG
    img = Image.fromarray(map_array, mode='L')
    img.save(f"{filename}.png")
    
    # Create YAML file
    yaml_content = f"""image: {filename}.png
resolution: 0.04
origin: [-{map_array.shape[1]*0.04/2}, -{map_array.shape[0]*0.04/2}, 0.0]
negate: 0
occupied_thresh: 0.65
free_thresh: 0.196"""
    
    with open(f"{filename}.yaml", 'w') as f:
        f.write(yaml_content)
    
    if show_plot:
        plt.figure(figsize=(10, 8))
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
    # Generate different types of maps
    
    # Simple corridor
    corridor_map = create_simple_corridor_map(600, 200)
    visualize_and_save_map(corridor_map, "corridor_map", show_plot=False)
    
    # Maze layout
    maze_map = create_maze_map(500, 500)
    visualize_and_save_map(maze_map, "maze_map", show_plot=False)
    
    # Multi-room environment
    room_map = create_room_map(600, 400)
    visualize_and_save_map(room_map, "room_map", show_plot=False)
    
    # Circular obstacles
    circular_map = create_circular_obstacles_map(500, 500)
    visualize_and_save_map(circular_map, "circular_map", show_plot=False)
    
    print("All maps generated successfully!")
    print("\nTo use these maps:")
    print("1. Copy the .png and .yaml files to your world/ directory")
    print("2. Update world/world.yaml to reference your new map")
    print("3. Update starting positions in your Python code")