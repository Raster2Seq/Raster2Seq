from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2


def plot_image_grid_matplotlib(images, rows, cols, titles=None, figsize=(12, 8)):
    """
    Plot multiple images in a grid using Matplotlib.
    
    Parameters:
    -----------
    images : list
        List of image arrays or paths to image files
    rows : int
        Number of rows in the grid
    cols : int
        Number of columns in the grid
    titles : list, optional
        List of titles for each image
    figsize : tuple, optional
        Figure size (width, height) in inches
    """
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    
    # Make sure axes is always a 2D array even if rows or cols is 1
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, cols)
    elif cols == 1:
        axes = axes.reshape(rows, 1)
    
    # Flatten the image and title lists to make them easier to iterate over
    flat_axes = axes.flatten()
    
    for i, ax in enumerate(flat_axes):
        if i < len(images):
            # Check if image is a file path or array
            if isinstance(images[i], str):
                img = plt.imread(images[i])
            else:
                img = images[i]
            
            # Display the image
            ax.imshow(img)
            
            # Add title if provided
            if titles and i < len(titles):
                ax.set_title(titles[i])
            
            # Remove axis ticks
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            # Hide unused subplots
            ax.axis('off')
    
    plt.tight_layout()
    return fig


def resize_floorplan(input_path, output_path, new_width=None, new_height=None, 
                     method=Image.BICUBIC, maintain_aspect=False, dpi=300):
    """
    Resize a floorplan image while preserving quality and readability.
    
    Parameters:
    -----------
    input_path : str
        Path to the input floorplan image
    output_path : str
        Path where the resized image will be saved
    new_width : int, optional
        Target width in pixels
    new_height : int, optional
        Target height in pixels
    method : PIL.Image resampling filter, optional
        Resampling method (default is BICUBIC)
    maintain_aspect : bool, optional
        Whether to maintain the aspect ratio (default is True)
    dpi : int, optional
        DPI to save the image with (default is 300)
    
    Returns:
    --------
    PIL.Image
        The resized image object
    """
    # Open the image
    img = Image.open(input_path)
    
    # Get original dimensions
    width, height = img.size
    
    # Calculate new dimensions if maintaining aspect ratio
    if maintain_aspect:
        if new_width and not new_height:
            new_height = int(height * (new_width / width))
        elif new_height and not new_width:
            new_width = int(width * (new_height / height))
        elif new_width and new_height:
            # Use the smallest scale factor to ensure the image fits
            width_ratio = new_width / width
            height_ratio = new_height / height
            if width_ratio < height_ratio:
                new_height = int(height * width_ratio)
            else:
                new_width = int(width * height_ratio)
    
    # Resize the image
    resized_img = img.resize((new_width, new_height), method)
    
    # Save with high DPI for print quality
    resized_img.save(output_path, dpi=(dpi, dpi))
    
    return resized_img

# Example usage:
if __name__ == "__main__":
    input_path = "/home/htp26/RoomFormerTest/data/coco_cubicasa5k/test/08104.jpg"
    # Using BICUBIC (good general-purpose resampling)
    resize_floorplan(input_path, "output_bicubic.png", new_width=256, new_height=256, method=Image.BICUBIC)
    
    # Using LANCZOS (high-quality downsampling)
    resize_floorplan(input_path, "output_lanczos.png", new_width=256, new_height=256, method=Image.LANCZOS)
    
    # Using NEAREST (preserves hard edges, good for pixel art or very simple plans)
    resize_floorplan(input_path, "output_nearest.png", new_width=256, new_height=256, method=Image.NEAREST)
    
    # Using BOX (similar to NEAREST, but with some smoothing)
    resize_floorplan(input_path, "output_box.png", new_width=256, new_height=256, method=Image.BOX)

    # Using BILINEAR (Faster than BICUBIC and LANCZOS but lower quality)
    resize_floorplan(input_path, "output_bilinear.png", new_width=256, new_height=256, method=Image.BILINEAR)

    titles = ['BICUBIC', 'LANCZOS', 'NEAREST', 'BOX', 'BILINEAR']
    fig = plot_image_grid_matplotlib(["output_bicubic.png", "output_lanczos.png", "output_nearest.png", "output_box.png", "output_bilinear.png"], 1, 5, titles)
    fig.savefig("resize_comp.png")

# Comparison of different methods for specific floorplan types
def compare_resize_methods(input_path, output_prefix, target_width):
    """Generate resized versions using different methods for comparison"""
    methods = {
        "nearest": Image.NEAREST,
        "box": Image.BOX,
        "bilinear": Image.BILINEAR,
        "bicubic": Image.BICUBIC,
        "lanczos": Image.LANCZOS,
    }
    
    results = {}
    for name, method in methods.items():
        output_path = f"{output_prefix}_{name}.png"
        img = resize_floorplan(input_path, output_path, new_width=target_width, method=method)
        results[name] = output_path
    
    return results