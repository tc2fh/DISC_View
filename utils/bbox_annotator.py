import os
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
import numpy as np
from PIL import Image
import pandas as pd


class MultiBBoxSelector:
    def __init__(self, folder_path):
        """
        Initialize the multi-image bounding box selector.

        :param folder_path: Path to the folder containing images
        """
        # List of supported image file extensions
        self.image_extensions = ['.png', '.jpg',
                                 '.jpeg', '.bmp', '.tif', '.tiff']

        # Get list of image files in the folder
        self.image_files = [
            f for f in os.listdir(folder_path)
            if os.path.isfile(os.path.join(folder_path, f))
            and os.path.splitext(f)[1].lower() in self.image_extensions
        ]

        # Sort the image files to ensure consistent order
        self.image_files.sort()

        # Store full paths to images
        self.image_paths = [os.path.join(folder_path, f)
                            for f in self.image_files]

        # Current image index
        self.current_index = 0

        # Stores bounding boxes for all images (DataFrame-compatible format)
        self.bbox_data = {
            'image_path': [],
            'x1': [],
            'y1': [],
            'x2': [],
            'y2': []
        }

        # Set up the plot
        self.fig, self.ax = plt.subplots()
        self.fig.suptitle(
            'Bounding Box Selector - Use Left Mouse Button to Select')

        # Add navigation and save buttons
        self.ax_prev = plt.axes([0.6, 0.02, 0.1, 0.04])
        self.ax_next = plt.axes([0.71, 0.02, 0.1, 0.04])
        self.ax_save = plt.axes([0.82, 0.02, 0.1, 0.04])

        self.btn_prev = plt.Button(self.ax_prev, 'Previous')
        self.btn_next = plt.Button(self.ax_next, 'Next')
        self.btn_save = plt.Button(self.ax_save, 'Save CSV')

        # Connect navigation and save buttons
        self.btn_prev.on_clicked(self.previous_image)
        self.btn_next.on_clicked(self.next_image)
        self.btn_save.on_clicked(self.save_bboxes)

        # Load first image
        self.load_image()

        # Create rectangle selector
        self.rect_selector = RectangleSelector(
            self.ax,
            self.onselect,
            useblit=True,
            button=[1],  # Left mouse button
            interactive=True
        )

        plt.show()

    def load_image(self):
        """Load the current image and clear previous annotations."""
        # Clear previous plot
        self.ax.clear()

        # Load current image
        current_image_path = self.image_paths[self.current_index]
        image = np.array(Image.open(current_image_path).resize((512, 512)))

        # Display image
        self.ax.imshow(image, cmap='gray')

        # Update title with current image name and index
        self.ax.set_title(
            f'Image {self.current_index + 1}/{len(self.image_files)}: {self.image_files[self.current_index]}')

        # Redraw existing bounding boxes for this image
        current_image_path = self.image_paths[self.current_index]
        matching_bbox_indices = [
            i for i, path in enumerate(self.bbox_data['image_path'])
            if path == current_image_path
        ]

        for idx in matching_bbox_indices:
            x1 = self.bbox_data['x1'][idx]
            y1 = self.bbox_data['y1'][idx]
            x2 = self.bbox_data['x2'][idx]
            y2 = self.bbox_data['y2'][idx]

            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1,
                                 fill=False, edgecolor='r', linewidth=2)
            self.ax.add_patch(rect)

        # Refresh the plot
        self.fig.canvas.draw()

    def onselect(self, eclick, erelease):
        """Capture bounding box coordinates."""
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)

        # Ensure coordinates are in correct order
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)

        # Get current image path
        current_image_path = self.image_paths[self.current_index]

        # Store bounding box
        self.bbox_data['image_path'].append(current_image_path)
        self.bbox_data['x1'].append(x1)
        self.bbox_data['y1'].append(y1)
        self.bbox_data['x2'].append(x2)
        self.bbox_data['y2'].append(y2)

        # Draw the bounding box
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1,
                             fill=False, edgecolor='r', linewidth=2)
        self.ax.add_patch(rect)

        # Refresh the plot
        self.fig.canvas.draw()

    def next_image(self, event):
        """Move to the next image."""
        if self.current_index < len(self.image_files) - 1:
            self.current_index += 1
            self.load_image()

    def previous_image(self, event):
        """Move to the previous image."""
        if self.current_index > 0:
            self.current_index -= 1
            self.load_image()

    def save_bboxes(self, event):
        """Save bounding boxes to a CSV file."""
        if not self.bbox_data['image_path']:
            print("No bounding boxes to save.")
            return

        # Create DataFrame
        df = pd.DataFrame(self.bbox_data)

        # Save to CSV
        save_path = 'bounding_boxes.csv' # update file path as needed
        df.to_csv(save_path, index=False)
        print(f"Bounding boxes saved to {save_path}")
        print(df)


# Example usage
if __name__ == "__main__":
    # Change this to the path of your image folder
    folder_path = "Subject-4"
    MultiBBoxSelector(folder_path)
