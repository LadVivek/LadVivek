import os
import torch
import cv2
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from sklearn.metrics import accuracy_score

# Load a pre-trained YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.eval()

# Define a custom dataset
class CraterDataset(Dataset):
    def __init__(self, images_dir, transform=None):
        self.images_dir = images_dir
        self.image_paths = [os.path.join(images_dir, fname) 
    for fname in os.listdir(images_dir)]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        return image, img_path

# Define paths to your datasets
test_dir ='C:\\Users\\Vivek\\Downloads\\finaltry'
output_dir = 'C:\\WebDevlopment\\New folder (2)'  # Output directory
csv_file = 'C:\\WebDevlopment\\csv.csv'  # CSV file path

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Define image parameters
img_height, img_width = 640, 640  # Set dimensions to be multiples of 32
batch_size = 10  # Use batch size 1 for processing one image at a time

# Data augmentation and preprocessing
transform = transforms.Compose([ 
    transforms.Resize((img_height, img_width)),  # Ensure dimensions are compatible with YOLO
    # transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.1,contrast=0.1,saturation=0.1,hue=0.05),
    transforms.ToTensor()
])

# Load datasets
test_dataset = CraterDataset(test_dir, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


def calculate_precision_recall(pred_boxes, gt_boxes, iou_threshold=0.5):
        true_positives = 0
        false_positives = 0
        false_negatives = len(gt_boxes)

        for pred_box in pred_boxes:
            iou_max = 0
            best_gt_idx = -1
        for gt_idx, gt_box in enumerate(gt_boxes):
            iou = calculate_iou(pred_box, gt_box)
            if iou > iou_max:
                iou_max = iou
                best_gt_idx = gt_idx

            if iou_max >= iou_threshold:
                true_positives += 1
                false_negatives -= 1
            else:
                false_positives += 1

            return true_positives, false_positives, false_negatives

# mAP Calculation
def calculate_mAP(pred_boxes_list, gt_boxes_list, iou_threshold=0.5):
    total_true_positives = 0
    total_false_positives = 0
    total_false_negatives = 0

    for pred_boxes, gt_boxes in zip(pred_boxes_list, gt_boxes_list):
        tp, fp, fn = calculate_precision_recall(pred_boxes, gt_boxes, iou_threshold)
        total_true_positives += tp
        total_false_positives += fp
        total_false_negatives += fn

    precision = total_true_positives / (total_true_positives + total_false_positives) if (total_true_positives + total_false_positives) > 0 else 0
    recall = total_true_positives / (total_true_positives + total_false_negatives) if (total_true_positives + total_false_negatives) > 0 else 0

    return precision, recall
df_losses = pd.DataFrame(columns=["Image", "Loss"])

confidence_threshold=0.3

for images, img_paths in test_loader:
    with torch.no_grad():
        predictions = model(images)

    for i, (image, prediction, img_path) in enumerate(zip(images, predictions, img_paths)):
        image = image.permute(1, 2, 0).numpy()  # Convert tensor to numpy array for OpenCV
        image = (image * 255).astype(np.uint8)  # Convert to uint8 for display
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenCV

        # Extract bounding boxes, confidence scores, and class labels from the tensor output
        boxes = prediction[:, :4].numpy()  # Coordinates of bounding boxes
        scores = prediction[:, 4].numpy()  # Confidence scores
        # labels = prediction[:, 5].numpy()  # Class labels
        labels = prediction[:, 5].numpy()

loss=np.mean((scores-1)**2)

        # Append loss to DataFrame
new_row = pd.DataFrame({"Image": [os.path.basename(img_path)], "Loss": [loss]})
df_losses = pd.concat([df_losses.dropna(axis=1, how='all'), new_row], ignore_index=True)

        # Draw bounding boxes
for box, score, label in zip(boxes, scores, labels):
            if score >= 0.1:  # Threshold for displaying bounding boxes
                x1, y1, x2, y2 = box
                x1,y1,x2,y2 = map(int, [x1,y1,x2,y2])
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(image.shape[1], x2)
                y2 = min(image.shape[0], y2)

                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, f'{score:.2f}', (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Save the processed image
output_path = os.path.join(output_dir, os.path.basename(img_path))
cv2.imwrite(output_path, image)

# Save the loss data to a CSV file
df_losses.to_csv(csv_file, index=False)


# Function to evaluate accuracy using IoU
def calculate_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1_gt, y1_gt, x2_gt, y2_gt = box2

    # Calculate intersection coordinates
    xi1 = max(x1, x1_gt)
    yi1 = max(y1, y1_gt)
    xi2 = min(x2, x2_gt)
    yi2 = min(y2, y2_gt)
    
    inter_width = max(xi2 - xi1, 0)
    inter_height = max(yi2 - yi1, 0)
    
    # Area of the intersection rectangle
    inter_area = inter_width * inter_height
    
    # Areas of the original bounding boxes
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_gt - x1_gt) * (y2_gt - y1_gt)
    
    # Union area
    union_area = box1_area + box2_area - inter_area
    
    # Calculate IoU
    iou = inter_area / union_area if union_area > 0 else 0
    return iou

def evaluate_model_accuracy(pred_boxes, gt_boxes, iou_threshold=0.5):
    matches = 0
    total_gt_boxes = len(gt_boxes_list)
    total_pred_boxes = len(pred_boxes_list)
    
    for pred_box in pred_boxes:
        for gt_box in gt_boxes:
            iou = calculate_iou(pred_box, gt_box)
            if iou >= iou_threshold:
                matches += 1
                break  # Once a match is found, move to the next predicted box

    accuracy = matches / total_gt_boxes if total_gt_boxes > 0 else 0
    return accuracy

# Sample usage during prediction
# gt_boxes_list = [
#     [[100, 100, 200, 200], [300, 300, 400, 400]],  # Ground truth boxes for image 1
#     [[150, 150, 250, 250]]                        # Ground truth boxes for image 2
# ]




# pred_boxes_list = [
gt_boxes_list = [
    [[100, 100, 200, 200], [300, 300, 400, 400]],  # Ground truth boxes for image 1
    [[150, 150, 250, 250]]                        # Ground truth boxes for image 2
]
pred_boxes_list = [
    [[105, 105, 195, 195], [295, 295, 405, 405]],  # Predicted boxes for image 1
    [[150, 160, 260, 260]]                        # Predicted boxes for image 2
]
#     [[105, 105, 195, 195], [295, 295, 405, 405]],  # Predicted boxes for image 1
#     [[150, 160, 260, 260]]                        # Predicted boxes for image 2
# ]


accuracies = []
for pred_boxes, gt_boxes in zip(pred_boxes_list, gt_boxes_list):
    accuracy = evaluate_model_accuracy(pred_boxes, gt_boxes)
    accuracies.append(accuracy)

precision, recall = calculate_mAP(pred_boxes_list, gt_boxes_list, iou_threshold=0.5)
print(f"Precision: {precision*100:.2f}%, Recall: {recall*100:.2f}%")
mAP =(precision + recall) /2
print(f"mAP: {mAP*100:.2f}%")

mean_accuracy = np.mean(accuracies)
print(f"Mean Accuracy: {mean_accuracy * 100:.2f}%")


import tkinter as tk
from tkinter import ttk
from PIL import ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Define paths to the processed images and the CSV file
processed_dir = 'C:\\WebDevlopment\\New folder (2)'
csv_file = 'C:\\WebDevlopment\\csv.csv'

# Non-Maximum Suppression function
def non_maximum_suppression(boxes, confidences, overlap_thresh=0.45):
    if len(boxes) == 0:
        return []

    boxes = np.array(boxes)
    confidences = np.array(confidences)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(confidences)[::-1]

    pick = []

    while len(idxs) > 0:
        i = idxs[0]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[1:]])
        yy1 = np.maximum(y1[i], y1[idxs[1:]])
        xx2 = np.minimum(x2[i], x2[idxs[1:]])
        yy2 = np.minimum(y2[i], y2[idxs[1:]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = (w * h) / area[idxs[1:]]

        idxs = idxs[np.where(overlap <= overlap_thresh)[0] + 1]

    return pick

class CraterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Crater Detection")
        
        # Set the background color of the main window
        self.root.configure(bg='lightblue')

        # Main Frame to hold image and plot
        self.main_frame = ttk.Frame(root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Frame for image
        self.image_frame = ttk.Frame(self.main_frame)
        self.image_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Canvas for image display
        self.canvas_width = 1000
        self.canvas_height = 1000
        self.canvas = tk.Canvas(self.image_frame, width=self.canvas_width, height=self.canvas_height, bg='lightblue')
        self.canvas.pack(side=tk.TOP, fill='both', expand=True)

        # Frame for plot
        self.plot_frame = ttk.Frame(self.main_frame)
        self.plot_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        # Setup initial plot
        self.fig, self.ax = plt.subplots(figsize=(10, 7))
        self.bar_plot = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.bar_plot.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Customize plot appearance
        self.ax.set_facecolor('lightblue')  # Set plot background to light blue
        self.fig.patch.set_facecolor('lightblue')  # Set figure background to light blue

        self.current_image_idx = 0
        self.images = []
        self.crater_losses = []
        self.displayed_indices = []

        self.load_images_and_losses()

        # Debug: Print out how many images were loaded
        print(f"Loaded {len(self.images)} images.")

        if self.images:
            self.show_image(self.current_image_idx)
        else:
            print("No images to display.")

    def load_images_and_losses(self):
        self.images = []
        self.crater_losses = []

        # Load the CSV file with loss data
        if not os.path.isfile(csv_file):
            print(f"Error: CSV file '{csv_file}' not found.")
            return

        df_losses = pd.read_csv(csv_file)

        for fname in os.listdir(processed_dir):
            img_path = os.path.join(processed_dir, fname)
            
            if not os.path.isfile(img_path):
                print(f"Warning: Image file '{img_path}' not found.")
                continue
            
            image = cv2.imread(img_path)
            if image is None:
                print(f"Error: Unable to read image file '{img_path}'.")
                continue

            # Check if the image exists in the loss CSV
            if fname in df_losses['Image'].values:
                loss = df_losses.loc[df_losses['Image'] == fname, 'Loss'].values[0]
            else:
                loss = None
                print(f"Warning: Loss value for {fname} not found.")

            # Simulate detection results (replace with actual detection logic)
            boxes = [[100, 100, 200, 200], [110, 110, 210, 210]]  # Example bounding boxes
            confidences = [0.9, 0.8]  # Example confidence scores

            # Apply NMS
            indices = non_maximum_suppression(boxes, confidences)
            filtered_boxes = [boxes[i] for i in indices]
            filtered_confidences = [confidences[i] for i in indices]

            self.images.append((image, loss, filtered_boxes, filtered_confidences))
            if loss is not None:
                self.crater_losses.append(loss)
            else:
                self.crater_losses.append(0)  # Append 0 if no loss value is found

        # Debug: Print out loaded losses
        print("Loaded losses:", self.crater_losses)

    def show_image(self, idx):
        if idx >= len(self.images):
            print(f"Error: Image index {idx} out of bounds.")
            return

        image, loss, boxes, confidences = self.images[idx]

        if image is None:
            print(f"Error: Image data is empty for index {idx}.")
            return

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        
        # Draw bounding boxes from NMS on the image
        for (box, confidence) in zip(boxes, confidences):
            x1, y1, x2, y2 = box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box
            cv2.putText(image, f'{confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # Resize image to fit within the canvas
        image = Image.fromarray(image)
        image = self.resize_image(image, self.canvas_width, self.canvas_height)
        
        # Create a light blue background image
        background = Image.new('RGB', (self.canvas_width, self.canvas_height), 'lightblue')
        
        # Paste the image on top of the background
        background.paste(image, (0, 0))
        
        photo = ImageTk.PhotoImage(image=background)

        self.canvas.delete("all")  # Clear previous image
        self.canvas.create_image(self.canvas_width // 2, self.canvas_height // 2, anchor=tk.CENTER, image=photo)
        self.canvas.image = photo  # Keep reference to avoid garbage collection

        # Update plot
        self.update_plot(idx)

        # Schedule the next image update
        self.root.after(1000, self.next_image)

    def resize_image(self, image, max_width, max_height):
        # Resize image to fit within max_width and max_height while maintaining aspect ratio
        original_width, original_height = image.size

        width_ratio = max_width / original_width
        height_ratio = max_height / original_height
        scale_ratio = min(width_ratio, height_ratio)

        new_width = int(original_width * scale_ratio)
        new_height = int(original_height * scale_ratio)

        return image.resize((new_width, new_height), Image.LANCZOS)

    def update_plot(self, idx):
        if idx not in self.displayed_indices:
            self.displayed_indices.append(idx)

        # Update bar plot with all displayed images
        self.ax.clear()
        indices = self.displayed_indices
        losses = [self.crater_losses[i] for i in indices]

        self.ax.bar(indices, losses, color='red')  # Set bar color to red
        self.ax.set_xlabel("Image Index")
        self.ax.set_ylabel("Loss")
        self.ax.set_title("Crater Detection Loss vs. Image Index")

        # Redraw the plot
        self.bar_plot.draw()

    def next_image(self):
        
        
        # Debug: Print the current index and image count
     class CraterApp:
        def __init__(self, root):
            self.root = root
        self.root.title("Crater Detection")
        
        # Set the background color of the main window
        self.root.configure(bg='lightblue')

        # Main Frame to hold image and plot
        self.main_frame = ttk.Frame(root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Frame for image
        self.image_frame = ttk.Frame(self.main_frame)
        self.image_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Canvas for image display
        self.canvas_width = 1000
        self.canvas_height = 1000
        self.canvas = tk.Canvas(self.image_frame, width=self.canvas_width, height=self.canvas_height, bg='lightblue')
        self.canvas.pack(side=tk.TOP, fill='both', expand=True)

        # Frame for plot
        self.plot_frame = ttk.Frame(self.main_frame)
        self.plot_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        # Setup initial plot
        self.fig, self.ax = plt.subplots(figsize=(10, 7))
        self.bar_plot = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.bar_plot.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Customize plot appearance
        self.ax.set_facecolor('lightblue')  # Set plot background to light blue
        self.fig.patch.set_facecolor('lightblue')  # Set figure background to light blue

        self.current_image_idx = 0
        self.images = []
        self.crater_losses = []
        self.displayed_indices = []

        self.load_images_and_losses()

        # Debug: Print out how many images were loaded
        print(f"Loaded {len(self.images)} images.")

        if self.images:
            self.show_image(self.current_image_idx)
        else:
            print("No images to display.")

    def load_images_and_losses(self):
        self.images = []
        self.crater_losses = []

        # Load the CSV file with loss data
        if not os.path.isfile(csv_file):
            print(f"Error: CSV file '{csv_file}' not found.")
            return

        df_losses = pd.read_csv(csv_file)

        for fname in os.listdir(processed_dir):
            img_path = os.path.join(processed_dir, fname)
            
            if not os.path.isfile(img_path):
                # print(f"Warning: Image file '{img_path}' not found.")
                continue
            
            image = cv2.imread(img_path)
            if image is None:
                # print(f"Error: Unable to read image file '{img_path}'.")
                continue

            # Check if the image exists in the loss CSV
            if fname in df_losses['Image'].values:
                loss = df_losses.loc[df_losses['Image'] == fname, 'Loss'].values[0]
            else:
                loss = None
                print(f"Warning: Loss value for {fname} not found.")

            # Simulate detection results (replace with actual detection logic)
            boxes = [[100, 100, 200, 200], [110, 110, 210, 210]]  # Example bounding boxes
            confidences = [0.9, 0.8]  # Example confidence scores

            # Apply NMS
            indices = non_maximum_suppression(boxes, confidences)
            filtered_boxes = [boxes[i] for i in indices]
            filtered_confidences = [confidences[i] for i in indices]

            self.images.append((image, loss, filtered_boxes, filtered_confidences))
            if loss is not None:
                self.crater_losses.append(loss)
            else:
                self.crater_losses.append(0)  # Append 0 if no loss value is found

        # Debug: Print out loaded losses
        print("Loaded losses:", self.crater_losses)

    def show_image(self, idx):
        if idx >= len(self.images):
            print(f"Error: Image index {idx} out of bounds.")
            return

        image, loss, boxes, confidences = self.images[idx]

        if image is None:
            print(f"Error: Image data is empty for index {idx}.")
            return

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        
        # Draw bounding boxes from NMS on the image
        for (box, confidence) in zip(boxes, confidences):
            x1, y1, x2, y2 = box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box
            cv2.putText(image, f'{confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # Resize image to fit within the canvas
        image = Image.fromarray(image)
        image = self.resize_image(image, self.canvas_width, self.canvas_height)
        
        # Create a light blue background image
        background = Image.new('RGB', (self.canvas_width, self.canvas_height), 'lightblue')
        
        # Paste the image on top of the background
        background.paste(image, (0, 0))
        
        photo = ImageTk.PhotoImage(image=background)

        self.canvas.delete("all")  # Clear previous image
        self.canvas.create_image(self.canvas_width // 2, self.canvas_height // 2, anchor=tk.CENTER, image=photo)
        self.canvas.image = photo  # Keep reference to avoid garbage collection

        # Update plot
        self.update_plot(idx)

        # Schedule the next image update
        self.root.after(1000, self.next_image)

    def resize_image(self, image, max_width, max_height):
        # Resize image to fit within max_width and max_height while maintaining aspect ratio
        original_width, original_height = image.size

        width_ratio = max_width / original_width
        height_ratio = max_height / original_height
        scale_ratio = min(width_ratio, height_ratio)

        new_width = int(original_width * scale_ratio)
        new_height = int(original_height * scale_ratio)

        return image.resize((new_width, new_height), Image.LANCZOS)

    def update_plot(self, idx):
        if idx not in self.displayed_indices:
            self.displayed_indices.append(idx)

        # Update bar plot with all displayed images
        self.ax.clear()
        indices = self.displayed_indices
        losses = [self.crater_losses[i] for i in indices]

        self.ax.bar(indices, losses, color='red')  # Set bar color to red
        self.ax.set_xlabel("Image Index")
        self.ax.set_ylabel("Loss")
        self.ax.set_title("Crater Detection Loss vs. Image Index")

        # Redraw the plot
        self.bar_plot.draw()

    def next_image(self):
        # Print the current index and image count
        print(f"Displaying image {self.current_image_idx + 1}/{len(self.images)}")
        self.current_image_idx += 1

        # self.current_image_idx = (self.current_image_idx + 1) % len(self.images)
        if self.current_image_idx < len(self.images):
            self.show_image(self.current_image_idx)

if __name__ == "__main__":
    root = tk.Tk()
    app = CraterApp(root)
    root.mainloop()
