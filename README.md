# Skylark


Aerial GCP Detection – Computer Vision Assignment

1. Overview

The goal of this task is to automatically detect Ground Control Points (GCPs) from aerial drone imagery.
For each input image containing a GCP marker, the model must predict:

1. The pixel coordinates (x, y) of the marker center.
2. The shape of the marker, classified as one of:
   - Cross
   - Square

This problem is formulated as a multi-task learning problem combining coordinate regression and shape classification.

2. Approach

Since each image contains exactly one GCP marker, the problem can be simplified from object detection to direct regression + classification.

Instead of using a heavy detection model such as YOLO, we use a ResNet-18 based multi-task neural network (YOLO would also be suitable but took lot of training time)

Normal CNN was not used as it may cause slower convergence and training instability and difficult for multi-class learning


Architecture:

Input Image
↓
ResNet18 Backbone (feature extraction)
↓
Shared Feature Representation
↓
Coordinate Head → (x,y) center
Classification Head → Cross / Square

Backbone:
A pretrained ResNet-18 model is used as the backbone to extract high-level visual features.

Reasons for choosing ResNet:
- Strong feature extraction capability
- Works well with small datasets using transfer learning
- Computationally efficient


The final fully connected layer of ResNet is removed and replaced with two task-specific heads.

3. Training Strategy

Image Preprocessing:
Original images: 2048 × 1365 pixels
Resized to: 128 × 128 pixels for efficient training.

Coordinate Normalization:

x_normalized = x / image_width
y_normalized = y / image_height

This ensures stable gradients and scale invariance.

Loss Functions

Localization Loss:
Mean Squared Error (MSE)

L_coord = MSE(predicted_xy, ground_truth_xy)

Classification Loss:
Cross Entropy Loss

L_class = CrossEntropy(predicted_class, ground_truth_class)

Total Loss:

L_total = L_coord + L_class

Optimization:
Optimizer: Adam
Learning Rate: 1e-3
Epochs: 1

Since pretrained weights are used, only a small number of epochs are required.

4. Dataset Handling

Dataset Structure:

train_dataset/
test_dataset/

Training annotations are stored in:

curated_gcp_marks.json

Each entry contains:

{
  "image_path": {
      "mark": {
          "x": value,
          "y": value
      },
      "verified_shape": "Cross"
  }
}

The test dataset contains only images without labels.

Nested directories are handled using recursive traversal with os.walk().

5. Challenges and Mitigation

Nested Directory Structure:
Images are stored inside multiple nested folders.
Solution: recursive directory traversal using os.walk().

Missing Shape Labels:
Some entries may not contain verified_shape.
Solution:
shape_name = data.get("verified_shape", "Cross")

Large Image Size:
Original images are large and slow to process.
Solution: resize images to 128×128.

Marker Size Variability:
Markers are small relative to the aerial image.
Deep CNN features help capture global context and marker patterns.

6. Evaluation Strategy

Since test labels are unavailable, evaluation was performed on the training set.

Metrics:
- Pixel localization error
- Shape classification accuracy

Visual verification was also performed by plotting predicted marker centers.

7. Running the Inference Pipeline

Run aerial_gcp_cnn

The script produces predictions.json

Example:

{
 "folder/subfolder/image.JPG": {
   "mark": {
     "x": 2461.37,
     "y": 956.81
   },
   "verified_shape": "Cross"
 }
}

8. Summary

This solution formulates GCP detection as a multi-task regression and classification problem using a pretrained ResNet backbone.

The approach is:
- computationally efficient
- suitable for small datasets
- simple to deploy
- capable of predicting both marker location and shape.

