# Cricket Action Classification using YOLOv8

A deep learning project that classifies cricket actions from images and videos using YOLOv8 classification models. This project can identify four main cricket actions: **batting**, **bowling**, **fielding**, and **others**.

## 🏏 Project Overview

This project uses YOLOv8 classification models to automatically categorize cricket scenes from images and videos. It's particularly useful for:
- Cricket video analysis
- Sports content categorization
- Automated highlight generation
- Cricket analytics applications

## 📁 Project Structure

```
Cricket/
├── dataset/                    # Training and validation datasets
│   ├── train/                 # Training images
│   │   ├── batting/          # Batting action images
│   │   ├── bowling/          # Bowling action images
│   │   ├── fielding/         # Fielding action images
│   │   └── others/           # Other cricket scenes
│   └── val/                  # Validation images (same structure)
├── model/                     # Trained models and predictions
│   ├── runs/                 # Training runs and model weights
│   └── predictions/          # Model prediction outputs
├── src/                      # Source code
│   └── cricket_classify.py   # Main classification script
├── test_model/               # Test images for evaluation
├── More Dataset/             # Additional dataset samples
├── cricket_classify.ipynb    # Jupyter notebook for training/testing
└── yolov8n.pt               # Pre-trained YOLOv8 model
```

## 🎯 Classification Categories

The model classifies cricket scenes into four categories:

1. **Batting** - Players batting, hitting shots, or in batting stance
2. **Bowling** - Bowlers in action, bowling deliveries
3. **Fielding** - Fielders catching, throwing, or fielding the ball
4. **Others** - General cricket scenes, celebrations, crowd shots, etc.

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- PyTorch 1.8+
- Ultralytics YOLOv8
- OpenCV
- PIL (Pillow)

### Installation

```bash
pip install ultralytics
pip install opencv-python
pip install pillow
```

### Training the Model

1. **Prepare your dataset** in the following structure:
   ```
   dataset/
   ├── train/
   │   ├── batting/
   │   ├── bowling/
   │   ├── fielding/
   │   └── others/
   └── val/
       ├── batting/
       ├── bowling/
       ├── fielding/
       └── others/
   ```

2. **Train the model** using the Jupyter notebook:
   ```python
   from ultralytics import YOLO
   
   # Load model
   model = YOLO('yolov8n.pt')
   
   # Train the model
   model.train(data='dataset', epochs=25, imgsz=640)
   ```

### Making Predictions

#### Image Classification
```python
from ultralytics import YOLO

# Load trained model
model = YOLO('runs/classify/train/weights/best.pt')

# Predict on an image
results = model.predict('path/to/image.jpg')
```

#### Video Processing
Use the `cricket_classify.py` script to process videos:

```python
# The script will:
# 1. Extract frames from input video
# 2. Classify each frame
# 3. Create separate videos for each category
# 4. Save results in output_videos/
```

## 📊 Dataset Statistics

- **Training Images**: ~1,108 images across 4 categories
- **Validation Images**: ~1,121 images across 4 categories
- **Total Dataset**: ~2,229 images

### Category Distribution:
- **Batting**: ~227 training, ~172 validation
- **Bowling**: ~182 training, ~172 validation  
- **Fielding**: ~222 training, ~170 validation
- **Others**: ~477 training, ~407 validation

## 🔧 Usage Examples

### Basic Image Classification
```python
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont

# Load model
model = YOLO('runs/classify/train/weights/best.pt')

# Predict
img_path = 'test_image.jpg'
results = model.predict(img_path, stream=True)

# Process results
for r in results:
    probs = r.probs
    top1 = probs.top1
    names = r.names
    predicted_class = names[top1]
    print(f"Predicted: {predicted_class}")
```

### Video Frame Classification
```python
# The cricket_classify.py script provides:
# - Frame extraction from videos
# - Real-time classification
# - Separate video output for each category
# - Visual annotations on frames
```

## 📈 Model Performance

The trained model achieves good accuracy across all cricket action categories. Performance metrics are available in the training logs within `model/runs/`.

## 🎥 Video Output

The system generates separate video files for each category:
- `output_videos/batting.mp4`
- `output_videos/bowling.mp4` 
- `output_videos/fielding.mp4`
- `output_videos/others.mp4`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add your improvements
4. Submit a pull request

## 📝 License

This project is part of the Ultralytics YOLOv8 ecosystem. Please refer to the main repository license.

## 🙏 Acknowledgments

- Built with [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- Cricket dataset collected and curated for sports analytics
- Community contributions and feedback

---

**Note**: This project is designed for cricket action classification and can be extended to other sports or action recognition tasks by modifying the dataset and training parameters. 