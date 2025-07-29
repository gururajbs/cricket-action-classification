from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import os
import sys
import cv2
import numpy as np

# Define paths
image_folder = "Cricket/test_model/"  
os.makedirs(image_folder, exist_ok=True)
video_path = "input.mp4"
output_dir = "output_videos"
project_dir = 'C:\\AIML\\Cricket Classify\\ultralytics' 
os.chdir(project_dir)

class_names = ["batting", "bowling", "fielding", "others"]

cwd = os.getcwd()
print('Guru - current dir:' + cwd)

# Function to make prediction with YOLO model 
def predict_with_yolo(img_path):
    results = model.predict(img_path, stream=True)    
    return results


def load_model(model_path):
    if not os.path.isfile(model_path):
        print(f"Model file '{model_path}' not found.")
        return None
    
    try:    
        loaded_model = YOLO(model_path, 'classify')
        return loaded_model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
    

def create_video_capture_object():
    capture = cv2.VideoCapture(video_path)
    # Check if video opened successfully
    if not capture.isOpened():
        print("Error opening video!")
        exit()

    return capture

def extract_frames_to_dir(capture_obj):
    # Frame count (optional, for tracking progress)
    frame_count = 0

    # Loop through video frames
    while True:
        # Capture frame-by-frame
        ret, frame = capture_obj.read()

        # Break loop if frame not captured
        if not ret:
            break

        # Extract frame number
        frame_num = capture_obj.get(cv2.CAP_PROP_POS_FRAMES) - 1

        # Create filename
        filename = f"{image_folder}{int(frame_num):05d}.jpg" # Use consistent padding (e.g., 05d for 5 digits)

        # Save frame
        cv2.imwrite(filename, frame)

        frame_count += 1
        print(f"Frame {frame_count} saved.")

    return frame_count


def stitch_videos_from_frames(capture_obj, yolo_model):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Define video writer dictionaries (one for each class)
    writers = {name: cv2.VideoWriter(f"{output_dir}/{name}.mp4", cv2.VideoWriter_fourcc(*"mp4v"),
                                   capture_obj.get(cv2.CAP_PROP_ＦPS), (int(capture_obj.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                                            int(capture_obj.get(cv2.CAP_PROP_FRAME_HEIGHT)))) 
                                                            for name in class_names}
    
    # Define color labels for bounding boxes (adjust colors as needed)
    colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (128, 128, 128)]
    
    # Loop through video frames
    while True:
        # Capture frame-by-frame
        ret, frame = capture_obj.read()
    
        # Break loop if frame not captured
        if not ret:
            break
    
        predictions = predict_with_yolo(frame)
        # break

        for i, r in enumerate(predictions):
            probs = r.probs
            top1 = probs.top1
            names = r.names

            writer = writers.get(names[top1])
            if writer is None:  # Skip if class doesn't have a writer
                continue
            else:
                writer.write(frame)        
    
    for writer in writers.values():
        if writer:  # Release only if the writer was created
            writer.release()
    
    print("Video processing complete!")



# Path to the yolov8-cls.yaml file
yaml_path = 'ultralytics/cfg/models/v8/yolov8-cls-cricket.yaml'
best_model_path = 'Cricket\\model\\runs\\classify\\train4\\weights\\best.pt'

# Load the model
model = load_model(best_model_path)
if not model:    
    print("No model loaded.")
    model = load_model(yaml_path)
    sys.exit(1)
    # model.train(data='Cricket/dataset', epochs=5, imgsz=640)


cap = create_video_capture_object()
# frames = extract_frames_to_dir(cap)
stitch_videos_from_frames(cap, model)

# Release resources
cap.release()
cv2.destroyAllWindows()

# print(f"Extracted {frames} frames to {image_folder}")

# Get image files
image_files = [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]  # Get only files

for image_file in image_files:
    image_path = os.path.join(image_folder, image_file)
    # Make prediction
    results = predict_with_yolo(image_path)

    # Visualize the results
    '''
    for i, r in enumerate(results):
        probs = r.probs
        top1 = probs.top1
        print("top1: " + str(top1))
        names = r.names 
        print("names: " + str(names))

        img = Image.open(image_path)
        draw = ImageDraw.Draw(img, "RGBA")

        txt = names[probs.top1]
        myFont = ImageFont.load_default(65)
        draw.text((10, 10), txt, font=myFont, fill =(255, 0, 0))
        img.save(f'Cricket/model/predictions/{names[probs.top1]}/{image_file}.jpg')
        # img.show()
    '''
    for result in results:
        print('Guru: ' + str(result))
        print(result.probs)
        names = result.names 

        if result.probs is not None:  
            probs = result.probs

            # Get the confidence scores from the tensor
            top5_confidences = probs.top5conf.cpu().numpy()  # Convert to NumPy array

            # Convert to percentages and format with two decimal places
            top5_percentages = top5_confidences * 100.0
            formatted_percentages = [f"{score:.2f}%" for score in top5_percentages]

            img = Image.open(image_path)
            draw = ImageDraw.Draw(img, "RGBA")
            myFont = ImageFont.load_default(15)
            # Print the top 5 scores in percentages
            # print("Top 5 scores (%):")
            for i, score in enumerate(formatted_percentages):
                print(f"Class {class_names[probs.top5[i]]}: {score}")                
                txt = f"{class_names[probs.top5[i]]}: {score}"
                draw.text((10, (10 + i * 25)), txt, font=myFont, fill =(255, 255, 255))                
        else:
            print("No detections found in this frame.")

        img.save(f'Cricket/model/predictions/{names[probs.top1]}/{image_file}.jpg')
    # break        

