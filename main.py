# Sets an overall threshold for depth-based calculations
OVERALL_THRESHOLD = 4000

# Base used in an exponential decay function
DECAY_BASE = 8

# Standard Python libraries
import os
import numpy as np
# from scipy import ndimage  # Commented out: for image processing (unused here)
# from pydub import AudioSegment  # Commented out: for audio manipulation (unused here)
# from pydub.playback import play  # Commented out: for playing audio (unused here)
import pyrealsense2 as rs  # Library for RealSense camera
import cv2  # OpenCV library for image processing and display
import threading  # For running tasks in parallel threads
import pygame  # For handling audio playback
from gtts import gTTS  # Google Text-to-Speech
from PIL import Image  # Pillow for image manipulation
from yolov7 import YOLOv7  # YOLOv7 object detection library
from imread_from_url import imread_from_url  # Library to read images from URLs

# Hugging Face Transformers pipeline for image captioning
from transformers import pipeline as caption_pipeline

# Initialize the pipeline for image captioning using a specific model
image_to_text = caption_pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")

# Create a RealSense pipeline object
pipeline = rs.pipeline()

# Configure the RealSense pipeline streams
config = rs.config()
config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)  # Depth stream with specified resolution and frame rate
config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8,30)  # Color stream with specified resolution and frame rate

# List of class names for YOLOv7 object detection
class_name_list = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]

# Path to the YOLOv7 model file
model_path = "/home/jiang/ONNX-YOLOv7-Object-Detection/models/yolov7-tiny_640x640.onnx"

# Initialize YOLOv7 object detector with confidence and IoU thresholds
yolov7_detector = YOLOv7(model_path, conf_thres=0.2, iou_thres=0.3)

# Initialize Pygame for audio playback
pygame.init()

# Create a Pygame display surface (necessary for audio in some environments)
screen = pygame.display.set_mode((200,200))

# Initialize the Pygame mixer with multiple channels
pygame.mixer.init(channels=6)

# Path to the folder containing sound files
sound_folder = "/home/jiang/Documents/python_code/blindsensor/music3"

# Load sound effects for different positional cues
far_left_sound = pygame.mixer.Sound(os.path.join(sound_folder, "quincys_shaker_1M_far_left.wav"))
left_sound = pygame.mixer.Sound(os.path.join(sound_folder, "quincys_shaker_1M_left.wav"))
center_sound = pygame.mixer.Sound(os.path.join(sound_folder, "string_chords_1M_center.wav"))
far_right_sound = pygame.mixer.Sound(os.path.join(sound_folder, "bennys_drums_1M_far_right.wav"))
right_sound = pygame.mixer.Sound(os.path.join(sound_folder, "bennys_drums_1M_right.wav"))
center_down_sound = pygame.mixer.Sound(os.path.join(sound_folder, "Electric_bass_center_bottom.wav"))

# Assign each sound to its own mixer channel
far_left_channel = pygame.mixer.Channel(0)
left_channel = pygame.mixer.Channel(1)
center_channel = pygame.mixer.Channel(2)
far_right_channel = pygame.mixer.Channel(3)
right_channel = pygame.mixer.Channel(4)
center_down_channel = pygame.mixer.Channel(5)

# Play each channel's sound in a loop (loops=-1)
far_left_channel.play(far_left_sound, loops=-1)
left_channel.play(left_sound, loops=-1)
center_channel.play(center_sound, loops=-1)
far_right_channel.play(far_right_sound, loops=-1)
right_channel.play(right_sound, loops=-1)
center_down_channel.play(center_down_sound, loops=-1)

# Set the initial volume of each channel to 0 (silent)
far_left_channel.set_volume(0)
left_channel.set_volume(0)
center_channel.set_volume(0)
far_right_channel.set_volume(0)
right_channel.set_volume(0)
center_down_channel.set_volume(0)

# Interval for refreshing audio feedback
audio_refresh_rate = 0.3

# Global variable for depth data (if needed)
depth_data = 0

# Function to retrieve frame data from the RealSense pipeline
def get_frame_data():
    global pipeline, frame_data
    frame_data = pipeline.wait_for_frames()
    if not frame_data:
        return None
    else:
        return frame_data

# Function to calculate volume levels from depth data
def get_depth_from_image(depth_image, threshold=6000):
    #resampled_depth_image = ndimage.zoom(depth_image, (64 / depth_image.shape[0], 48 / depth_image.shape[1]), order=0)
    #resampled_depth_image = resampled_depth_image
    #print(depth_image.shape)

    # Downsample the depth image by taking every 16th pixel in both dimensions
    resampled_depth_image = depth_image[::16, ::16]

    # Divide the resampled image into various regions
    far_left = resampled_depth_image[:, :10]
    left = resampled_depth_image[:, 10:20]
    center = resampled_depth_image[:20, 20:33]
    right = resampled_depth_image[:, 33:43]
    far_right = resampled_depth_image[:, 43:53]
    center_down = resampled_depth_image[20:, 20:33]

    # Helper function to convert average depth in each region to a volume level
    factor = threshold
    def convert_to_volume(data):
        #data_mean = np.mean(data[np.isfinite(data)])
        mask = np.isfinite(data)
        finite_data = data[mask]
        data_mean = np.mean(finite_data)

        if finite_data.shape[0] == 0:
            data_mean = threshold

        if data_mean < threshold:
            data_mean = -np.mean(data)/factor + 1
            volume = (DECAY_BASE**data_mean - 1) / (DECAY_BASE - 1)
        else:
            volume = 0
        
        return volume

    # Compute the volume levels for each region
    far_left_mean = convert_to_volume(far_left)
    left_mean = convert_to_volume(left)
    center_mean = convert_to_volume(center)
    far_right_mean = convert_to_volume(far_right)
    right_mean = convert_to_volume(right)
    center_down_mean = convert_to_volume(center_down)
    
    return far_left_mean, left_mean, center_mean, far_right_mean, right_mean, center_down_mean

# Thread function to describe the scene using image captioning
def describe_scene_thread(image, pre_text = ""):
    print("start_ai")
    print(image)
    text_description = image_to_text(image)
    
    print(text_description)
    text_to_read = text_description[0]["generated_text"]
    print(text_to_read)
    myobj = gTTS(text=text_to_read, lang="en", slow=False)
    myobj.save("/home/jiang/scene_content.wav")
    os.system("mpg321 /home/jiang/scene_content.wav")

# Helper function to convert millimeters to feet and inches
def mm_to_ft_in(mm):
    # Constants for conversion
    mm_per_inch = 25.4
    inch_per_ft = 12

    # Convert mm to inches
    total_inches = mm / mm_per_inch

    # Split inches into feet and inches
    feet = total_inches // inch_per_ft
    inches = total_inches % inch_per_ft

    # Round to the nearest integer
    feet = round(feet)
    inches = round(inches)

    return feet, inches

# Function to perform object detection and return a descriptive text
def image_to_obj_detect_text(image, depth_data):
    boxes, scores, class_ids = yolov7_detector(image)
    all_class_name = ""
    i = 0
    for class_id in class_ids:
        class_name = class_name_list[class_id]

        box = boxes[i]
        box_center = [int((box[1]+box[3])/2), int((box[0]+box[2])/2)]
        box = [int(box[0]), int(box[1]), int(box[2]), int(box[3])]
        
        if box_center[1] < 283:  # If object's center is in the left region
            location_name = "left"
        elif box_center[1] > 566:  # If object's center is in the right region
            location_name ="right"
        else:
            location_name = "center"
        
        print(box)
        print(depth_data.shape)
        if box[2] <= depth_data.shape[1] and box[3] <= depth_data.shape[0]:
            depth_value = np.percentile(depth_data[box[1] : box[3], box[0]: box[2]], 20)
            feet_value, inch_value = mm_to_ft_in(depth_value)
            feet_text = str(feet_value)
            inch_text = str(inch_value)
        else:
            depth_text = "no "

        all_class_name = all_class_name + " a " + class_name + " is " + feet_text + " feet and " + inch_text + " inches to the " + location_name

        i = i + 1
    print(all_class_name)
    
    return all_class_name

# Thread function to detect objects and speak the results
def object_detection_thread(image, depth_data):
    print("start_obj_ai")
    print(image)
    text_description = image_to_obj_detect_text(image,depth_data)
    
    print(text_description)
    text_to_read = text_description
    print(text_to_read)
    myobj = gTTS(text=text_to_read, lang="en", slow=False)
    myobj.save("/home/jiang/scene_obj_content.wav")
    os.system("mpg321 /home/jiang/scene_obj_content.wav")

# Main function handling the RealSense data, audio, and events
def main(is_record = False, is_display_view = False):
    #global pipeline, frame_data

    # Start streaming from the pipeline
    profile = pipeline.start(config)

    output_data = []

    try:
        while True:
            frame_data = get_frame_data()
            # Extract depth frame
            depth_data = np.asanyarray(frame_data.get_depth_frame().get_data())

            # Calculate volume values for different regions of the depth image
            far_left_mean, left_mean, center_mean, far_right_mean, right_mean, center_down_mean = get_depth_from_image(depth_data, threshold = OVERALL_THRESHOLD)

            # Set the volume for each corresponding audio channel
            far_left_channel.set_volume(far_left_mean)
            left_channel.set_volume(left_mean)
            center_channel.set_volume(center_mean)
            far_right_channel.set_volume(far_right_mean)
            right_channel.set_volume(right_mean)
            center_down_channel.set_volume(center_down_mean)

            # Process Pygame events
            events = pygame.event.get()
            for event in events:
                # If UP arrow key is pressed
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        print("Page up pressed")
                        color_frame = frame_data.get_color_frame()
                        if not color_frame:
                            continue
                        else:
                            print("running caption")
                            rgb_data = np.asanyarray(color_frame.get_data())
                            rgb_data = Image.fromarray(np.uint8(rgb_data)).convert("RGB")
                            rgb_data.save("/home/jiang/image_to_caption.jpeg")
                            # Launch a thread to describe the scene
                            threading.Thread(target = describe_scene_thread, args = ("/home/jiang/image_to_caption.jpeg",)).start()

                # If DOWN arrow key is pressed
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_DOWN:
                        print("Page down pressed")
                        color_frame = frame_data.get_color_frame()
                        if not color_frame:
                            continue
                        else:
                            print("running caption")
                            rgb_data_np = np.asanyarray(color_frame.get_data())
                            rgb_data = Image.fromarray(np.uint8(rgb_data_np)).convert("RGB")
                            rgb_data.save("/home/jiang/image_to_object_detection.jpeg")
                            depth_data = np.asanyarray(frame_data.get_depth_frame().get_data())
                            # Launch a thread for object detection
                            threading.Thread(target = object_detection_thread, args = (rgb_data_np, depth_data)).start()

            # If display view is requested, show the depth image
            if is_display_view:
                depth_colormap = cv2.normalize(depth_data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                depth_colormap = cv2.applyColorMap(depth_colormap, cv2.COLORMAP_JET)
                cv2.imshow('Depth Image', depth_colormap)

                # Press 'q' to quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    if is_record and len(output_data) > 0:
                        np.save("data.npy", output_data)
                    break
    finally:
        # Stop streaming and close windows
        pipeline.stop()
        cv2.destroyAllWindows()

# Entry point of the script
if __name__ == "__main__":
    main()