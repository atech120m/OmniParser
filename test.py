import torch
from PIL import Image, ImageDraw
import base64
import io
from utils import check_ocr_box, get_yolo_model, get_caption_model_processor, get_som_labeled_img

# Load models and configurations
yolo_model = get_yolo_model(model_path='weights/icon_detect/best.pt')
caption_model_processor = get_caption_model_processor(model_name="florence2", model_name_or_path="weights/icon_caption_florence")

# Bounding box configuration
draw_bbox_config = {
    'text_scale': 0.8,
    'text_thickness': 2,
    'text_padding': 2,
    'thickness': 2,
}

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import torch
from PIL import Image, ImageDraw
import base64
import io
from utils import check_ocr_box, get_yolo_model, get_caption_model_processor, get_som_labeled_img

# Load models and configurations
yolo_model = get_yolo_model(model_path='weights/icon_detect/best.pt')
caption_model_processor = get_caption_model_processor(model_name="florence2", model_name_or_path="weights/icon_caption_florence")

# Bounding box configuration
draw_bbox_config = {
    'text_scale': 0.8,
    'text_thickness': 2,
    'text_padding': 2,
    'thickness': 2,
}

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import torch
from PIL import Image, ImageDraw, ImageFont
import base64
import io
from utils import check_ocr_box, get_yolo_model, get_caption_model_processor, get_som_labeled_img

# Define color mapping for different element types
ELEMENT_COLOR_MAPPING = {
    "Text Box": "blue",
    "Icon Box": "green",
    "Button": "orange",
    # Add more mappings as needed
}

# Fallback color if the element type is not in the mapping
DEFAULT_COLOR = "red"

# Load models and configurations
yolo_model = get_yolo_model(model_path='weights/icon_detect/best.pt')
caption_model_processor = get_caption_model_processor(model_name="florence2", model_name_or_path="weights/icon_caption_florence")

# Bounding box configuration
draw_bbox_config = {
    'text_scale': 0.8,
    'text_thickness': 2,
    'text_padding': 2,
    'thickness': 2,
}

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def process_image(image_path, box_threshold=0.05, iou_threshold=0.1, 
                 output_text_path="output.txt", output_image_path="output_with_boxes.png"):
    """
    Processes an image, saves parsed screen elements with pixel coordinates to a text file,
    and saves an annotated image with bounding boxes.
    
    Parameters:
    - image_path: str, path to the input image.
    - box_threshold: float, threshold for removing bounding boxes with low confidence.
    - iou_threshold: float, threshold for removing overlapping bounding boxes.
    - output_text_path: str, path to save the parsed content.
    - output_image_path: str, path to save the annotated image with bounding boxes.
    """
    
    # Load the image in PIL format
    image_input = Image.open(image_path).convert('RGB')
    image_save_path = 'processed_image.png'
    image_input.save(image_save_path)

    # Perform OCR to detect text boxes
    ocr_bbox_rslt, is_goal_filtered = check_ocr_box(
        image_save_path, 
        display_img=False, 
        output_bb_format='xyxy', 
        goal_filtering=None, 
        easyocr_args={'paragraph': False, 'text_threshold': 0.9}
    )
    
    text, ocr_bbox = ocr_bbox_rslt

    # Perform YOLO detection and get labeled image
    dino_labeled_img, label_coordinates, parsed_content_list = get_som_labeled_img(
        image_save_path, 
        yolo_model, 
        BOX_TRESHOLD=box_threshold, 
        output_coord_in_ratio=False,  # Set to False to get pixel coordinates
        ocr_bbox=ocr_bbox, 
        draw_bbox_config=draw_bbox_config, 
        caption_model_processor=caption_model_processor, 
        ocr_text=text, 
        iou_threshold=iou_threshold
    )

    # Combine parsed content list into a single string with coordinates
    output_content = []
    
    # Ensure that the number of parsed contents matches the number of coordinates
    if len(parsed_content_list) != len(label_coordinates):
        print("Warning: The number of parsed content entries does not match the number of detected coordinates.")
    
    for content, coords in zip(parsed_content_list, label_coordinates):
        if len(coords) == 4:
            x_min, y_min, w, h = coords
            x_max, y_max = x_min + w, y_min + h

            # Extract element type from content
            # Assuming the content starts with "Text Box", "Icon Box", etc.
            element_type = "Unknown"
            for key in ELEMENT_COLOR_MAPPING.keys():
                if content.startswith(key):
                    element_type = key
                    break

            # Assign color based on element type
            color = ELEMENT_COLOR_MAPPING.get(element_type, DEFAULT_COLOR)

            coordinate_info = f"{content} - Coordinates: (x_min: {x_min}, y_min: {y_min}, x_max: {x_max}, y_max: {y_max})"
            print(coordinate_info)  # Print coordinates to console
            output_content.append(coordinate_info)
        else:
            print(f"Invalid coordinates for content '{content}': {coords}")
            output_content.append(f"{content} - Coordinates: Invalid")

    # Save output with coordinates to a text file
    with open(output_text_path, 'w') as f:
        f.write("\n".join(output_content))
    print(f"Parsed content with coordinates saved to {output_text_path}.")

    # Draw bounding boxes on the image
    annotated_image = image_input.copy()
    draw = ImageDraw.Draw(annotated_image)

    # Optionally, specify a font. If not, default font will be used.
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except IOError:
        font = ImageFont.load_default()

    for content, coords in zip(parsed_content_list, label_coordinates):
        if len(coords) == 4:
            x_min, y_min, w, h = coords
            x_max, y_max = x_min + w, y_min + h

            # Extract element type from content
            element_type = "Unknown"
            for key in ELEMENT_COLOR_MAPPING.keys():
                if content.startswith(key):
                    element_type = key
                    break

            # Assign color based on element type
            color = ELEMENT_COLOR_MAPPING.get(element_type, DEFAULT_COLOR)

            # Draw rectangle with the assigned color
            draw.rectangle([x_min, y_min, x_max, y_max], outline=color, width=draw_bbox_config['thickness'])

            # Draw text label above the bounding box
            text_position = (x_min, y_min - 15)  # Adjust as needed
            draw.text(text_position, content, fill=color, font=font)
        else:
            print(f"Skipping drawing for content '{content}' due to invalid coordinates.")

    # Save the annotated image
    annotated_image.save(output_image_path)
    print(f"Annotated image saved to {output_image_path}.")

# Example usage
if __name__ == "__main__":
    process_image('example.png', box_threshold=0.05, iou_threshold=0.1, 
                 output_text_path='parsed_content.txt', output_image_path='annotated_image.png')

# Example usage
if __name__ == "__main__":
    process_image('example.png', box_threshold=0.05, iou_threshold=0.1, 
                 output_text_path='parsed_content.txt', output_image_path='annotated_image.png')

# Example usage
if __name__ == "__main__":
    process_image('example.png', box_threshold=0.05, iou_threshold=0.1, 
                 output_text_path='parsed_content.txt', output_image_path='annotated_image.png')
