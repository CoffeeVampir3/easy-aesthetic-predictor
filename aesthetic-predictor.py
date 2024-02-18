import gradio as gr
import numpy as np
import torch
import os
from PIL import Image
from torchvision import transforms
from huggingface_hub import hf_hub_download

def download_example_model():
    # Define the repository and filename
    repository_id = "Blackroot/Anime-Aesthetic-Predictor-Medium"
    filename = "trained_aesthetic_scorer_12_epochs_v1.pth"
    target_dir = "models/"
    
    os.makedirs(target_dir, exist_ok=True)

    # Download the file
    model_path = hf_hub_download(repo_id=repository_id, filename=filename, local_dir=target_dir, local_dir_use_symlinks=False)

MODEL = None
def load_first_model_from_folder(folder_path, bad_thing_happened=False):
    """
    Iterates over the files in the given folder and loads the first .pth file as a PyTorch model.
    
    Args:
    - folder_path (str): Path to the folder containing the .pth files.

    Returns:
    - torch.nn.Module: The loaded PyTorch model, or None if no .pth file is found.
    """
    
    if not os.path.exists(folder_path):
        download_example_model()
    
    for filename in os.listdir(folder_path):
        if filename.endswith('.pth'):
            model_path = os.path.join(folder_path, filename)
            print(f"Loading model from {model_path}")
            model = torch.load(model_path).to('cuda')
            model.eval()
            return model
    
    download_example_model()
    if not MODEL and bad_thing_happened:
        print("Couldn't download model and there's none in the model directory. Get one from: https://huggingface.co/Blackroot/Anime-Aesthetic-Predictor-Medium")
        exit()
    return load_first_model_from_folder(folder_path, True)

def predict_image(img_path):
    global MODEL
    if not MODEL:
        MODEL = load_first_model_from_folder("models")
        
    if not MODEL:
        print("There's no model in the models folder.")
        exit()
    
    image = Image.open(img_path).convert('RGB')
    # Define the transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize the image to 224x224 pixels
        transforms.ToTensor(),  # Convert the image to a tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize
    ])

    image_tensor = transform(image)
    
    # Unsqueeze to add a batch dimension
    image_tensor = image_tensor.unsqueeze(0).to("cuda")
    
    # Perform inference
    with torch.no_grad():
        output = MODEL(image_tensor)

    return round(output.item())

# Define the Gradio interface
interface = gr.Interface(fn=predict_image,
                         inputs=gr.Image(type="filepath"),
                         outputs=gr.Textbox(label="Score"),
                         title="Image Score Predictor",
                         allow_flagging="never")

# Run the Gradio app
interface.launch()