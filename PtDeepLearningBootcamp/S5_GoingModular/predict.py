# YOUR CODE HERE

import argparse
import torch
import model_builder

from torchvision import transforms
from PIL import Image
from pathlib import Path

parser = argparse.ArgumentParser(prog="predict",
                                 description="Predict given image with trained TinyVGG model.")

parser.add_argument("--image", 
                    type=str,
                    help="Path of pizza_steak_sushi image to test model prediction.")

args = parser.parse_args()

img_to_predict = args.image
img_path = Path(img_to_predict)
img_to_predict = Image.open(img_to_predict)
data_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

img_transformed:torch.tensor = data_transform(img_to_predict)
img_transformed = img_transformed.unsqueeze(dim=0)

print("Loading model...")
model = model_builder.TinyVGG(
    input_shape=3,
    hidden_units=128,
    output_shape=3
)
model.load_state_dict(
    state_dict=torch.load(r"models\05_going_modular_script_mode_tinyvgg_model.pth")
    )

print("Performing prediction")
model.eval()
with torch.inference_mode():
    y_pred_logit = model(img_transformed)
    y_pred_prob = torch.softmax(y_pred_logit, dim=1)
    y_pred = torch.argmax(y_pred_prob, dim=1)

img_dict = {
    0:'pizza',
    1:'steak',
    2:'sushi'
}

print(f"Prediction: {img_dict[y_pred.item()]} | Actual: {str(img_path.parent.stem)} | Probability: {y_pred_prob.max():0.2f}")

