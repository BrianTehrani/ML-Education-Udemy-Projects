"""
Trains a PyTorch image classification model using device-agnostic code.
"""

import os
import torch
import argparse
import data_setup, engine, model_builder, utils

from torchvision import transforms

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Train TinyVGG model on pizza_steak_sushi img data imported from GitHub repo.")
  parser.add_argument("--num_epochs", 
                      type=int, 
                      default=5,
                      help="Number of training epochs.")
  parser.add_argument("--batch_size", 
                      type=int, 
                      default=32,
                      help="Batch size value for image data.")
  parser.add_argument("--hidden_units", 
                      type=int, 
                      default=10,
                      help="Number of model hidden units.")
  parser.add_argument("--learning_rate", 
                      type=float, 
                      default=0.0003,
                      help="Number of training epochs.")
  parser.add_argument("--train_dir", 
                      type=str, 
                      default="data/pizza_steak_sushi/train",
                      help="Training directory image data.")
  parser.add_argument("--test_dir", 
                      type=str, 
                      default="data/pizza_steak_sushi/test",
                      help="Testing directory image data.")

  args = parser.parse_args()

  # Setup hyperparameters
  NUM_EPOCHS = args.num_epochs
  BATCH_SIZE = args.batch_size
  HIDDEN_UNITS = args.hidden_units
  LEARNING_RATE = args.learning_rate

  # Setup directories
  train_dir = args.train_dir
  test_dir = args.test_dir

  # Setup target device
  device = "cuda" if torch.cuda.is_available() else "cpu"

  # Create transforms
  data_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
  ])

  print("Creating Data Loaders...")
  # Create DataLoaders with help from data_setup.py
  train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
      train_dir=train_dir,
      test_dir=test_dir,
      transform=data_transform,
      batch_size=BATCH_SIZE
  )

  print("Creating Model...")
  # Create model with help from model_builder.py
  model = model_builder.TinyVGG(
      input_shape=3,
      hidden_units=HIDDEN_UNITS,
      output_shape=len(class_names)
  ).to(device)

  # Set loss and optimizer
  loss_fn = torch.nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(),
                              lr=LEARNING_RATE)

  print("Training Model... ")
  # Start training with help from engine.py
  engine.train(model=model,
              train_dataloader=train_dataloader,
              test_dataloader=test_dataloader,
              loss_fn=loss_fn,
              optimizer=optimizer,
              epochs=NUM_EPOCHS,
              device=device)

  print("Saving Model... ")
  # Save the model with help from utils.py
  utils.save_model(model=model,
                  target_dir="models",
                  model_name="05_going_modular_script_mode_tinyvgg_model.pth")
