# LightlyTrain Tutorial

**LightlyTrain** is a powerful new framework for training computer vision models on **unlabeled data**.  
It allows you to get started with model training immediately using just a directory of images — no labels needed!

This repo provides guides on how to use **LightlyTrain** effectively.

## 🚀 What is LightlyTrain?

LightlyTrain is designed to make **self-supervised learning** easy and accessible. It wraps around powerful computer vision techniques and enables fast training with just a few lines of code.

- ✅ Works with raw image folders (no annotations required)
- ✅ Supports pretrained backbones like ResNet
- ✅ Easily fine-tune for classification later

## 📦 Installation

```
pip install lightly-train
```

📁 Example Dataset

You can use the official clothing dataset:

```
git clone https://github.com/lightly-ai/dataset_clothing_images.git my_data_dir
rm -rf my_data_dir/.git  
```

🧠 Training with LightlyTrain

```
import lightly_train

lightly_train.train(
    out="out/my_experiment",            # Output directory
    data="my_data_dir",                 # Directory with images
    model="torchvision/resnet18",       # Model to train
    epochs=10,                          # Number of epochs
    batch_size=32,                      # Batch size
)
```

📚 Resources

- [Lightly AI](https://docs.lightly.ai/train/stable/index.html)

