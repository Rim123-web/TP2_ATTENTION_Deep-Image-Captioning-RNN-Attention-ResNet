# Image Captioning with Attention and ResNet

 

## Overview
This project implements an **image captioning system** that generates textual descriptions for images using a combination of **pre-trained ResNet50** as a feature extractor and a **custom LSTM with attention** mechanism.  

The model is trained and evaluated on the **Flickr30k dataset**, containing over 30,000 images with multiple textual captions.

---

## Project Goals
- Load and preprocess the **Flickr30k dataset** with proper image transformations.
- Build a vocabulary from captions including **special tokens** (`<pad>`, `<bos>`, `<eos>`, `<unk>`).
- Tokenize and detokenize captions for input and output handling.
- Apply **transfer learning** using a frozen pre-trained **ResNet50** to extract image features.
- Implement a **custom attention module** that guides the LSTM to focus on relevant regions of the image.
- Construct a **Caption LSTM model** that integrates attention and generates sequences of words.
- Train the model with **CrossEntropyLoss** and **Adam optimizer**, including **early stopping**.
- Evaluate the model by generating captions for validation images and comparing them to ground truth.

---

## Dataset
- **Flickr30k** from Hugging Face Datasets
- Images resized to `224x224`
- Train/validation split with **90/10 ratio**
- Captions cleaned and normalized (lowercasing, punctuation removal)
- Maximum caption length set to 20 tokens

---

## Model Components
- **ResNet50 (pre-trained)**: Extracts convolutional features from images. We freeze its parameters to avoid updating them during training.
- **Attention Module**: Computes a context vector as a weighted sum of image features, depending on the current hidden state of the LSTM.
- **LSTM Caption Generator**: Receives word embeddings and attention context to generate captions sequentially.
- **Word Embeddings**: Vocabulary embeddings to convert tokens to vectors and vice versa.

---

## Training
- Batch size: 32
- Learning rate: 5e-4
- Early stopping with patience of 3 epochs
- Best validation loss checkpoint saved automatically
- Extracted features used to speed up training

---

## Evaluation
- Captions generated for sample images from validation set
- Predicted captions compared visually and semantically with ground truth
- Attention maps can be inspected to verify which regions of the image the model focuses on

---

## Results
- Successfully learned to generate relevant captions for a variety of images
- Attention mechanism allows the model to focus on key objects in images
- Early stopping prevented overfitting and ensured stable convergence

---

## Author
**Rim Gourram**  
Master’s student in AI and Data Science  



