# 🛰️ Ground-to-Satellite Image Matching via Synthetic Satellite Generation and Joint Feature Learning

This repository provides an end-to-end pipeline for matching ground-level images to their corresponding satellite views. 
It combines synthetic satellite image generation, semantic segmentation, and a joint feature learning architecture for robust cross-view image matching.

---

## 📌 Overview

Matching ground-level photos to satellite imagery is challenging due to drastic perspective and scale differences. 
This project tackles this by generating synthetic aerial views from ground photos, segmenting them, and learning a joint representation using a contrastive approach.

---

## 🧭 Pipeline

The full pipeline consists of the following stages:

1. **Synthetic Satellite Generation**  
   - 🔍 Ground image ➝ VisualCLIP encoder (from Openai, pretrained)
   - 🌀 VisualCLIP features ➝ DiT (from Meta, pretrained on ImageNet) ➝ Synthetic Aerial Image
     
       Reference: https://arxiv.org/pdf/2212.09748
   
   Normally a DiT receives word tokens as inputs. In this case, since the input needed to be an image, the embedding layer had to be changed to a VisualCLIP.
   VisualCLIP was frozen and did not undergo any finetuning. Similarly the VAEncoder which is part of DiT was frozen.
   Both the encoders were of course pretrained. VisualClip's training is openai/clip-vit-base-patch32 which is a training on publicly available image-caption data (https://huggingface.co/openai/clip-vit-base-       patch32).
   The latter was a default choice embedding by Meta team.
   The DiT model chosen to tackle this task was a DiT-XL/2 which is a rather big model but not the biggest of its family. It was chosen since the task is quite complex.
   Here it is possible to find more informations https://github.com/facebookresearch/DiT?tab=readme-ov-file.

3. **Semantic Segmentation**  
   - 🗺️ Synthetic aerial ➝ SegFormer-B3 (pretrained on ADE20K) ➝ Segmentation map
     
     Reference: https://arxiv.org/pdf/2105.15203

     Comparing different models of the same family led to the choice of the b3 model. An intermediate model which is not too small to be capable of capturing relevant features,
     nor too big to make it difficult to train. The model was pretrained on ADE20K which is a great dataset for this task since it comprehends urban and aerial views.

---
          Ground Image
              (G)
               |
               v
         +-------------+
         |  VisualCLIP |
         |   Encoder   |
         +-------------+
               |
      Visual Feature Vector
               |
               v
         +-------------+
         |     DiT     |  ← Pretrained on ImageNet (Meta)
         +-------------+
               |
    Synthetic Aerial Image (A)
               |
               v
       +------------------+
       |  SegFormer B3    |  ← Pretrained on ADE20K
       | Semantic Segment |
       +------------------+
               |
               v
     Segmented Aerial Image (S)
     
---


4. **Joint Feature Matching**  
   - Input:  
     - Ground image (G)  
     - Synthetic aerial (A)  
     - Segmented aerial (S)  
     - Candidate satellite image (C)  

   - Process:  
     - Feature extraction via VGG16  
     - A and C share weights  
     - Fusion of features into two embeddings:  
       - `embedding_G` = FFN(G + A + S)  
       - `embedding_C` = FFN(A + C)  
     - Final match score computed via **triplet loss**

---

## 🧠 Architecture: JointFeatureLearningNet
                +--------+      +----------+
      G ------> | VGG_G  |      |          |
                +--------+      |          |
                                 |          |
                +--------+      |          |
      A ------> | VGG_A  | ---> |          |
                +--------+      |          |
                                 | FFN_G    | ---> embedding_G
                +--------+      |          |
      S ------> | VGG_S  |      |          |
                +--------+      |          |
                                 |          |
                                 +----------+

                +--------+      +----------+
      C ------> | VGG_C  | ---> |          |
                +--------+      |          |
                                 | FFN_C    | ---> embedding_C
                +--------+      |          |
      A ------> | VGG_A  | ---> |          |
                +--------+      +----------+

                 🔗 Triplet Loss(embedding_G, embedding_C)

- `VGG_G`, `VGG_S`, `VGG_A`, and `VGG_C` are VGG16 backbones.
- VGG_A and VGG_C **share weights**.
- FeatureFusionNets (FFN) project feature concatenations into a shared embedding space.
- Matching is supervised using **triplet loss**.

---
