# Lab 1: Hands-On Image Generation and Diffusion Fundamentals

**Course:** Diffusion Models: Theory and Applications  
**Assignment Type:** Hands-on Programming Lab  
**Duration:** 90 minutes  
**Team Size:** 2 students  


## 📋 Assignment Overview

In this foundational lab, you will build image generation systems from scratch, starting with a naive direct generator and progressing to a complete diffusion model. By the end of this lab, you'll have implemented the core components of diffusion models and understand why they work better than simpler approaches.

### 🎯 Learning Objectives

By completing this lab, you will be able to:

1. **Implement** a naive direct image generator and analyze its limitations
2. **Design and code** different noise addition schedules for forward diffusion
3. **Build** a U-Net denoising network with time embeddings from scratch
4. **Train** diffusion models using the noise prediction objective
5. **Generate** new images using reverse diffusion sampling
6. **Compare** different generation approaches quantitatively and qualitatively
7. **Form** your course mini-project team

## 🛠️ Prerequisites

Before starting this lab, ensure you have:

- **Programming Skills:**
  - Basic PyTorch knowledge (tensors, nn.Module, training loops)
  - Understanding of convolutional neural networks
  - Familiarity with image data representation

- **Mathematical Background:**
  - Basic probability and statistics
  - Understanding of neural network training
  - Familiarity with loss functions and optimization

- **Environment:**
  - Python 3.7+ with PyTorch, torchvision, matplotlib, numpy
  - GPU access recommended but not required
  - Jupyter notebook environment

## 📚 Assignment Structure

### Part 1: Team Formation & Setup (10 minutes)
- Form your 2-person mini-project team
- Set up the development environment
- Load and explore the MNIST dataset

### Part 2: Naive Direct Generator (25 minutes)
- Implement a neural network that maps noise directly to images
- Train using statistical matching approach
- Evaluate quality and identify limitations

### Part 3: Progressive Noise Addition (20 minutes)
- Build a noise scheduler with different schedules (linear, cosine, exponential)
- Implement the forward diffusion process
- Visualize how clean images become noise

### Part 4: U-Net Denoising Network (25 minutes)
- Construct a U-Net architecture with skip connections
- Add time embedding for timestep conditioning
- Train the model to predict noise from noisy images

### Part 5: Reverse Sampling (10 minutes)
- Implement the reverse diffusion sampling process
- Generate new images by iteratively denoising pure noise
- Compare all approaches side-by-side

## ✅ Core Implementation Tasks

You must complete these essential components:

### **Direct Generator (Part 2)**
```python
class DirectGenerator(nn.Module):
    # TODO: Implement architecture: latent_dim → 256 → 512 → 784 → reshape
    # TODO: Complete forward pass with proper reshaping
```

### **Noise Scheduler (Part 3)**
```python
class NoiseScheduler:
    # TODO: Implement get_beta_schedule() for linear schedule
    # TODO: Complete precompute_coefficients() for diffusion math
    # TODO: Implement add_noise() for forward diffusion
```

### **U-Net Denoiser (Part 4)**
```python
class SimpleDenoiser(nn.Module):
    # TODO: Implement U-Net encoder (downsampling path)
    # TODO: Add bottleneck layer
    # TODO: Implement decoder with skip connections
    # TODO: Complete forward pass with time embeddings
```

### **Training Functions**
```python
def train_direct_generator():
    # TODO: Implement statistical matching training

def train_denoiser():
    # TODO: Implement noise prediction training

def reverse_sampling():
    # TODO: Implement iterative denoising process
```

## 📊 Expected Results

By the end of the lab, you should achieve:

- **Working direct generator** producing blurry but recognizable MNIST digits
- **Complete noise scheduler** showing smooth forward diffusion process
- **Trained denoiser** accurately predicting noise at different timesteps
- **Functional reverse sampling** generating new digit images
- **Clear comparison** showing diffusion models outperform direct generation

## 📝 Submission Requirements

### What to Submit

Submit your completed Jupyter notebook (.ipynb) containing:

#### **✅ Code Implementations**
- [ ] Functional `DirectGenerator` with training loop
- [ ] Complete `NoiseScheduler` with linear schedule implementation
- [ ] Working `SimpleDenoiser` U-Net architecture
- [ ] Successful `reverse_sampling` function
- [ ] All training functions properly implemented

#### **✅ Generated Results**
- [ ] Sample images from trained direct generator
- [ ] Forward diffusion process visualization (clean → noise)
- [ ] Denoising progress images during training
- [ ] Final generated images from diffusion model
- [ ] Side-by-side comparison of all approaches

#### **✅ Analysis**
- [ ] Performance metrics comparing both generators
- [ ] Discussion of observed strengths and limitations
- [ ] Reflection on training challenges and insights
- [ ] Answers to discussion questions

#### **✅ Code Quality**
- [ ] Clean, well-commented implementations
- [ ] Proper error handling where appropriate
- [ ] Clear variable naming and organization
- [ ] All cells run without errors

### Evaluation Criteria

| Component | Points | Criteria |
|-----------|--------|----------|
| **Direct Generator** | 20 | Correct architecture, training, and evaluation |
| **Noise Scheduler** | 25 | Proper forward diffusion implementation |
| **U-Net Denoiser** | 30 | Complete architecture with time embeddings |
| **Reverse Sampling** | 20 | Functional image generation process |
| **Analysis & Discussion** | 5 | Thoughtful comparison and reflection |

**Total: 100 points**

## 🔍 Success Metrics

### Minimum Requirements
- [ ] All core functions run without errors
- [ ] Generated images are recognizable as digits
- [ ] Forward and reverse processes work correctly
- [ ] Basic performance comparison completed

### Excellence Indicators
- [ ] High-quality generated images with good diversity
- [ ] Insightful analysis of different approaches
- [ ] Additional noise schedules implemented (cosine, exponential)
- [ ] Creative visualizations or extensions

## 🚨 Common Pitfalls

**Architecture Issues:**
- Forgetting to reshape tensors properly in generator
- Missing skip connections in U-Net decoder
- Incorrect time embedding dimensions

**Training Problems:**
- Not normalizing MNIST to [-1, 1] range
- Using wrong loss functions for each approach
- Insufficient training epochs for convergence

**Sampling Issues:**
- Incorrect noise removal in reverse process
- Missing noise injection during sampling
- Wrong coefficient extraction from scheduler

## 📖 Resources

### Helpful References
- **DDPM Paper:** "Denoising Diffusion Probabilistic Models" (Ho et al., 2020)
- **U-Net Paper:** "U-Net: Convolutional Networks for Biomedical Image Segmentation"
- **PyTorch Documentation:** Official tutorials on neural networks
- **Course Materials:** Lecture slides on diffusion model fundamentals

### Getting Help
- **During Lab:** Ask instructors or TAs for clarification
- **Office Hours:** Scheduled help sessions for debugging
- **Discussion Forum:** Post questions and collaborate with classmates
- **Partner Collaboration:** Work together but ensure both understand all code

## ⏰ Timeline Suggestions

| Time | Activity |
|------|----------|
| 0-10 min | Team formation, environment setup, data exploration |
| 10-35 min | Implement and train direct generator |
| 35-55 min | Build noise scheduler and forward diffusion |
| 55-80 min | Construct U-Net and train denoiser |
| 80-90 min | Implement reverse sampling and final comparison |

---
