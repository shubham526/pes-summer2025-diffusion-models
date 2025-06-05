# Lab 6: Conditional Generation - From Random to Controllable

**Course:** Diffusion Models: Theory and Applications  
**Assignment Type:** Advanced Conditional Generation Lab  
**Duration:** 90 minutes  
**Team Size:** 2 students (same teams from Labs 1-5)  

## 📋 Assignment Overview

This capstone lab transforms your diffusion models from random generators into controllable creative tools. You'll implement the three fundamental approaches to conditional generation that power modern AI art: class-conditional diffusion, classifier guidance, and classifier-free guidance (CFG). By the end, you'll understand the techniques behind DALL-E, Stable Diffusion, and Midjourney.

### 🎯 Learning Objectives

By completing this lab, you will be able to:

1. **Implement** class-conditional diffusion models with embedding injection
2. **Build** classifier guidance systems for external steering during sampling
3. **Create** classifier-free guidance (CFG) for modern conditional generation
4. **Construct** U-Net modifications for conditioning at multiple scales
5. **Analyze** trade-offs between conditioning approaches in terms of quality, speed, and flexibility
6. **Deploy** conditional generation systems for practical creative applications

## 🚀 The Control Revolution

This lab addresses the fundamental limitation of generative models: **How do we control what gets generated?**

### The Control Problem:
- **Unconditional Generation:** Random samples from full distribution
- **User Intent:** "Generate a specific type of image"
- **Practical Applications:** Need precise control over output
- **Creative Tools:** Must respond to user instructions

### The Three Solutions:
1. **Class-Conditional:** Simple, direct conditioning via embeddings
2. **Classifier Guidance:** External steering using trained classifiers
3. **Classifier-Free Guidance:** Modern unified approach (powers Stable Diffusion)

## 🛠️ Prerequisites

Before starting this lab, ensure you have:

- **Complete Diffusion Mastery:**
  - Completion of Labs 1-5 (theory, ELBO, sampling)
  - Deep understanding of U-Net architectures
  - Experience with training and sampling algorithms
  - Knowledge of optimization and production considerations

- **Advanced Implementation Skills:**
  - Complex neural network modifications
  - Multi-objective training systems
  - Advanced sampling algorithm implementation
  - Performance analysis and trade-off evaluation

## 🧮 The Conditioning Architecture

This lab implements the complete spectrum of conditional generation:

### **Part 1: Understanding Unconditional Limitations (10 min)**
Experience why random generation fails for practical applications

### **Part 2: Class-Conditional Diffusion (25 min)**
Implement the simplest and most direct conditioning approach

### **Part 3: Classifier Guidance (25 min)**
Build external steering with trained classifiers and gradients

### **Part 4: Classifier-Free Guidance (20 min)**
Implement the modern standard that powers production systems

### **Part 5: Advanced Conditioning (10 min)**
Explore multi-scale and hierarchical conditioning techniques

## ✅ Core Implementation Tasks

You must complete these advanced conditioning components:

### **Unconditional Limitations Analysis**
```python
class UnconditionalLimitations:
    def demonstrate_random_generation_problem(self, n_samples):
        # TODO: Show why random sampling is inefficient for targeted generation
        
    def analyze_generation_efficiency(self, target_class, max_attempts):
        # TODO: Quantify the inefficiency of unconditional approaches
```

### **Class-Conditional Implementation**
```python
class ClassConditionalTrainer:
    def training_step(self, x_batch, class_batch):
        # TODO: L = E[||ε - ε_θ(x_t, y, t)||²] - conditional training objective
        
class ClassConditionalSampler:
    def ddim_step(self, x_t, class_labels, t, s):
        # TODO: Class-conditional DDIM sampling step
        
    def sample_class_conditional(self, class_labels, num_steps):
        # TODO: Complete conditional generation pipeline
```

### **Classifier Guidance System**
```python
class ClassifierGuidanceSampler:
    def compute_classifier_gradient(self, x_t, target_class, t):
        # TODO: ∇_x log p(y|x_t) for guidance
        
    def classifier_guided_step(self, x_t, target_class, t, s, guidance_scale):
        # TODO: ε̃ = ε_θ(x_t, t) - ω√(1-ᾱ_t) ∇_x log p(y|x_t)
        
    def sample_with_classifier_guidance(self, target_class, num_samples, guidance_scale):
        # TODO: Complete classifier-guided generation
```

### **Classifier-Free Guidance**
```python
class CFGTrainer:
    def cfg_training_step(self, x_batch, class_batch):
        # TODO: Joint conditional/unconditional training with dropout
        
class CFGSampler:
    def cfg_step(self, x_t, class_labels, t, s, guidance_scale):
        # TODO: ε̃ = (1+ω)ε_cond - ωε_uncond - CFG formula
        
    def sample_cfg(self, class_labels, num_steps, guidance_scale):
        # TODO: Complete CFG generation pipeline
```

## 📊 Expected Results

By the end of the lab, you should achieve:

- **Working class-conditional system** generating specific classes on demand
- **Functional classifier guidance** with adjustable steering strength
- **Complete CFG implementation** matching modern production systems
- **Comprehensive comparison** showing trade-offs between approaches
- **Understanding of practical deployment** considerations

## 🔬 Key Mathematical Insights

### **Class-Conditional Training Objective**
```
L = E_{x,y,t,ε} [||ε - ε_θ(x_t, y, t)||²]
```
Direct conditioning via class embeddings injected into the U-Net.

### **Classifier Guidance Formula**
```
ε̃ = ε_θ(x_t, t) - ω√(1-ᾱ_t) ∇_x log p(y|x_t)
```
External steering using classifier gradients during sampling.

### **Classifier-Free Guidance Formula**
```
ε̃ = (1 + ω)ε_cond - ωε_uncond
= ε_uncond + ω(ε_cond - ε_uncond)
```
Unified model trained to do both conditional and unconditional generation.

### **CFG Training Strategy**
```python
# Randomly drop conditioning during training
if torch.rand(1) < dropout_prob:
    class_labels = None  # Train unconditionally
predicted_noise = model(x_noisy, class_labels, t)
```

## 📝 Submission Requirements

### What to Submit

Submit your completed Jupyter notebook (.ipynb) containing:

#### **✅ Class-Conditional Implementation**
- [ ] Complete training objective with class embedding injection
- [ ] Modified U-Net architecture supporting class conditioning
- [ ] DDIM sampling adapted for conditional generation
- [ ] Demonstration of controlled class-specific generation

#### **✅ Classifier Guidance Implementation**
- [ ] Noise-aware classifier training on corrupted data
- [ ] Gradient computation for classifier steering
- [ ] Classifier-guided sampling with adjustable guidance scales
- [ ] Analysis of guidance strength effects on quality and diversity

#### **✅ Classifier-Free Guidance Implementation**
- [ ] Joint training system with conditioning dropout
- [ ] CFG sampling implementing the guidance formula
- [ ] Unified model handling both conditional and unconditional cases
- [ ] Comparison of CFG with other conditioning approaches

#### **✅ Comprehensive Analysis**
- [ ] Speed vs quality vs flexibility trade-off analysis
- [ ] Quantitative comparison between all three methods
- [ ] Understanding of when to use each approach
- [ ] Connection to real-world applications (DALL-E, Stable Diffusion)

#### **✅ Advanced Understanding**
- [ ] Analysis of unconditional generation limitations
- [ ] Exploration of hierarchical and multi-scale conditioning
- [ ] Production deployment considerations
- [ ] Future directions in conditional generation

### Evaluation Criteria

| Component | Points | Criteria |
|-----------|--------|----------|
| **Class-Conditional** | 25 | Correct embedding injection, functional training and sampling |
| **Classifier Guidance** | 30 | Proper gradient computation, working guidance system |
| **Classifier-Free Guidance** | 30 | Joint training implementation, CFG formula correct |
| **Analysis & Comparison** | 10 | Deep understanding of trade-offs and applications |
| **Code Quality** | 5 | Documentation, efficiency, production considerations |

**Total: 100 points**

## 🔍 Success Metrics

### Minimum Requirements
- [ ] Class-conditional generation produces recognizable target classes
- [ ] Classifier guidance shows controllable steering effects
- [ ] CFG implementation functionally equivalent to modern systems
- [ ] All three methods demonstrate clear conditional control

### Excellence Indicators
- [ ] Smooth interpolation between different guidance scales
- [ ] Comprehensive analysis of practical trade-offs
- [ ] Creative exploration of advanced conditioning techniques
- [ ] Production-ready considerations and optimizations

## 📖 Mathematical Reference

### Core Implementation Patterns

**Class Embedding Injection:**
```python
# Time embedding
t_embed = self.time_mlp(t / T)

# Class embedding
class_embed = self.class_embedding(class_labels)

# Combined conditioning
combined_embed = t_embed + class_embed
```

**Classifier Gradient Computation:**
```python
# Enable gradients for input
x_t.requires_grad_(True)

# Get classifier prediction
logits = classifier(x_t, t)
log_prob = F.log_softmax(logits, dim=-1)[range(len(x_t)), target_class]

# Compute gradient
grad = torch.autograd.grad(log_prob.sum(), x_t)[0]
```

**CFG Training with Dropout:**
```python
# Random conditioning dropout
mask = torch.rand(batch_size) < dropout_prob
class_labels[mask] = None  # or use null token

# Forward pass handles both cases
predicted_noise = model(x_noisy, class_labels, t)
```

**CFG Sampling Formula:**
```python
# Two forward passes
eps_cond = model(x_t, class_labels, t)
eps_uncond = model(x_t, None, t)

# Apply CFG formula
eps_guided = (1 + guidance_scale) * eps_cond - guidance_scale * eps_uncond
```

## 🕐 Timeline Suggestions

| Time | Activity |
|------|----------|
| 0-10 min | Team setup, unconditional limitations analysis |
| 10-35 min | Class-conditional implementation and testing |
| 35-60 min | Classifier guidance system implementation |
| 60-80 min | Classifier-free guidance implementation |
| 80-90 min | Comprehensive comparison and advanced techniques |

