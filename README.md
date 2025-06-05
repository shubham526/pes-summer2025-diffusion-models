# Diffusion Models: Theory and Applications
**PES University, Department of Computer Science and Engineering**
**Summer 2025 (5-Day Intensive Summer Course)**

Welcome to the GitHub repository for the **Diffusion Models: Theory and Applications** summer course! This repository contains all lecture slides, lab materials, and other relevant resources for the course.

---
## Course Overview
This intensive course provides a comprehensive introduction to diffusion probabilistic models, from theoretical foundations to practical implementations. Students will learn the mathematics of forward and reverse diffusion processes, implementation techniques, and how to apply these powerful generative models to various applications.

---
## Instructor
**Dr. Shubham Chatterjee**
Department of Computer Science
Missouri University of Science and Technology, USA

---
## Prerequisites
To succeed in this course, students should have:
* A strong foundation in machine learning and deep learning concepts.
* Proficiency in Python programming and PyTorch.
* A basic understanding of probabilistic models and stochastic processes.
* Familiarity with computer vision concepts.

---
## Learning Objectives
By the end of this course, students will be able to:
1.  Explain the mathematical foundations of diffusion models.
2.  Implement basic diffusion processes for generative modeling.
3.  Train and optimize diffusion models for various applications.
4.  Apply conditional generation techniques to guide the generation process.
5.  Develop efficient sampling strategies for diffusion models.
6.  Create a working implementation for a real-world application.

---
## Course Structure
The course is structured as a 5-day intensive program:
* **Days 1-4:** Focused on teaching theoretical concepts and hands-on implementation sessions.
* **Day 5:** Reserved for mini-project presentations and evaluations.

---
## Topics Covered 📚
### Day 1: Foundations & Forward Diffusion Process
* **Lecture 1: The Challenge of Generating Reality**: Course overview, Generative AI capabilities, defining "realistic," the universality of the generation challenge, statistical perspective ($p(x)$), the Manifold Hypothesis, and an overview of Generative Model Families (GANs, VAEs). Introduction to Diffusion Models and their advantages. Mini-Project announcement. [![View Slides](https://img.shields.io/badge/View-PDF-red)](https://github.com/<username>/<repo>/raw/main/slides/Lecture1.pdf)

* **Lecture 2: The Forward Diffusion Process - Learning to Destroy Data Systematically** [▶️ View Slides](Lectures/Day1_Lec2_ForwardDiffusion.pdf): Recap, core insight of fixing the forward process, Markov Chains, mathematical formulation of forward process ($q(x_t|x_{t-1})$), noise schedules ($\beta_t$), the reparameterization trick, variance evolution, recursive structure ($\alpha_t, \bar{\alpha}_t$), and the forward jump formula ($x_t$ from $x_0$) for efficient training data generation.

### Day 2: Reverse Diffusion Process & Training Objective
* **Lecture 3: Mathematical Foundations of Generative Models** [▶️ View Slides](Lectures/Day2_Lec3_MathFoundationsGenerativeModels.pdf): Maximum Likelihood approach, the hidden variable problem, intractability of $p(x)$, essential mathematical toolkit (marginal distributions, expected values, Bayes' rule, KL divergences, Jensen's inequality), and Variational Inference strategy leading to the Evidence Lower Bound (ELBO).
* **Lecture 4: The ELBO for Diffusion Models - Learning to Reverse Chaos** [▶️ View Slides](Lectures/Day2_Lec4_ELBOForDiffusion.pdf): Adapting ELBO for sequential latents in diffusion models ($x_{1:T}$ hidden), exploiting Markovian structure, ELBO decomposition ($\mathcal{L}_0, \mathcal{L}_T, \mathcal{L}_{t-1}$), the denoising matching term ($\mathcal{L}_{t-1}$), the true reverse posterior $q(x_{t-1}|x_t, x_0)$, parameterizing the learned reverse step $p_{\theta}(x_{t-1}|x_t)$, the reparameterization breakthrough for $\mu_{\theta}$ by predicting noise $\epsilon_{\theta}(x_t, t)$, and the simplified loss objective $L_{simple}$.

### Day 3: Training, Sampling & Introduction to Conditional Generation
* **Lecture 5: Sampling from Trained Diffusion Models - From Noise to Data** [▶️ View Slides](Lectures/Day3_Lec5_SamplingFromDiffusion.pdf): Recap of training $\epsilon_{\theta}(x_t, t)$, the U-Net architecture for $\epsilon_{\theta}(x_t, t)$ (input, output, structure). DDPM Sampling Algorithm (Ancestral Sampling), its trade-offs. DDIM (Denoising Diffusion Implicit Models) for faster, deterministic sampling, the deterministic case, and DDIM sampling algorithm.
* **Lecture 6: Conditional Generation - From Random to Controllable** [▶️ View Slides](Lectures/Day3_Lec6_ConditionalGeneration.pdf): Motivation for conditional generation $p(x|y)$. U-Net adaptations for conditioning (time embedding, attention, cross-attention).
    * **Approach 1: Class-Conditional Diffusion Models:** Training $\epsilon_{\theta}(x_t, y, t)$, architectural modifications, pros & cons.
    * **Approach 2: Classifier Guidance:** Steering unconditional models with a classifier $p_{\phi}(y|x_t)$, mathematical basis, modified noise prediction, challenges, guidance scale $\omega$, pros & cons.
    * **Approach 3: Classifier-Free Guidance (CFG):** Implicitly learning classifier gradient, training with conditioning dropout, CFG sampling equation, dominance in text-to-image models.

### Day 4: Advanced Topics & Applications
* **Lecture 7: Score-Based Generative Models - Learning the Geometry of Data** [▶️ View Slides](Lectures/Day4_Lec7_ScoreBasedModels.pdf): Analogy for score functions, score function definition $s(x)=\nabla_x \log p(x)$, geometric intuition, Energy-Based Perspective, examples (Gaussian, Mixture of Gaussians). Sampling with Langevin Dynamics. Learning score functions via Score Matching, the explicit Score Matching objective, and its computational cost.
* **Lecture 8: Research Discussion - Generative Models for Information Retrieval, NLP, and RAG Systems** [▶️ View Slides](Lectures/Day4_Lec8_ResearchDiscussion.pdf): Exploring high-impact research opportunities, convergence of generative models and emerging needs. Research themes including Generative IR, Neural Document Generation, Advanced RAG Architectures, Multimodal Knowledge Integration, Personalized Knowledge Systems, Factual Accuracy, Efficient Systems, AI for Indian Communities, and Agentic AI.

### Day 5: Project Day
* Mini-Project Presentations by student teams.
* Technical discussions and knowledge sharing.
* Course wrap-up and future directions.

---
## Course Schedule 🗓️
(Main lecture topics and hands-on session titles. See "Labs" section below for lab details.)

| Time          | Day 1: Foundations & Forward                      | Day 2: Reverse & Training                         | Day 3: Training & Sampling                        | Day 4: Conditional & Advanced                     | Day 5: Project Day                   |
|---------------|---------------------------------------------------|---------------------------------------------------|---------------------------------------------------|---------------------------------------------------|--------------------------------------|
| 9:00-10:30    | **Lecture 1:** The Challenge of Generating Reality | **Lecture 3:** Mathematical Foundations of Generative Models | **Lecture 5:** Sampling from Trained Diffusion Models | **Lecture 7:** Score-Based Generative Models | Mini-Project Presentations (Groups 1-3) |
| 10:30-10:45   | Break                                             | Break                                             | Break                                             | Break                                             | Break                                |
| 10:45-12:00   | **Lecture 2:** The Forward Diffusion Process      | **Lecture 4:** The ELBO for Diffusion Models     | **Lecture 6:** Conditional Generation             | **Lecture 8:** Research Discussion: Generative Models for IR, NLP, RAG | Mini-Project Presentations (Groups 4-6) |
| 12:00-1:00    | Lunch Break                                       | Lunch Break                                       | Lunch Break                                       | Lunch Break                                       | Lunch Break                          |
| 1:00-3:00     | **Hands-on Session 1:** Naive Generators & Forward Diffusion Setup | **Hands-on Session 3:** U-Net for Diffusion & Reparameterization Trick | **Hands-on Session 5:** Diffusion ELBO Insights & Training Loop Setup | **Hands-on Session 7:** Advanced DDIM & Classifier-Free Guidance Intro | Mini-Project Presentations (Groups 7-9) |
| 3:00-3:15     | Break                                             | Break                                             | Break                                             | Break                                             | Break                                |
| 3:15-5:00     | **Hands-on Session 2:** Advanced Noise Schedules & Forward Jumps | **Hands-on Session 4:** Math Foundations: KL, Jensen's & VAE ELBO | **Hands-on Session 6:** DDPM Sampling: Stochastic Generation | **Hands-on Session 8:** Conditional Generation: CFG & Other Techniques | Mini-Project Presentations (Groups 10-12) & Course Wrap-up |
---
## Labs

| Lab #                                                                 | Colab                                                                                                                                                   | Readme                                              |
|-----------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------|
| **Lab 1** – Naive Generators & Forward Diffusion Setup <br/>_Intro to generative concepts, forward diffusion process, image data setup. (Syllabus: Lab 1 Parts 1,2, Start 3)_ | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shubham-lab/diffusion-course/blob/main/Labs/Lab1/Lab1.ipynb) | [View README](/Labs/Lab1/README.md)               |
| **Lab 2** – Advanced Noise Schedules & Forward Jumps <br/>_Exploring noise schedules ($\beta_t$) & implementing the "forward jump" formula. (Syllabus: Lab 1 Finish 3; Lab 2 P3,4)_    | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shubham-lab/diffusion-course/blob/main/Labs/Lab2/Lab2.ipynb) | [View README](/Labs/Lab2/README.md)               |
| **Lab 3** – U-Net for Diffusion & Reparameterization Trick <br/>_Building/adapting U-Net for noise prediction & applying reparameterization trick. (Syllabus: Lab 1 P4.1; Lab 2 P2)_      | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shubham-lab/diffusion-course/blob/main/Labs/Lab3/Lab3.ipynb) | [View README](/Labs/Lab3/README.md)               |
| **Lab 4** – Math Foundations: KL, Jensen's & VAE ELBO <br/>_Exercises on KL divergence, Jensen's inequality & VAE ELBO concepts. (Syllabus: Lab 3 P2,3,4,5)_                            | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shubham-lab/diffusion-course/blob/main/Labs/Lab4/Lab4.ipynb) | [View README](/Labs/Lab4/README.md)               |
| **Lab 5** – Diffusion ELBO Insights & Training Loop Setup <br/>_Simplified DDPM loss objective & setting up the training loop. (Syllabus: Lab 1 P4.2; Lab 4 P3,4.2)_                         | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shubham-lab/diffusion-course/blob/main/Labs/Lab5/Lab5.ipynb) | [View README](/Labs/Lab5/README.md)               |
| **Lab 6** – DDPM Sampling: Stochastic Generation <br/>_Implementing DDPM ancestral sampling to generate data. (Syllabus: Lab 5 Part 3)_                                                   | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shubham-lab/diffusion-course/blob/main/Labs/Lab6/Lab6.ipynb) | [View README](/Labs/Lab6/README.md)               |
| **Lab 7** – Advanced DDIM ($\eta$) & Classifier-Free Guidance Intro <br/>_DDIM for faster sampling & intro to classifier-free guidance. (Syllabus: Lab 5 P4.2; Lab 6 P5 Initial)_          | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shubham-lab/diffusion-course/blob/main/Labs/Lab7/Lab7.ipynb) | [View README](/Labs/Lab7/README.md)               |
| **Lab 8** – Conditional Generation: CFG & Other Techniques <br/>_Implementing Classifier-Free Guidance (CFG) & project work. (Syllabus: Lab 6 P3,4.5; Project Work)_                  | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shubham-lab/diffusion-course/blob/main/Labs/Lab8/Lab8.ipynb) | [View README](/Labs/Lab8/README.md)               |

---
## Mini-Project 🛠️
The mini-project is a key component of this course, designed to give students hands-on experience in developing, training, and applying diffusion models to practical problems.
* **Format:** Team-based (teams of two students). Projects will be announced on Day 1.
* **Task:** Develop, implement, and present a working diffusion model application. This must demonstrate the successful implementation of a diffusion process, training methodology, and sampling procedure.
* **Work:** Teams are expected to work on the project independently outside of class hours. No separate time will be allocated during teaching hours for project work.
* **Evaluation:** Projects will be evaluated on Day 5.
* **Creativity:** Creativity and originality are encouraged.
* **Guidelines:**
    * All projects must include a working code implementation and a live demonstration during the final presentation.
    * Students are encouraged to explore creative applications of diffusion models beyond the examples provided.
    * Clear documentation and well-organized code will be considered in the final assessment.

### Suggested Mini-Project Topics
Teams of two students will select a project topic from the list below or propose a custom topic (subject to instructor approval).
* Image Denoising using Diffusion Models (e.g., MNIST, CIFAR-10)
* Image Inpainting with Diffusion
* Class-Conditional Image Generation
* Text-to-Image Generation with Simple Prompts
* Super-Resolution via Diffusion
* Latent Diffusion Modeling
* Time Series Data Generation (e.g., stock prices, ECG patterns)
* Style Transfer through Diffusion Interpolation
* Noise Scheduling Experiments (linear, cosine, learned)
* Guided Handwriting Generation (conditioned on characters)

### Custom Projects
Students may propose their own mini-projects, provided they meet the following requirements:
* The project must involve the core concepts of diffusion modeling, training, and sampling.
* The project must be sufficiently challenging and require original implementation work.
* A short proposal (title, description, expected outcomes, and dataset to be used) must be submitted for approval by the instructor within the first two days of the course. (A proposal form is available in the syllabus).

---
## Assessment 💯
The course will be evaluated based on a team-based mini-project, which constitutes **100% of the final grade**. The assessment will be based on demonstrating technical understanding, problem-solving skills, creativity, and the ability to apply course concepts to real-world scenarios.

The assessment dimensions are:
* **Technical Implementation (40%):** Correctness, efficiency, and completeness of the diffusion model implementation.
* **Oral Presentation (20%):** Clarity, structure, and professionalism of the project presentation.
* **Quality of Results (20%):** Quality of generated outputs, experimental results, and evaluation metrics.
* **Understanding of Theoretical Concepts (20%):** Depth of understanding demonstrated through explanation and responses to questions.

A detailed rubric is available in the course syllabus (Table 2).

---
## Required Reading 📖
* **"Probabilistic Machine Learning: Advanced Topics"**
    * Available at: [https://probml.github.io/pml-book/book2.html](https://probml.github.io/pml-book/book2.html)
    * Specific chapters will be specified in the detailed schedule.

---
## Repository Contents 📂
This repository is structured to provide easy access to all course materials:
* `README.md`: This file, providing an overview of the course and repository.
* `/Syllabus`: The course syllabus PDF document (`syllabus.pdf`).
* `/Lectures`: Contains PDF versions of the lecture slides for each day/session (e.g., `Day1_Lec1_ChallengeOfGeneratingReality.pdf`).
* `/Labs`: Contains folders for each lab session (e.g., `/Lab1`, `/Lab2`). Each lab folder will include:
    * `README.md`: A detailed description and instructions for the lab.
    * `*.ipynb`: A Colab/Jupyter notebook for the hands-on exercises (e.g., `Lab1.ipynb`).
* `/Mini_Project_Resources`: May contain the mini-project proposal form, detailed rubric (or links to the syllabus), and any other resources relevant to the mini-project.

---
## Software and Setup 💻
* **Python:** A recent version of Python (e.g., 3.8+) is recommended.
* **PyTorch:** Proficiency in PyTorch is a prerequisite. Ensure you have a working PyTorch environment.
* **Other Libraries:** Specific labs might require additional Python libraries (e.g., `numpy`, `matplotlib`, `torchvision`, etc.). These will be mentioned in the respective lab READMEs and notebooks.
* **Google Colab:** Lab notebooks are designed to be run on Google Colab to ensure a consistent environment with necessary compute resources (like GPUs).

It is highly recommended to use a virtual environment (e.g., Conda, venv) if running labs locally, though Colab is the primary platform for labs.

---
## How to Use This Repository
1.  **Clone the Repository (Optional):**
    ```bash
    git clone <repository_url>
    ```
    You can also download specific files or use materials directly via GitHub links.
2.  **Navigate to Folders:** Access lecture slides in `/Lectures` and lab materials in `/Labs`. The syllabus is in `/Syllabus`.
3.  **Review Materials:** It's recommended to review the lecture slides before or after each session.
4.  **Engage with Labs:** For each lab session:
    * Refer to the "Labs" section in this README for direct links to the lab's README and Colab notebook.
    * Alternatively, navigate to the lab folder (e.g., `/Labs/Lab1/`) to find its `README.md` and `*.ipynb` file.
5.  **Mini-Project:** Utilize resources in `/Mini_Project_Resources` and the syllabus for guidance on the mini-project.

---
## Disclaimer
All materials in this repository are provided for educational purposes as part of the "Diffusion Models: Theory and Applications" course at PES University, Summer 2025.
