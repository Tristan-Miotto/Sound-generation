# Sound generation 🔊

This project was made in 2023 with Clément Souvignet, as part of our academic education in Polytech Clermont.

## Description of the project
The objective of this group project is to...
- Learn about the theory of **sound processing** as well as it's application on Python (using the Python library Librosa).
- Build **neural networks** (using Tensorflow) that process different tasks about sounds. 

More specifically, we developed 4 deep learning models...
1) Classification of numbers pronounced in English 🔢. 
2) Gender classification 👩👨. 
3) Age classification 👦👴. 
4) Generation of virtual voices that pronounce a number (between 0 and 9) in English 🔢. 

In this repository, I only present simplified code for the **sound generator model**, which I believe is the most interesting. Despite the limitations of our computational environment, we achieved good results. In fact, we were generally able to identify the numbers pronounced by the virtual voices. I created Jupyter Notebooks with summaries and comments so that we can understand our work.

## Content of the repository

This repo contains:

### 1 – A *data_processing.ipynb* Jupyter Notebook 💻

Before running this code, we need to download the [Audio MNIST](https://www.kaggle.com/datasets/sripaadsrinivasan/audio-mnist) dataset available in Kaggle. The program provided transforms each audio signal to a **mel spectrogram**. They represent sounds with a frequency scale that more closely resembles how humans perceive pitch. Mel spectrograms will constitute the input data of our model. Finally, the program save the data (train and test) as well as normalization data (min and max) in a *data* folder.

### 2 – A *generation.ipynb* Jupyter Notebook 💻

In this Python file, we build a variational autoencoder (VAE) that generates virtual voices.
- First, we load data from the *data* folder.
- Then, we create our VAE and we train it with input data (mel spectrograms). As we want our latent space to follow a normal distribution, we need to define the **Kullback-Leibler (KL) divergence**.
- After the training phase of the model, we generate a vector following a standard normal distribution.
- Finally, we denormalize the generated mel spectrogram and we transform it into an audio signal, which we can then listen to.

As a result, we get a virtual voice (it looks like a robot) saying a digit. Most of the time, we can recognize the number pronounced. We assume that, to obtain clearer sounds, we need to improve the quality of the mel spectrograms. In other terms, we must increase their dimension (width and height).

### 3 –  A *mathematical_proof_KL_divergence.png* file ➗✖️

KL divergence is a measure that quantifies how one probability distribution $P$ differs from a second $Q$. It is defined as followed :

$$D_{KL}(P \|\| Q) = \int_{-\infty}^{\infty} p(x) \log\left(\frac{q(x)}{p(x)}\right) dx$$

where $p$ (resp. $q$) is the density associated with $P$ (resp. $Q$).

The *mathematical_proof_KL_divergence.png* file contains the written demonstration of the KL divergence in context of VAEs, which is :

$$\frac{1}{2} \sum_{i=1}^{d} \mu_i^2 + \sigma_i^2 - 1 - \log(\sigma_i^2)$$

where $\mu_i^2$ and $\sigma_i^2$ are outputs of the encoder used to build the latent space.

### 4 –  A *report_project_in_french.pdf* file 📄

This document is the report of the project, written is French. It describes more in detail...
- Sound processing.
- The classifaction models that we built.
- The sound generator. In fact, we actually made a **grid search** to find the best combination of hyperparameters. We also experimented with different data preprocessing techniques (dimension of melspectrograms, normalization). Here, to keep it simple, I manually defined the hyperparameters without performing a grid search.

## 📌 Note

⚠️ It seems that the *generation.ipynb* Notebook can't run with a recent version of Tensorflow . However, it's working with the 2.10 version. 
