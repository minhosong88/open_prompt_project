# Prompt-Based Learning Repository

This repository contains a collection of Jupyter notebooks that demonstrate various prompt-based learning techniques applied to natural language processing (NLP) tasks. The focus is on utilizing **prompt engineering** to adapt pre-trained models for tasks like reading comprehension, question answering, and binary classification. The notebooks explore methods like **manual templates**, **prefix tuning**, and **p-tuning**, leveraging popular pre-trained models such as **T5**, **RoBERTa**, and **GPT-2**.

---
## Main Environment Factors

This repository relies on a Python environment configured for machine learning and NLP tasks. Below are the key environment specifications derived from the provided `environment.yml`:

### 1. **Python Version**
- Python 3.10 is the base version for compatibility with required libraries.

### 2. **Key Libraries**
- **Numpy**, **Pandas**: For data manipulation and preprocessing.
- **Scikit-learn**: For additional machine learning utilities.
- **Matplotlib**, **Tqdm**: For data visualization and progress tracking.

### 3. **Machine Learning Frameworks**
- **PyTorch** (including `torchvision`, `torchaudio`): For deep learning model implementation.
- **CUDA Toolkit 11.3**: For GPU acceleration (ensure compatibility with your GPU).

### 4. **NLP-Specific Libraries**
- **HuggingFace Transformers (v4.25.0)**: For pre-trained models like T5, RoBERTa, and GPT-2.
- **OpenPrompt (v1.0.1)**: Core library for prompt-based learning techniques.
- **SentencePiece**: For tokenizer compatibility with certain models.
- **NLTK**: For additional text preprocessing utilities.
- **Datasets**, **Evaluate**: HuggingFace libraries for benchmark dataset loading and model evaluation.

### 5. **Development Environment**
- **JupyterLab**: For running and experimenting with the provided notebooks.

---

## How to Set Up

### 1. **Using Conda**
To replicate the environment, create a Conda environment using the provided `environment.yml` file:
```bash
conda env create -f environment.yml 
```
### 2. Activate the Environment
Activate the environment for use:
```bash
conda activate prompt_learning_env
```
---
### 3. GPU Compatibility
Ensure that your system supports CUDA Toolkit 11.3 for GPU acceleration. If using a CPU-only setup, remove cudatoolkit=11.3 from the environment.yml file and install an appropriate CPU-compatible PyTorch version.

## Contents

### 1. **Notebooks**
- **`manual_template_qa_t5.ipynb`**:
  - Implements a multiple-choice question-answering (QA) task using **T5**.
  - Utilizes a manual template and verbalizer to structure the task as a binary classification problem.
  - Explores both zero-shot and few-shot learning.

- **`p_tuning_template_roBERTa.ipynb`**:
  - Demonstrates **p-tuning** with **RoBERTa** for binary sentiment classification.
  - Fine-tunes soft prompts without altering the underlying model weights.

- **`prefix_tuning_template_gpt2.ipynb`**:
  - Applies **prefix tuning** to **GPT-2** for text generation and classification tasks.
  - Focuses on optimizing a small set of virtual tokens while keeping the model parameters frozen.

- **`prefix_tuning_template_t5.ipynb`**:
  - Implements **prefix tuning** with **T5** for binary classification tasks.
  - Highlights the impact of hyperparameter tuning on model performance.

- **`qa_multiple_choice_id_t5.ipynb`**:
  - Explores question-answering as a classification task with **T5**.
  - Frames the task using manual templates and evaluates the model using both few-shot and zero-shot approaches.

- **`rc_manual_template_t5.ipynb`**:
  - Implements a **reading comprehension** task with **T5**, reframed as binary classification.
  - Demonstrates how fine-tuning on a small dataset of examples sharing the same background can improve performance.

---

## Key Techniques

### 1. **Manual Templates and Verbalizers**
- Structures input data using human-readable templates for task-specific formatting.
- Maps model predictions to predefined label words for classification tasks.

### 2. **Prefix Tuning**
- Optimizes virtual tokens appended to the input sequence, guiding the model without modifying its parameters.
- Efficient for fine-tuning large pre-trained models like T5 and GPT-2.

### 3. **P-Tuning**
- Leverages continuous prompts to improve performance on tasks like sentiment classification.
- Demonstrates the flexibility of soft prompts with minimal parameter tuning.

### 4. **Few-Shot and Zero-Shot Learning**
- Evaluates models with minimal labeled data (few-shot) or without additional training (zero-shot).
- Highlights the adaptability of pre-trained models for diverse NLP tasks.

---

## Observations and Limitations

- **Few-Shot Learning**:
  - Boosts performance for structured tasks, particularly with consistent context or background in the training data.
- **Fine-Tuning Challenges**:
  - While fine-tuning improves task-specific accuracy, it may confuse some pre-trained models, reducing their baseline performance.
- **Binary Classification**:
  - Simplifies complex tasks (e.g., reading comprehension) but may lose nuanced understanding or ranking capabilities.

---

## Applications

- **Reading Comprehension**: Reframes traditional comprehension tasks into simpler binary classification problems.
- **Question Answering**: Adapts pre-trained models for multiple-choice QA tasks using prompt-based techniques.
- **Sentiment Analysis**: Demonstrates the effectiveness of prompt tuning for binary and multiclass classification.
- **Few-Shot and Zero-Shot Learning**: Explores the limits of pre-trained models with minimal labeled data.

---

## How to Use

1. **Set Up Environment**:
   - Install the required dependencies listed in the notebooks (e.g., OpenPrompt, Transformers, Datasets).
   - Configure the paths for pre-trained models and datasets.

2. **Run Notebooks**:
   - Open any of the provided notebooks in Jupyter or similar environments.
   - Follow the step-by-step implementation for each task.

3. **Experiment**:
   - Modify the templates, verbalizers, or hyperparameters to observe their impact on model performance.
   - Test additional datasets or pre-trained models to explore generalizability.

---

## Conclusion

This repository provides a comprehensive guide to applying prompt-based learning techniques to various NLP tasks. By leveraging pre-trained models with minimal tuning, it demonstrates the power and flexibility of prompt engineering for both zero-shot and few-shot learning scenarios.

Contributions and feedback are welcome!
