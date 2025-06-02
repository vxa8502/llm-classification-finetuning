# LLM Classification Finetuning

This project fine-tunes a DeBERTa-v3 extra-small language model to predict human preference in chatbot responses using conversations from Chatbot Arena. The model is trained on a Kaggle dataset and optimized for performance and memory efficiency.  The primary goal is to classify which of two chatbot responses to a given prompt is preferred by a human judge, or if there is a tie.

## Overview

The core task is to develop a preference model that accurately predicts which response users will choose in a head-to-head comparison of chatbots powered by Large Language Models (LLMs).  This is analogous to building a reward model in Reinforcement Learning from Human Feedback (RLHF) and addresses biases like position bias, verbosity bias, and self-enhancement bias that LLMs often exhibit when directly prompted for preference predictions.

## Approach

This project employs a siamese-like architecture with a shared DeBERTa-v3 backbone to process and compare two chatbot responses for a given prompt. Key techniques used include:

* **Data Preprocessing:**  Careful cleaning and formatting of the prompt/response pairs from the Chatbot Arena dataset, including handling encoding failures and converting the original string representation of lists into usable text.
* **Response Swapping Augmentation:** This technique is crucial for mitigating position bias, where the model might learn to favor responses presented in a particular order.
* **Contextualized Responses:** Combining prompts and responses into formatted text pairs helps the model learn the relationship between the two.
* **Mixed Precision Training:** Using FP16 for the forward pass and FP32 for loss computation reduces memory usage and speeds up training, essential for managing the large DeBERTa model.
* **Gradient Accumulation:** Simulates a larger effective batch size while staying within memory constraints.
* **Cosine Annealing Learning Rate Scheduler with Warmup:**  Improves training stability and convergence by gradually increasing the learning rate before decaying it over epochs.
* **Model Checkpointing:** Saves the best model weights based on validation loss.

## Dataset

The dataset is from the [LLM Classification Finetuning Kaggle competition](link-to-competition-if-available).  It includes:

* `train.csv`: Contains prompts, responses from two models (model_a and model_b), and the human judge's preferred winner (winner_model_a, winner_model_b, or winner_tie).
* `test.csv`: Contains prompts and responses, requiring predictions for the preferred winner.

## Code

The code is provided in a Kaggle notebook (`llm-classification-notebook.ipynb`).  It details the data loading, preprocessing, model building, training, prediction, and submission generation steps.

## Results

The notebook includes plots of training and validation loss and accuracy. The final validation accuracy achieved with this setup is approximately 38.5%.

## Potential Improvements

* **Ensemble Methods:** Combining predictions from multiple models can improve robustness.
* **Advanced Augmentation Techniques:** Exploring paraphrasing or backtranslation could enrich the training data and further reduce biases.
* **Larger DeBERTa Models:**  If computational resources permit, using a larger DeBERTa-v3 model (base or large) could capture more nuanced relationships in the text.
* **Feature Engineering:**  Adding features like response length ratios or readability scores might provide additional signals for the model.
* **k-Fold Cross-Validation:**  Would provide a more robust estimate of the model's generalization performance.
