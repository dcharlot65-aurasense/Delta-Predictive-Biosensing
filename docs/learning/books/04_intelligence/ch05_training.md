# Chapter 5: Training - Practice Makes Perfect

## From Random Guessing to Expert Performance

Imagine learning to shoot basketball free throws. Your first attempt? The ball might go anywhere - way too far, too short, left, right, or hitting the rim. You're essentially guessing.

But you try again. And again. Each attempt gives you feedback: too hard, too soft, too far left. Your brain adjusts. After dozens of attempts, you're making occasional baskets. After hundreds, you're pretty good. After thousands, you might become an expert, making 90% of your shots.

This is training. You start with random performance and gradually improve through practice with feedback.

Neural networks learn exactly the same way. A newly created network with random weights makes random predictions - no better than chance. But after training on thousands or millions of examples, getting feedback on each one, the network becomes an expert.

This chapter reveals exactly how training works. It's not mysterious. It's systematic adjustment based on errors, repeated until performance improves.

---

## The Training Dataset: Your Practice Problems

Before you can train a network, you need training data - lots of examples showing inputs and correct outputs.

For heart monitoring, training data might include:
- 10,000 ECG recordings
- Each labeled by cardiologists: "normal," "atrial fibrillation," "ventricular tachycardia," etc.
- From diverse patients: young, old, male, female, different health conditions
- Representing different scenarios: resting, exercising, sleeping

Each example teaches the network something. "When you see this ECG pattern, the correct answer is atrial fibrillation." The network adjusts its weights to produce this answer for similar patterns in the future.

**Quality over quantity** (but quantity helps too):
- 1,000 diverse, accurately labeled examples > 10,000 repetitive, poorly labeled ones
- More data usually improves performance, but only if it's good data
- Balanced datasets prevent bias (don't train on 99% normal hearts if you want to detect problems!)

Training data is the foundation. Everything else builds on it.

> **Did You Know?**
>
> ImageNet, a famous image dataset, contains over 14 million labeled images. Creating it required thousands of people spending years labeling pictures. But this dataset enabled breakthrough advances in computer vision! Sometimes the hardest part of machine learning isn't the algorithms - it's getting good training data.

---

## The Training Process: An Overview

Training follows a cycle repeated thousands of times:

1. **Forward Pass**: Feed an input through the network to get a prediction
2. **Calculate Error**: Compare the prediction to the correct answer
3. **Backward Pass**: Calculate how each weight contributed to the error
4. **Update Weights**: Adjust weights to reduce the error
5. **Repeat**: Move to the next training example

After processing all training examples once, that's one "epoch." Training typically requires many epochs - 10, 100, even 1000 or more - until the network's performance plateaus.

Let's explore each step in detail.

---

## Forward Pass: Making a Prediction

The forward pass is straightforward - you learned this in Chapter 3. Data flows from input layer through hidden layers to output layer:

**Layer 1 (Input)**: Receives raw features (ECG signal values, heart rate, etc.)

**Layer 2 (Hidden)**: Each neuron computes weighted sum of inputs, applies activation function
- Neuron 1: sum = (x₁ × w₁,₁) + (x₂ × w₁,₂) + ... + bias₁
- Output 1: activation(sum)
- Repeat for all neurons in this layer

**Layer 3 (Hidden)**: Takes previous layer's outputs as inputs, repeats the process
- More layers mean more complex patterns can be learned
- Information gets progressively transformed

**Output Layer**: Final predictions
- For classification: probabilities for each category
- For regression: predicted numerical value

Example: An ECG signal enters the network. Hidden layers detect features like wave peaks, intervals, and rhythm. The output layer produces probabilities: 85% normal, 10% atrial fibrillation, 5% other.

At the start of training, with random weights, these outputs are meaningless. That's okay - learning will fix them.

---

## Calculating Error: How Wrong Are We?

The network made a prediction. Now compare it to the correct answer and measure the error.

**Loss Function**: A mathematical formula that quantifies how wrong the prediction is. Smaller loss = better prediction.

**For Classification** (picking a category):

**Cross-Entropy Loss**: Measures the difference between predicted probabilities and the true label.
- True label: "normal" (represented as [1, 0, 0] for normal, atrial fib, other)
- Prediction: [0.85, 0.10, 0.05]
- Loss: Penalizes the difference, especially when predictions are confident but wrong

If the network predicts 85% normal and the true answer is normal, loss is small. If it predicts 85% normal but the true answer is atrial fibrillation, loss is large.

**For Regression** (predicting a number):

**Mean Squared Error (MSE)**: Average of squared differences between predictions and true values.
- True blood sugar: 120 mg/dL
- Prediction: 135 mg/dL
- Error: 135 - 120 = 15
- Squared error: 15² = 225

Squaring makes large errors count more than small errors. A 20-point miss is more than twice as bad as a 10-point miss (400 vs 100).

The loss function is crucial because it guides learning. The network tries to minimize loss, so the loss function defines what "good performance" means.

---

## Backward Pass: Backpropagation

Here's where it gets interesting. We know the network made an error. But which weights caused it? The network might have thousands or millions of weights. Which ones should we adjust?

**Backpropagation** (backward propagation of errors) solves this problem through calculus. The key insight: we can calculate how much each weight contributed to the error.

Think of it like diagnosing a problem. A factory produces defective products. You trace backward through the assembly line, identifying which steps introduced the defects. Similarly, backpropagation traces backward through the network, identifying which weights contributed most to the error.

**The Math (Simplified)**:

For each weight, calculate its "gradient" - mathematically, how much the loss would change if we slightly increased this weight. Think of it as the weight's error contribution.

- Positive gradient: Increasing this weight would increase the error (bad!) → Decrease it
- Negative gradient: Increasing this weight would decrease the error (good!) → Increase it
- Large gradient: This weight has strong influence → Adjust it significantly
- Small gradient: This weight barely matters → Adjust it slightly

**Working Backward Through Layers**:

1. Start at the output layer: Calculate how output neuron errors relate to their incoming weights
2. Move to the previous layer: Calculate how its errors contributed to the output errors
3. Continue backward through all layers to the input

Each layer's gradients depend on the next layer's gradients - that's why it's "backpropagation." Errors propagate backward from output to input.

The calculus chain rule makes this computation efficient. Without going into mathematical details, just know that we can calculate all gradients with roughly the same computational cost as the forward pass.

---

## Updating Weights: Gradient Descent

Now we know each weight's gradient - its contribution to the error. Time to update the weights to reduce the error.

**Gradient Descent**: A simple but powerful algorithm.

For each weight:
**new_weight = old_weight - (learning_rate × gradient)**

That's it! Adjust each weight in the direction that reduces loss.

**The Learning Rate**: This crucial parameter controls how big each step is.
- Too large: The network might overshoot, jumping past good solutions and becoming unstable
- Too small: Learning takes forever, requiring millions of examples
- Just right: Steady improvement, balancing speed and stability

Typical learning rates: 0.001 to 0.1, depending on the problem.

**Analogy: Descending a Mountain in Fog**

Imagine you're on a mountain in thick fog. You want to reach the valley (lowest point). You can't see far, but you can feel which direction is downhill.

- The mountain's height represents the loss function
- Your position represents the current weights
- Downhill represents negative gradient
- Your step size represents the learning rate

You feel which direction is downhill and take a step that way. Repeat. Eventually, you reach a valley (low loss). You might not find the absolute lowest point on the entire mountain, but you'll find a pretty good spot.

This is gradient descent: follow the gradient (downhill direction) until you reach a minimum.

> **Did You Know?**
>
> Advanced optimizers like Adam and RMSprop modify basic gradient descent with "momentum" (like a ball rolling down the hill, building speed) and adaptive learning rates (different rates for different weights). These techniques dramatically improve training speed and reliability.

---

## Batches and Epochs

Processing one training example at a time is inefficient. Modern training uses batches:

**Batch**: A group of training examples processed together.
- Calculate predictions for all examples in the batch (forward pass)
- Calculate average loss across the batch
- Calculate gradients based on average loss
- Update weights once per batch

**Batch Size**: Typically 16, 32, 64, or 128 examples.
- Larger batches: More stable gradient estimates, better GPU utilization
- Smaller batches: More frequent updates, sometimes better final performance

**Epoch**: One complete pass through all training data.
- If you have 10,000 examples and batch size 100, one epoch = 100 batches
- Training typically requires many epochs (10-1000+)

**Training Progress**:
- Epoch 1: Random weights, loss is high, accuracy is low (maybe 25% for 4-category classification)
- Epoch 10: Weights improving, loss decreasing, accuracy rising (maybe 60%)
- Epoch 50: Good performance (maybe 85%)
- Epoch 100: Excellent performance (maybe 92%), but improvement is slowing
- Epoch 200: Marginal improvements (92.5%), might be overfitting (more on this in Chapter 6)

You monitor loss and accuracy after each epoch to track progress.

---

## When to Stop Training

How do you know when training is done? Several signals:

**Loss Plateaus**: If loss stops decreasing for many epochs, additional training might not help. You've found a good solution (or at least a local minimum).

**Validation Performance Peaks**: Hold out some data (not used for training) as a validation set. If validation accuracy starts decreasing while training accuracy increases, you're overfitting (Chapter 6 covers this). Stop training!

**Time/Resource Limits**: Sometimes you simply run out of time or computing budget. Take the best model so far.

**Target Reached**: If you achieve your performance goal (say, 95% accuracy), you can stop.

**Early Stopping**: Monitor validation loss. If it doesn't improve for N epochs (say, 20), automatically stop. This prevents overfitting.

There's no perfect rule. Training is part science, part art. Experience teaches you what works for different problems.

---

## Hyperparameters: The Settings You Choose

Unlike the weights (which the network learns), hyperparameters are settings you choose before training:

**Learning Rate**: How fast to adjust weights (0.001? 0.01? 0.1?)

**Batch Size**: How many examples per batch (32? 64? 128?)

**Number of Layers**: How deep is your network (3 layers? 10? 100?)

**Neurons Per Layer**: How wide is each layer (64? 128? 512?)

**Activation Functions**: ReLU? Sigmoid? Tanh?

**Optimizer**: Basic gradient descent? Adam? RMSprop?

**Regularization**: Techniques to prevent overfitting (Chapter 6)

These choices dramatically affect performance. Finding good hyperparameters often requires experimentation:

**Grid Search**: Try many combinations systematically
**Random Search**: Sample random combinations (often works surprisingly well!)
**Bayesian Optimization**: Use sophisticated algorithms to search efficiently
**Trial and Error**: Learn from experience what works for similar problems

Professional ML engineers spend significant time on hyperparameter tuning. The network architecture and training settings matter as much as the algorithm itself.

> **Did You Know?**
>
> Google's AutoML systems automatically search for optimal hyperparameters and architecture. They try thousands of combinations, training each for hours or days, to find the best configuration. This requires massive computing power but can discover architectures that human experts wouldn't think to try!

---

## Practical Example: Training a Heart Monitor

Let's walk through training a network to detect abnormal heart rhythms:

**Data Preparation**:
- 10,000 ECG recordings, each 10 seconds long
- Labels: normal (7,000), atrial fibrillation (2,000), other arrhythmia (1,000)
- Split: 8,000 training, 1,000 validation, 1,000 test
- Preprocessing: Normalize signals to mean=0, std=1

**Network Architecture**:
- Input layer: 1,000 values (ECG signal samples)
- Hidden layer 1: 128 neurons, ReLU activation
- Hidden layer 2: 64 neurons, ReLU activation
- Output layer: 3 neurons (one per category), softmax activation

**Training Settings**:
- Batch size: 32
- Learning rate: 0.001
- Optimizer: Adam
- Loss: Cross-entropy
- Epochs: 100 maximum, with early stopping

**Training Process**:
- Epoch 1: Loss=1.2, Training accuracy=40%, Validation accuracy=38%
- Epoch 10: Loss=0.7, Training accuracy=70%, Validation accuracy=68%
- Epoch 30: Loss=0.3, Training accuracy=88%, Validation accuracy=85%
- Epoch 50: Loss=0.15, Training accuracy=94%, Validation accuracy=87%
- Epoch 70: Loss=0.08, Training accuracy=97%, Validation accuracy=87%

Notice: Training accuracy keeps improving, but validation accuracy stopped improving after epoch 50. This signals overfitting - the network is memorizing training data rather than learning general patterns.

**Result**: Stop at epoch 50, use those weights. Final model: 87% validation accuracy.

This is realistic training. Not perfect, but good enough for real-world use (with proper testing, covered in Chapter 6).

---

## Challenges in Training

Training doesn't always go smoothly. Common problems:

**Vanishing Gradients**: In very deep networks, gradients become tiny as they propagate backward. Early layers barely learn. Solutions: better activation functions (ReLU instead of sigmoid), skip connections, batch normalization.

**Exploding Gradients**: Opposite problem - gradients become huge, causing unstable updates. Solutions: gradient clipping, careful initialization, lower learning rates.

**Local Minima**: Gradient descent might get stuck in local valleys instead of finding the global minimum. Solutions: momentum, random restarts, better optimizers.

**Slow Convergence**: Training takes forever. Solutions: higher learning rates (carefully!), better optimizers, more powerful hardware, transfer learning.

**Overfitting**: The network memorizes training data instead of learning general patterns. Chapter 6 addresses this crucial issue.

Modern techniques handle most of these challenges, but training large networks still requires expertise and experimentation.

---

## Transfer Learning: Standing on Shoulders

Here's a powerful trick: don't start from random weights!

**Transfer Learning**: Start with a network pre-trained on related tasks, then fine-tune it for your specific task.

Example: You want to classify medical images. Instead of training from scratch, you:
1. Start with a network pre-trained on millions of general images (ImageNet)
2. Replace the final layer with one suited to your task
3. Train on your medical images, adjusting all weights but starting from a good initialization

This works because early layers learn general features (edges, textures, shapes) useful for many tasks. You're transferring that knowledge to your problem.

**Benefits**:
- Much faster training (hours instead of weeks)
- Requires less data (thousands instead of millions of examples)
- Often better performance

For biosignaling, you might use a network pre-trained on general time-series data, then fine-tune it for ECG, EEG, or glucose prediction.

Transfer learning is like learning Spanish after already knowing Italian - the grammar and vocabulary similarities give you a head start.

---

## Why This Matters for Biosensing

Understanding training helps you evaluate biosensing AI systems:

**Data Requirements**: A system claiming 99% accuracy - how much training data did it use? Thousands of examples? Millions? Diverse populations or narrow?

**Training Transparency**: Can the developers explain their training process, hyperparameters, and validation approach? Or is it a black box?

**Performance Claims**: Training accuracy vs. validation accuracy vs. test accuracy. If only training accuracy is reported, be skeptical - overfitting might mean real-world performance is much worse.

**Continuous Learning**: Some systems retrain on your personal data, adapting to your unique patterns. This requires careful training to avoid overfitting to noise.

**Resource Costs**: Training large models requires significant computing power and energy. Understanding this helps appreciate the engineering investment behind biosensing AI.

When you trust an AI to monitor your health, you want confidence that it was trained properly with good data, validated thoroughly, and will generalize to your specific situation.

---

## What's Next?

Training creates a network that performs well on training data. But that's not enough. We need to know: Does it work on new data it has never seen?

This is the crucial question addressed in Chapter 6: Testing. You'll learn about train/test splits, overfitting, validation, and how to measure real-world performance.

A model that memorizes training examples perfectly but fails on new data is useless. Testing reveals whether the network truly learned general patterns or just memorized specific examples.

Understanding the difference between training performance and testing performance is essential for critically evaluating any AI system. Let's explore how to test properly.

---

**Chapter 5 Summary**: Training is the process of adjusting neural network weights to minimize prediction errors. It involves: (1) forward passes making predictions, (2) calculating loss measuring prediction errors, (3) backpropagation computing each weight's error contribution, and (4) gradient descent updating weights to reduce loss. This cycle repeats for thousands of examples across multiple epochs. Hyperparameters like learning rate, batch size, and architecture must be chosen carefully. Training stops when loss plateaus or validation performance peaks. Common challenges include vanishing/exploding gradients, local minima, and overfitting. Transfer learning speeds training by starting with pre-trained weights. Understanding training helps evaluate whether biosensing AI systems were developed rigorously with sufficient, diverse data.

**Next**: [Chapter 6: Testing - Does It Really Work?](ch06_testing.md)
