# Chapter 3: Building Artificial Neurons

## Can We Build a Brain Cell?

A biological neuron is incredibly complex. It has thousands of dendrites receiving signals, intricate chemical machinery, genes that control its behavior, and the ability to grow, heal, and adapt. It's a living cell with all the complexity that implies.

Do we need to replicate all that complexity to create artificial intelligence?

Fortunately, no. Engineers discovered that you can capture the essential function of a neuron - receiving inputs, making decisions, producing outputs - with something much simpler. You don't need chemistry, dendrites, or living cells. You need mathematics.

This chapter introduces the perceptron, the simplest artificial neuron. It's nothing like a real neuron physically, but it mimics the basic input-process-output principle. And when you connect many perceptrons together, remarkable capabilities emerge.

---

## The Perceptron: A Simple Decision Maker

Imagine you're deciding whether to go outside. You consider several factors:

- Is it raining? (Yes = -1, No = +1)
- Is it warm? (Yes = +1, No = -1)
- Do you have free time? (Yes = +1, No = -1)
- Are friends available? (Yes = +1, No = -1)

You don't weight these factors equally. Weather is very important - you really don't want to go out in the rain. Friend availability is moderately important. Time and temperature matter less.

So you assign importance weights:
- Rain: 5 points
- Warmth: 2 points
- Free time: 1 point
- Friends available: 3 points

Now you calculate: multiply each factor by its weight and add them up. If the total is positive, you go outside. If negative, you stay in.

Example: It's not raining (+1×5=5), it's cold (-1×2=-2), you have time (+1×1=1), but friends are busy (-1×3=-3). Total: 5-2+1-3 = 1. Positive! You go outside despite the cold because not dealing with rain outweighs everything else.

You just simulated a perceptron - the simplest artificial neuron.

---

## Anatomy of a Perceptron

A perceptron has three main components:

**Inputs (x₁, x₂, x₃, ...)**: These are numbers representing features or measurements. For heart monitoring: heart rate, rhythm regularity, wave amplitudes, etc. For image recognition: pixel brightness values. For our going-outside example: weather and social factors.

**Weights (w₁, w₂, w₃, ...)**: These are numbers showing the importance of each input. Large positive weights mean the input strongly encourages the output. Large negative weights mean it strongly discourages. Small weights mean the input barely matters.

**Activation Function**: This is the decision rule. The perceptron multiplies each input by its corresponding weight, adds them all up, and feeds the sum to the activation function. The activation function decides: should this neuron "fire" (output 1) or stay silent (output 0)?

Let's write this mathematically. Don't worry - it's simpler than it looks:

**Sum = (x₁ × w₁) + (x₂ × w₂) + (x₃ × w₃) + ...**

Then apply the activation function to this sum to get the output.

> **Did You Know?**
>
> The perceptron was invented in 1958 by Frank Rosenblatt, a psychologist and computer scientist. His Mark 1 Perceptron machine could learn to recognize simple shapes. The New York Times called it "the embryo of an electronic computer that [the Navy] expects will be able to walk, talk, see, write, reproduce itself and be conscious of its existence."

---

## The Activation Function: Making the Decision

The sum of weighted inputs is just a number - could be anything from negative infinity to positive infinity. The activation function converts this number into a decision.

**Step Function (Original Perceptron)**:
- If sum > 0: Output = 1 (neuron fires)
- If sum ≤ 0: Output = 0 (neuron stays silent)

This is the simplest activation function. It's like a light switch - either on or off, nothing in between.

**Sigmoid Function (Modern Version)**:
Instead of an abrupt jump from 0 to 1, the sigmoid function creates a smooth S-shaped curve. Large negative sums give outputs near 0. Large positive sums give outputs near 1. Sums near zero give outputs around 0.5.

This smoothness makes learning easier (we'll see why in Chapter 5). It also provides a confidence measure: an output of 0.9 shows more certainty than 0.6.

**ReLU (Rectified Linear Unit)**:
This modern favorite is even simpler:
- If sum > 0: Output = sum
- If sum ≤ 0: Output = 0

ReLU is computationally efficient and works well in deep networks. It's like a valve that either blocks flow (negative inputs) or allows it through proportionally (positive inputs).

Different activation functions suit different purposes. The key idea remains the same: convert the weighted sum into an output.

---

## A Concrete Example: Detecting Heart Problems

Let's build a perceptron to detect abnormal heart rhythms from simplified ECG data. We'll use three inputs:

- **x₁**: Heart rate (beats per minute, scaled so 60-100 is normal = 0, higher is positive, lower is negative)
- **x₂**: Rhythm irregularity (0 = regular, 1 = irregular)
- **x₃**: Wave amplitude (scaled so normal = 0, abnormal is positive or negative)

After training (we'll learn how in Chapter 5), our perceptron learned these weights:
- **w₁ = 0.3**: Moderate weight for heart rate
- **w₂ = 0.8**: High weight for irregularity - very important!
- **w₃ = 0.4**: Moderate weight for amplitude

Let's test two patients:

**Patient A (Normal)**:
- Heart rate: 75 bpm → scaled to x₁ = 0 (normal)
- Rhythm: Regular → x₂ = 0
- Amplitude: Normal → x₃ = 0

Sum = (0 × 0.3) + (0 × 0.8) + (0 × 0.4) = 0

If we use a sigmoid activation and include a bias (threshold) of -0.5, we get:
Output = sigmoid(0 - 0.5) = 0.38

This is below 0.5, so we classify as "normal" ✓

**Patient B (Abnormal)**:
- Heart rate: 140 bpm → scaled to x₁ = 1.5 (high)
- Rhythm: Irregular → x₂ = 1
- Amplitude: High → x₃ = 0.8

Sum = (1.5 × 0.3) + (1 × 0.8) + (0.8 × 0.4) = 0.45 + 0.8 + 0.32 = 1.57

Output = sigmoid(1.57 - 0.5) = sigmoid(1.07) = 0.74

This is above 0.5, so we classify as "abnormal" ✓

The perceptron correctly identified the abnormal rhythm, weighted most heavily by the irregularity factor.

---

## The Bias: Setting the Threshold

Notice we included a "bias" term in that example. What's that about?

The bias shifts the decision boundary. Without it, the perceptron only fires when the weighted sum of inputs is positive. But what if you want it to fire only when the sum exceeds 2? Or allow it to fire even when the sum is slightly negative?

The bias is like a threshold you can adjust. Mathematically, it's just another weight, but its input is always 1:

**Sum = (x₁ × w₁) + (x₂ × w₂) + ... + (1 × bias)**

Or equivalently: **Sum = weighted_inputs + bias**

A positive bias makes the neuron more likely to fire. A negative bias makes it more conservative, requiring stronger input signals.

In our heart monitoring example, the bias of -0.5 means we need a positive weighted sum of at least 0.5 to classify as abnormal. This sets how sensitive our detector is. More negative bias = fewer false alarms, but might miss some real problems. Less negative bias = catches more problems, but more false alarms.

Adjusting the bias-weight tradeoff is crucial for different applications.

---

## Visual Understanding: Decision Boundaries

Here's a powerful way to visualize what a perceptron does: it draws a line (or hyperplane in higher dimensions) separating two categories.

Imagine a simple perceptron with two inputs: x₁ and x₂. You can plot all possible inputs on a 2D graph, with x₁ on the horizontal axis and x₂ on the vertical axis.

The perceptron's decision rule creates a line on this graph:
**w₁×x₁ + w₂×x₂ + bias = 0**

Points on one side of this line get classified as 0 (negative class). Points on the other side get classified as 1 (positive class). The line itself is the decision boundary.

For heart monitoring with heart rate and irregularity:
- The perceptron draws a line in "heart rate vs. irregularity" space
- Normal hearts cluster on one side
- Abnormal hearts cluster on the other side
- The perceptron learned where to draw the line by seeing many examples

This visualization reveals an important limitation: a single perceptron can only separate categories that are linearly separable. If the categories are mixed together in a complex pattern, one straight line can't separate them. You need multiple perceptrons working together - which is exactly what neural networks do!

> **Did You Know?**
>
> The famous "XOR problem" showed that a single perceptron cannot solve certain simple problems. XOR (exclusive OR) means "one or the other, but not both." A single perceptron can learn AND or OR, but not XOR. This limitation led to the development of multi-layer neural networks in the 1980s.

---

## Learning: Adjusting the Weights

So far, we've assumed our perceptron already has good weights. But how does it learn them?

The learning algorithm is beautifully simple:

1. **Make a prediction** using current weights
2. **Check if it's correct** by comparing to the true label
3. **If wrong, adjust weights** in the direction that would have given the correct answer
4. **Repeat** with many examples until weights stabilize

Specifically, for each training example:
- If prediction is too low: increase weights for positive inputs, decrease for negative
- If prediction is too high: decrease weights for positive inputs, increase for negative

The size of adjustment depends on:
- **How wrong** the prediction was (larger errors → bigger adjustments)
- **Learning rate** (a parameter controlling how fast we learn)

After seeing thousands of examples, with corrections each time, the weights gradually settle into values that work well for most cases.

This is supervised learning in action - the same process from Chapter 1, now with specific mathematics.

---

## From One Neuron to Many: Neural Networks

A single perceptron is powerful but limited. It can only draw one straight line to separate categories. Real-world problems require more complex decision boundaries.

The solution: connect many perceptrons together in layers.

**Input Layer**: Receives the raw data (pixel values, sensor readings, etc.)

**Hidden Layer(s)**: Perceptrons that process the inputs. Each hidden neuron looks for a different pattern. One might detect "high heart rate AND irregular rhythm." Another might detect "normal rate BUT abnormal wave shape." These combine simple features into complex patterns.

**Output Layer**: Makes the final decision based on hidden layer outputs. For binary classification (normal/abnormal), you might have one output neuron. For multi-class (normal/afib/vtach/other), you'd have multiple output neurons, one per category.

Each connection between neurons has its own weight. A network with three layers of 100 neurons each has roughly 10,000 connections - 10,000 weights to learn!

The same learning principle applies: show examples, make predictions, calculate errors, adjust weights. But the math becomes more complex when errors must propagate backward through multiple layers. That's called backpropagation, which we'll explore in Chapter 5.

---

## Why Artificial Neurons Work

Artificial neurons are ridiculously simple compared to biological neurons:

**Biological**: Living cells, thousands of inputs, chemical signaling, gene expression, metabolism, growth, adaptation

**Artificial**: Just multiplication, addition, and a simple decision function

Yet artificial neurons work! Why?

Because they capture the essential principle: weighted combination of inputs producing an output. They simplify away biological complexity that isn't needed for computation.

It's like the difference between a bird and an airplane. Birds flap wings, have feathers, need to eat, and breathe. Airplanes use jet engines, have metal bodies, need fuel, and have no biological processes. Yet both fly because they both use the essential principle: generate lift with wings.

We don't need to replicate biology exactly. We need to implement the same functional principle in a form that works with our technology.

Artificial neurons work with electricity and computer memory. Biological neurons work with chemistry and living tissue. Different implementations, same basic idea: process inputs to produce outputs based on learned patterns.

---

## Limitations of Simple Perceptrons

Before we get too excited, let's acknowledge what simple perceptrons can't do:

**Non-linear Problems**: Single perceptrons can only draw straight decision boundaries. Many real problems have curved, complex boundaries requiring multiple neurons.

**Complex Features**: Perceptrons work with the features you give them. If the important pattern isn't in your input features, the perceptron won't find it. Deep learning (many layers) can learn complex features automatically.

**Temporal Patterns**: Basic perceptrons process inputs at one moment. They don't remember previous inputs or capture patterns over time. Recurrent networks solve this.

**Context and Meaning**: Perceptrons manipulate numbers. They don't understand what those numbers mean. A perceptron can classify ECG signals without understanding hearts, electricity, or medicine.

**Generalization Limits**: Perceptrons might memorize training data without learning general principles. Careful training and testing (Chapter 6) helps but doesn't eliminate this risk.

Despite these limitations, perceptrons remain fundamental. Modern deep learning uses modified perceptrons with better activation functions, clever connection patterns, and multiple layers. But the basic principle - weighted inputs, activation function, learned weights - remains the same.

---

## From Perceptrons to Modern AI

The journey from Rosenblatt's 1958 perceptron to modern AI involved several key advances:

**Multi-Layer Networks (1980s)**: Connecting perceptrons in multiple layers enabled learning complex patterns. Backpropagation made training these networks feasible.

**Better Activation Functions (1990s-2000s)**: ReLU and other functions improved learning speed and effectiveness.

**Deep Learning (2010s)**: Networks with many layers (hence "deep") achieve superhuman performance on image recognition, speech, translation, and more. GPUs made training these massive networks practical.

**Specialized Architectures (2010s-present)**: Convolutional networks for images, recurrent networks for sequences, transformer networks for language, and spiking networks (Chapter 4) for efficiency.

Each advance built on the perceptron foundation. The basic unit remains recognizable: inputs, weights, activation function, output. We've added layers, refined the math, and scaled to billions of connections, but the core concept persists.

---

## Why This Matters for Biosensing

Delta-Predictive-Biosensing uses artificial neurons throughout:

**ECG Classification**: Multiple neurons learn to recognize patterns like atrial fibrillation, ventricular tachycardia, and normal rhythms from ECG waveforms.

**Feature Detection**: Early layers detect simple features (wave peaks, intervals). Deeper layers combine these into complex patterns (dangerous rhythms).

**Real-time Processing**: Efficient neuron implementations run on wearable devices, analyzing your biosignals continuously without draining the battery.

**Adaptive Learning**: Neurons can retrain on your personal data, adapting to your unique biological patterns.

Understanding artificial neurons helps you understand what these systems can and can't do. They're pattern recognizers, not magic. They learn from data, so their performance depends on training quality. They make probabilistic decisions, so they're never 100% certain.

This knowledge empowers you to use AI-based biosensing tools effectively and critically.

---

## What's Next?

Standard artificial neurons use numbers - continuous values representing activation levels. But biological neurons use spikes - brief electrical pulses. Why the difference?

Turns out, spike-based communication has major advantages: energy efficiency, natural handling of time-based patterns, and better noise tolerance. Researchers developed spiking neural networks (SNNs) that communicate with spikes instead of numbers.

Chapter 4 explores SNNs. You'll discover why timing matters, how spike-based coding works, and why SNNs are particularly good for processing biosignals. We're moving from simplified artificial neurons to neurons that work more like biology.

The journey continues from simple perceptrons to sophisticated spiking networks that might revolutionize AI.

---

**Chapter 3 Summary**: The perceptron is the simplest artificial neuron, taking weighted inputs, summing them, and applying an activation function to produce an output. Weights determine input importance and are learned from training data. The bias sets the decision threshold. Perceptrons create linear decision boundaries, separating categories with straight lines. Multiple perceptrons connected in layers form neural networks capable of learning complex patterns. While much simpler than biological neurons, perceptrons capture the essential principle of weighted input combination. Learning means adjusting weights based on prediction errors. Modern AI builds on this foundation with deeper networks, better activation functions, and specialized architectures.

**Next**: [Chapter 4: Spiking Neural Networks](ch04_spiking_networks.md)
