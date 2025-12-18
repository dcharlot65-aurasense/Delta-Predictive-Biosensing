# Chapter 7: Making Predictions

## Your AI Is Ready—Now What?

You've collected data, trained your model, and tested it thoroughly. The accuracy looks good. Now comes the moment of truth: using your AI to make real predictions about new data it has never seen before.

It's like finally taking your driver's test after months of practice. You've learned the rules, practiced parallel parking, and driven around town with an instructor. But when you get behind the wheel for the actual test, it's different—this time, it counts.

When your AI makes predictions on real-world biosignals, the stakes can be even higher. A wrong prediction might mean a missed health warning or an unnecessary doctor's visit. So let's understand exactly how predictions work and when to trust them.

## Two Types of Predictions: Categories vs. Numbers

AI predictions generally fall into two categories, and understanding the difference is crucial.

### Classification: Which Category?

**Classification** is when your AI sorts things into groups or categories. It's answering questions like:
- "Is this heartbeat normal or abnormal?"
- "Cat or dog?"
- "Is the person awake, in light sleep, or deep sleep?"

Think of classification like sorting laundry. You have distinct piles: whites, darks, delicates. Each item goes into exactly one pile. There's no "halfway between whites and darks"—you make a decision.

**Real biosensing example:** Your smartwatch detects an irregular heart rhythm. It classifies your current heartbeat as one of these categories:
- Normal sinus rhythm
- Atrial fibrillation (irregular rhythm)
- Premature ventricular contraction (early heartbeat)
- Tachycardia (too fast)

The AI looks at the shape, timing, and pattern of your heartbeat and picks the category that best matches what it learned during training.

### Regression: What Number?

**Regression** is when your AI predicts a specific number or value. It's answering questions like:
- "What will the temperature be tomorrow?" (Maybe 72.3 degrees)
- "What's this person's stress level?" (Perhaps 6.8 on a scale of 1-10)
- "How many hours of deep sleep will they get tonight?" (Could be 1.7 hours)

Think of regression like guessing someone's height. You don't just say "tall" or "short" (that would be classification). You estimate an actual number: "I think they're about 5 feet 8 inches."

**Real biosensing example:** Your fitness tracker predicts how many calories you burned during your workout. It doesn't just say "a lot" or "a little." It gives you a specific number: "You burned 347 calories." This prediction comes from your heart rate data, movement patterns, and what the AI learned about exercise intensity.

> **Did You Know?**
> Some problems can be either classification or regression depending on how you frame them! Predicting heart rate could be regression ("your heart rate is 78 bpm") or classification ("your heart rate is normal/elevated/high").

## How Predictions Actually Work

Let's follow a prediction from start to finish using a sleep stage detector as our example.

**Step 1: Gather Input Data**
Your sleep tracker collects 30 seconds of data:
- Brain wave patterns (EEG)
- Eye movements
- Muscle tension
- Heart rate variability

**Step 2: Preprocess the Data**
The raw signals get cleaned up and transformed into features the AI understands:
- Frequency of different brain waves (alpha, beta, delta, theta)
- Number of eye movements
- Average heart rate
- Variability in muscle activity

**Step 3: Feed Through the Model**
The preprocessed data flows through the trained neural network. Think of this like water flowing through a pipe system—the data enters, passes through layers of calculations, and emerges transformed at the other end.

**Step 4: Generate Probabilities**
For our sleep stage classifier, the output might look like this:
- Wake: 5%
- Light Sleep: 23%
- Deep Sleep: 71%
- REM Sleep: 1%

The AI is saying "I'm 71% confident this is deep sleep."

**Step 5: Make the Final Prediction**
The AI picks the category with the highest probability: **Deep Sleep**

But notice something important—it also knows how confident it is in that answer. This leads us to one of the most crucial concepts in AI predictions.

## Confidence: How Sure Is It?

Every prediction comes with a confidence level, even if it's not always displayed to you. Understanding confidence is critical for knowing when to trust AI predictions.

**High Confidence (90%+)**
The AI is very sure. The input data strongly matches patterns it learned during training. For a heart rhythm classifier, this might mean:
- Clear, strong signal
- Textbook pattern that's easy to recognize
- Similar to many training examples

**Medium Confidence (60-90%)**
The AI thinks it's right but isn't certain. The data might be:
- Slightly noisy or unclear
- A borderline case between two categories
- Less common but still recognizable

**Low Confidence (<60%)**
The AI is basically guessing. This could mean:
- Very noisy data
- A pattern the AI rarely saw during training
- Something genuinely ambiguous

Here's the key insight: **A prediction is only as trustworthy as its confidence level.**

If your sleep tracker is only 55% confident you're in deep sleep (meaning it's almost equally likely you're in light sleep), you should take that prediction with a grain of salt. But if it's 95% confident, that's much more reliable.

> **Did You Know?**
> Some AI systems are "overconfident"—they give high confidence scores even when they're wrong! This is why testing and calibration are so important. Good AI developers adjust their models to make confidence scores more accurate.

## When to Trust Predictions

So when should you trust what an AI tells you? Here's a practical framework:

### GREEN LIGHT: Trust the Prediction
- **High confidence** (>90%)
- **Clear, quality data** (strong signal, no noise)
- **Common scenario** the AI trained on extensively
- **Low stakes** if wrong (like a step counter being slightly off)
- **Aligns with other information** (matches what you expect or other sensors confirm)

**Example:** Your fitness band says you walked 8,247 steps today with 95% confidence. The GPS confirms you walked about 4 miles. The prediction makes sense—trust it!

### YELLOW LIGHT: Use with Caution
- **Medium confidence** (60-90%)
- **Some data quality issues** (occasional noise)
- **Borderline case** between categories
- **Moderate stakes** if wrong
- **No other information** to confirm or deny

**Example:** Your sleep tracker says you had 45 minutes of REM sleep but only 68% confident. You might note this as approximate but don't make major life decisions based on this one number.

### RED LIGHT: Don't Trust Alone
- **Low confidence** (<60%)
- **Poor data quality** (lots of noise, sensor issues)
- **Unusual pattern** the AI rarely encountered
- **High stakes** if wrong (medical decisions)
- **Contradicts other evidence**

**Example:** Your heart monitor flags a dangerous rhythm but only 58% confident, and the signal was noisy because you were moving around. Don't panic, but do check with a doctor if you have symptoms.

**The Golden Rule:** For high-stakes decisions (especially medical ones), AI should inform human decisions, not replace them. Your doctor should see both the AI prediction AND the raw data, plus consider your symptoms, history, and other factors.

## Real Example: Heart Attack Risk Prediction

Let's see how predictions work in a real, high-stakes scenario.

**The System:** An AI model predicts heart attack risk based on:
- Heart rate variability
- Blood pressure patterns
- Activity level
- Age and medical history
- Previous ECG readings

**The Input:** A 55-year-old patient's data from their smartwatch over the past week.

**The Prediction:**
- Classification: "Elevated Risk"
- Regression: "23% probability of cardiac event in next 12 months"
- Confidence: 82%

**How to Use This:**

What this SHOULD do:
- Prompt the patient to schedule a doctor's appointment
- Trigger additional monitoring
- Alert healthcare providers to review the data
- Encourage the patient to avoid high-stress activities until checked

What this should NOT do:
- Cause panic (23% means 77% chance of being fine)
- Replace a full cardiac workup
- Lead to self-diagnosis
- Result in taking medications without doctor approval

The prediction is a valuable early warning system, but it's the start of a conversation with healthcare providers, not the final answer.

## Real Example: Sleep Stage Detection

Here's a more everyday example that shows how predictions can still be incredibly useful even when they're not perfect.

**The System:** Your sleep tracker classifies each 30-second period of the night into sleep stages.

**One Night's Predictions:**
- Wake: 45 minutes
- Light Sleep: 4.2 hours
- Deep Sleep: 1.8 hours
- REM Sleep: 1.5 hours

**Confidence Levels:**
- Average: 84%
- Range: 62% (some borderline periods) to 98% (obvious deep sleep)

**How to Use This:**

The exact numbers might be off by 15-20%, but the overall pattern is valuable:
- You can track trends over time (Are you getting more deep sleep this week?)
- Compare weekdays to weekends
- See if changes in routine affect sleep quality
- Notice if a new medication disrupts REM sleep

You don't need perfect accuracy to gain insights. If the tracker shows you got almost no deep sleep three nights in a row, that's worth paying attention to—even if the exact amounts are approximate.

## Predictions in the Real World: Limitations and Possibilities

Understanding what AI can and can't do helps you use it effectively.

**What Modern AI Does Well:**
- Recognizing patterns in clean, consistent data
- Classifying common, well-defined categories
- Making predictions similar to training examples
- Processing vast amounts of data quickly
- Detecting subtle patterns humans might miss

**What AI Struggles With:**
- Novel situations it never trained on
- Noisy, inconsistent data
- Rare events with few training examples
- Understanding context the way humans do
- Explaining *why* it made a prediction (we'll explore this in the next chapter!)

The future of AI predictions isn't about replacing human judgment—it's about augmenting it. The AI processes the data and flags patterns, while humans provide context, make final decisions, and handle unusual cases.

## What's Next?

You now understand how AI makes predictions, from classification to regression, and when to trust them. But there's still a mystery: How do you know *why* the AI made a particular prediction?

In the next chapter, we'll dive into the "black box problem" and explore the cutting edge of explainable AI. You'll learn how scientists are working to make AI show its work, just like your math teacher always asked you to do. This is crucial for building trust, especially in medical applications where understanding the "why" can be as important as knowing the "what."

Get ready to peek inside the black box!

---

**Key Takeaways:**
- Classification sorts things into categories; regression predicts specific numbers
- Every prediction has a confidence level that tells you how sure the AI is
- Trust predictions when confidence is high, data is clean, and stakes are low
- For high-stakes decisions, use AI to inform human experts, not replace them
- Perfect accuracy isn't always necessary—patterns and trends can be valuable too
- Understanding limitations helps you use AI predictions effectively and responsibly
