# Glossary of Biosignal Terms

## How to Use This Glossary

Welcome! This glossary helps you understand the terms used in biosignal analysis. Whether you're just starting out or need a quick refresher, you'll find clear definitions and real-world examples here.

**What You'll Find**:
- Terms are organized alphabetically (A-Z)
- Each entry includes: Definition, Example, and Related Terms
- Icons show which category each term belongs to
- Analogies help connect new ideas to things you already know

**Categories with Icons**:
- 🫀 **Body & Biology** - Parts of your body and how they work
- 📊 **Signals & Measurement** - How we capture and record body signals
- 🔢 **Math & Statistics** - Numbers and patterns in data
- 🤖 **Computers & AI** - How machines learn and make decisions
- ⚕️ **Medical & Clinical** - Healthcare and diagnosis

---

## A

### Accuracy
**Category**: 🤖 Computers & AI

**Simple Definition**: Accuracy measures how often a computer model makes correct predictions. If a model is right 9 times out of 10, its accuracy is 90%. Higher accuracy means the model makes fewer mistakes.

**Example**: Imagine a heartbeat detector that checks 100 heartbeats. If it correctly identifies 95 of them and misses 5, its accuracy is 95%. That's pretty good!

**See Also**: Precision, Recall, Validation

### Action Potential
**Category**: 🫀 Body & Biology

**Simple Definition**: An action potential is a quick electrical pulse that travels along a nerve or muscle cell. It's how your body sends messages from one place to another. The pulse lasts less than a millisecond and moves very fast.

**Example**: When you touch something hot, an action potential races from your finger to your brain at over 100 meters per second. That's faster than a car on the highway! Your brain gets the message "hot!" and tells your hand to pull away.

**See Also**: Neuron, Axon, Membrane potential, Synapse

### Algorithm
**Category**: 🔢 Math & Statistics

**Simple Definition**: An algorithm is a set of step-by-step instructions to solve a problem. It's like a recipe that a computer follows to complete a task. Good algorithms are clear, accurate, and efficient.

**Example**: To find your heart rate, an algorithm might: (1) Look at the ECG signal, (2) Find each heartbeat peak, (3) Count the peaks in 60 seconds, (4) Report the number as beats per minute. Follow those steps, and you get the answer!

**See Also**: Machine learning, Model, Training

### Amplitude
**Category**: 📊 Signals & Measurement

**Simple Definition**: Amplitude is the height or strength of a signal. In biosignals, it shows how strong the electrical activity is. Larger amplitude means stronger activity, while smaller amplitude means weaker activity.

**Example**: When you flex your bicep muscle hard, the EMG signal shows large amplitude. When you relax, the amplitude drops almost to zero. It's like the volume knob on a speaker - loud music has high amplitude, soft music has low amplitude.

**See Also**: Signal, Peak, Waveform

### Arrhythmia
**Category**: ⚕️ Medical & Clinical

**Simple Definition**: An arrhythmia is an irregular heartbeat pattern. Your heart might beat too fast, too slow, or with an uneven rhythm. Some arrhythmias are harmless, but others need medical attention.

**Example**: A healthy heart beats in a steady pattern: "lub-dub, lub-dub, lub-dub." An arrhythmia might sound like: "lub-dub, lub-dub, lub...lub-dub, lub-dub-dub." The timing is off, which an ECG can detect.

**See Also**: Bradycardia, Tachycardia, ECG, Cardiac cycle

### Artifact
**Category**: 📊 Signals & Measurement

**Simple Definition**: An artifact is unwanted noise or distortion in a signal that doesn't come from the body. Artifacts can be caused by movement, loose wires, or electrical interference. We usually want to remove them during data cleaning.

**Example**: You're recording an ECG, and someone's phone rings nearby. The electrical interference from the phone creates a spike in your recording that looks nothing like a heartbeat. That's an artifact - it's not real heart activity.

**See Also**: Noise, Baseline, Filter, Signal

### Artificial Intelligence
**Category**: 🤖 Computers & AI

**Simple Definition**: Artificial Intelligence (AI) is when computers perform tasks that usually require human thinking, like recognizing patterns, making decisions, or learning from experience. AI can find things in data that would take humans years to discover.

**Example**: An AI system can look at thousands of EEG recordings and learn to spot the patterns that happen before a seizure. Once trained, it can warn patients minutes before a seizure begins - something even expert doctors might miss.

**See Also**: Machine learning, Neural network, Deep learning, Training

### Autonomic Nervous System
**Category**: 🫀 Body & Biology

**Simple Definition**: The autonomic nervous system controls body functions that happen automatically without you thinking about them. It manages your heartbeat, breathing, digestion, and more. It has two parts that work opposite each other: sympathetic (speeds things up) and parasympathetic (slows things down).

**Example**: When you're scared, your autonomic nervous system makes your heart beat faster, your breathing quicken, and your palms sweat. You don't decide to do this - it happens automatically to prepare you for danger.

**See Also**: Sympathetic, Parasympathetic, Heart rate, Central nervous system

### Average (Mean)
**Category**: 🔢 Math & Statistics

**Simple Definition**: The average, also called the mean, is the sum of all numbers divided by how many numbers there are. It tells you the typical or middle value in a group of numbers.

**Example**: If your heart rate readings are 72, 75, 68, 70, and 75 beats per minute, the average is (72+75+68+70+75) ÷ 5 = 72 beats per minute. This gives you a single number that represents your typical heart rate.

**See Also**: Median, Standard deviation, Distribution

### Axon
**Category**: 🫀 Body & Biology

**Simple Definition**: An axon is a long, thin fiber that extends from a neuron's body. It carries electrical signals away from the neuron to other cells. Some axons are over a meter long, making them the longest cells in your body.

**Example**: Think of a neuron as a tree. The axon is like a long branch that reaches out to connect with other trees. When the neuron fires, an electrical signal zooms down the axon like a bead sliding down a wire, carrying the message to the next neuron.

**See Also**: Neuron, Dendrite, Soma, Action potential

---

## B

### Baseline
**Category**: 📊 Signals & Measurement

**Simple Definition**: The baseline is the normal, resting level of a signal when there's no special activity. It's the flat part of the signal that everything else is measured against. Signals go up and down from this baseline.

**Example**: In an ECG, the baseline is the flat line between heartbeats. When your heart contracts, the signal jumps up from the baseline and then returns. If the baseline drifts up or down (maybe from breathing), we might need to correct it.

**See Also**: Amplitude, Signal, Artifact, Filter

### Batch
**Category**: 🤖 Computers & AI

**Simple Definition**: A batch is a group of data examples that are processed together during machine learning training. Instead of learning from one example at a time, computers learn from batches to be more efficient and accurate.

**Example**: Imagine teaching someone to recognize different dog breeds. Instead of showing one picture at a time, you show them 32 pictures together (a batch). They look at all 32, learn the patterns, and then you show them the next batch of 32. It's faster than one-by-one learning.

**See Also**: Training, Dataset, Epoch (training), Machine learning

### Biomarker
**Category**: ⚕️ Medical & Clinical

**Simple Definition**: A biomarker is a measurable sign that indicates a health condition or disease. Biomarkers can be found in signals, blood tests, or other measurements. They help doctors diagnose problems and track treatment progress.

**Example**: Heart rate variability is a biomarker for stress and overall health. If your HRV decreases over time, it might indicate increasing stress or declining fitness. Doctors use this biomarker to understand what's happening in your body.

**See Also**: Heart rate variability, Diagnosis, Vital signs

### Bradycardia
**Category**: ⚕️ Medical & Clinical

**Simple Definition**: Bradycardia means a slower than normal heart rate, typically below 60 beats per minute in adults at rest. It can be normal for athletes, but it might also signal a heart problem that needs checking.

**Example**: Most people's resting heart rate is 60-100 beats per minute. An Olympic marathon runner might have a resting heart rate of 45 beats per minute due to their super-fit heart. That's bradycardia, but it's healthy. However, if someone suddenly develops a heart rate of 45 with dizziness, that needs medical attention.

**See Also**: Tachycardia, Heart rate, Arrhythmia, ECG

---

## C

### Cardiac Cycle
**Category**: 🫀 Body & Biology

**Simple Definition**: The cardiac cycle is one complete heartbeat, from the beginning of one beat to the beginning of the next. It includes all the stages of the heart filling with blood and pumping it out. One cycle takes about 0.8 seconds at a normal heart rate.

**Example**: During one cardiac cycle: First, your heart chambers relax and fill with blood (like filling water balloons). Then they squeeze hard to pump blood out to your lungs and body (like squeezing the balloons). Then the cycle starts again. This happens about 72 times every minute!

**See Also**: ECG, Ventricle, Sinoatrial node, Heart rate

### Central Nervous System
**Category**: 🫀 Body & Biology

**Simple Definition**: The central nervous system (CNS) consists of your brain and spinal cord. It's the command center that processes information, makes decisions, and controls your body. All other nerves connect to it.

**Example**: Think of your CNS as the main computer server in a building. Your brain is like the powerful processor that thinks and decides. Your spinal cord is like the main cable that connects the processor to all the other computers (your body parts). Messages flow in and out constantly.

**See Also**: Peripheral nervous system, Brain, Neuron, EEG

### Channel
**Category**: 📊 Signals & Measurement

**Simple Definition**: A channel is one recording pathway or sensor in a measurement system. Multi-channel systems can record from several locations at once. Each channel captures data from a different spot on the body.

**Example**: A sleep study EEG might use 8 channels - one on your forehead, one on each side of your head, and several more around your scalp. Each channel records brain activity from its own location. Together, they create a complete picture of what your brain is doing.

**See Also**: Montage, Electrode, Lead, Sensor

### Classification
**Category**: 🤖 Computers & AI

**Simple Definition**: Classification is sorting things into categories or groups. In machine learning, a computer learns to classify new examples based on patterns it learned from training data. It's like teaching a computer to put things in the right box.

**Example**: A sleep classification system learns to recognize patterns in EEG signals. When you sleep, it automatically classifies each 30-second window as "Awake," "Light Sleep," "Deep Sleep," or "REM Sleep." At the end of the night, you get a full sleep report.

**See Also**: Machine learning, Prediction, Label, Training

### Correlation
**Category**: 🔢 Math & Statistics

**Simple Definition**: Correlation measures how two things change together. Strong positive correlation means when one goes up, the other goes up too. Negative correlation means when one goes up, the other goes down. Zero correlation means they don't affect each other.

**Example**: Heart rate and breathing rate have positive correlation - when you exercise harder and your heart beats faster, you also breathe faster. But heart rate and heart rate variability have negative correlation - when your heart beats faster, the variation between beats usually gets smaller.

**See Also**: Data point, Time series, Pattern

---

## D

### Data Point
**Category**: 🔢 Math & Statistics

**Simple Definition**: A data point is a single measurement or value in a dataset. When you connect many data points together, they form a signal or graph. Each point represents one moment in time or one observation.

**Example**: If you measure your heart rate every second for a minute, you get 60 data points. Each point is one measurement: 72 bpm, 73 bpm, 72 bpm, and so on. Plot all 60 points on a graph, and you can see how your heart rate changed during that minute.

**See Also**: Time series, Signal, Dataset, Sample rate

### Dataset
**Category**: 🤖 Computers & AI

**Simple Definition**: A dataset is a collection of related data used for analysis or training machine learning models. It includes many examples that help computers learn patterns. Good datasets are large, accurate, and well-organized.

**Example**: A seizure detection dataset might contain EEG recordings from 500 patients, including both normal brain activity and seizure events. Each recording is labeled, so the computer knows which parts show seizures. This dataset teaches the AI what seizures look like in EEG signals.

**See Also**: Training, Validation, Label, Batch

### Deep Learning
**Category**: 🤖 Computers & AI

**Simple Definition**: Deep learning uses artificial neural networks with many layers to learn complex patterns in data. "Deep" refers to having many layers stacked together. These systems can learn features automatically without humans specifying what to look for.

**Example**: A deep learning system for ECG analysis might have 20 layers. The first layers learn to detect simple things like "is this going up or down?" Middle layers learn to recognize P-waves and QRS complexes. Deep layers learn to identify specific heart conditions. Each layer builds on what the previous layers learned.

**See Also**: Neural network, Machine learning, Artificial intelligence, Training

### Dendrite
**Category**: 🫀 Body & Biology

**Simple Definition**: Dendrites are short, branching fibers that extend from a neuron's body. They receive signals from other neurons and carry them toward the cell body. One neuron can have thousands of dendrites.

**Example**: If a neuron is like a tree, dendrites are the small branches and twigs that catch falling leaves (signals) from other trees. Each dendrite receives messages from other neurons. The neuron adds up all these messages and decides whether to fire its own signal down the axon.

**See Also**: Neuron, Axon, Soma, Synapse

### Diagnosis
**Category**: ⚕️ Medical & Clinical

**Simple Definition**: Diagnosis is the process of identifying a disease or condition based on symptoms, tests, and other information. Doctors use biosignals as one tool to help make accurate diagnoses.

**Example**: A patient feels chest pain and shortness of breath. The doctor records an ECG, which shows an abnormal pattern called ST-elevation. Combined with symptoms and other tests, the doctor diagnoses a heart attack and immediately starts treatment. The ECG signal was key to the diagnosis.

**See Also**: Prognosis, Screening, Biomarker, Pathology

### Distribution
**Category**: 🔢 Math & Statistics

**Simple Definition**: A distribution shows how often different values appear in a dataset. It tells you which values are common and which are rare. Distributions can be visualized with graphs called histograms.

**Example**: If you measure 1,000 people's resting heart rates, most will be between 60-80 bpm (that's the middle of the distribution). Fewer people will have rates of 50 or 90 bpm (the edges of the distribution). Very few will be below 40 or above 100 (the tails). The distribution shows this pattern.

**See Also**: Histogram, Average, Standard deviation, Normative data

---

## E

### ECG/EKG
**Category**: 📊 Signals & Measurement

**Simple Definition**: ECG stands for electrocardiogram (EKG is the same thing, using the German spelling). It's a recording of your heart's electrical activity. Electrodes on your skin detect the tiny electrical signals that make your heart beat.

**Example**: During a doctor visit, they stick 10 electrodes on your chest, arms, and legs. These record how electrical waves spread through your heart with each beat. The ECG printout shows characteristic bumps called P-waves, QRS complexes, and T-waves. Doctors read these patterns to check your heart health.

**See Also**: Lead, Cardiac cycle, Arrhythmia, Electrode

### EDA
**Category**: 📊 Signals & Measurement

**Simple Definition**: EDA stands for electrodermal activity, also called skin conductance or galvanic skin response. It measures how well your skin conducts electricity, which changes with sweat. More sweat means higher conductance, usually indicating stress or excitement.

**Example**: You're watching a scary movie, and there's a big jump scare. Your palms instantly start sweating (even if you don't notice). EDA sensors on your fingers detect this as a sudden increase in conductance. The signal shows exactly when you got scared - even if you tried to hide it!

**See Also**: Sensor, Autonomic nervous system, Stress, Amplitude

### EEG
**Category**: 📊 Signals & Measurement

**Simple Definition**: EEG stands for electroencephalogram. It's a recording of your brain's electrical activity. Electrodes on your scalp pick up tiny electrical signals from millions of neurons firing together. Different brain states create different patterns.

**Example**: During a sleep study, you wear a cap with 20 EEG electrodes. While you're awake, the EEG shows fast, irregular waves. As you fall asleep, the waves slow down and get bigger. In deep sleep, you see huge, slow "delta waves." The EEG reveals which sleep stage you're in at every moment.

**See Also**: Montage, Epoch, Channel, Brain, Central nervous system

### Electrode
**Category**: 🫀 Body & Biology

**Simple Definition**: An electrode is a sensor that detects electrical signals from your body. Electrodes usually stick to your skin and connect to recording equipment with wires. They convert your body's electrical activity into signals that machines can measure.

**Example**: ECG electrodes are small sticky pads placed on your chest. Each electrode contains metal that touches your skin and gel that conducts electricity. When your heart creates an electrical wave, it spreads through your body to your skin. The electrodes pick up this tiny voltage - usually less than 1 millivolt!

**See Also**: Sensor, Lead, Channel, Impedance

### EMG
**Category**: 📊 Signals & Measurement

**Simple Definition**: EMG stands for electromyogram. It's a recording of electrical activity in your muscles. When you activate a muscle, the nerve signals that control it create electrical patterns. EMG electrodes detect these patterns to show how hard muscles are working.

**Example**: A physical therapist puts EMG electrodes on your thigh muscle. When you lift your leg, the EMG shows bursts of electrical activity - that's your muscle fibers firing. The bigger the bursts, the harder you're contracting. As your muscle gets tired, the EMG pattern changes, showing fatigue.

**See Also**: Motor neuron, Muscle fiber, Amplitude, Electrode

### Epoch
**Category**: 📊 Signals & Measurement

**Simple Definition**: An epoch is a time window or segment of a longer recording. Signals are often divided into epochs for analysis. In sleep studies, epochs are typically 30 seconds long. Each epoch can be analyzed or classified separately.

**Example**: You record 8 hours of sleep EEG - that's too much data to analyze all at once. Instead, you divide it into 960 epochs of 30 seconds each. You classify each epoch as a sleep stage: Wake, Light Sleep, Deep Sleep, or REM. At the end, you know exactly how much time you spent in each stage.

**See Also**: Window, Time series, Segmentation, EEG

### Epoch (Training)
**Category**: 🤖 Computers & AI

**Simple Definition**: In machine learning, an epoch is one complete pass through the entire training dataset. If you have 1,000 examples and train for 10 epochs, the model sees all 1,000 examples 10 times. More epochs usually means better learning, up to a point.

**Example**: You're training a heartbeat detector with 10,000 ECG examples. In Epoch 1, it's terrible - only 60% accurate. By Epoch 5, it's learned a lot - 85% accurate. By Epoch 10, it plateaus at 92% accurate. Each epoch, the model learned from its mistakes and got better.

**See Also**: Training, Batch, Overfitting, Machine learning

---

## F

### Feature
**Category**: 🤖 Computers & AI

**Simple Definition**: A feature is a measurable property or characteristic used for analysis or prediction. Features are the input variables that machine learning models use to make decisions. Good features capture important information about the data.

**Example**: For heart rate classification, useful features might include: average heart rate (72 bpm), heart rate variability (45 ms), maximum heart rate (95 bpm), and minimum heart rate (65 bpm). The AI uses these four features to decide if the heart rhythm is normal or abnormal.

**See Also**: Machine learning, Classification, Dataset, Training

### Filter
**Category**: 📊 Signals & Measurement

**Simple Definition**: A filter removes unwanted parts of a signal while keeping the parts you care about. Different filters do different jobs: high-pass filters remove slow drifts, low-pass filters remove fast noise, and band-pass filters keep only a specific frequency range.

**Example**: Your ECG recording has two problems: slow breathing movement (0.3 Hz) and fast electrical noise (60 Hz from power lines). You apply a band-pass filter that keeps only 0.5-40 Hz. This removes both the breathing artifact and the electrical noise, leaving a clean ECG with just heartbeat signals.

**See Also**: Noise, Artifact, Frequency, Signal

### Fourier Transform
**Category**: 🔢 Math & Statistics

**Simple Definition**: A Fourier transform converts a signal from the time domain (amplitude over time) into the frequency domain (showing which frequencies are present). It reveals hidden rhythms and cycles in data. It's named after mathematician Joseph Fourier.

**Example**: Your ECG looks like a complex, wiggly line over time. Apply a Fourier transform, and you get a graph showing the different frequencies: a big peak at 1.2 Hz (your 72 bpm heart rate), a smaller peak at 2.4 Hz (the second harmonic), and tiny peaks from noise. The transform reveals the underlying rhythms.

**See Also**: Frequency domain, Spectrum, Time domain, Frequency

### Frequency
**Category**: 📊 Signals & Measurement

**Simple Definition**: Frequency is how often something repeats per unit of time. It's measured in Hertz (Hz), which means "cycles per second." Higher frequency means faster repetition, while lower frequency means slower repetition.

**Example**: Your heart beats 72 times per minute, which equals 1.2 times per second, or 1.2 Hz. That's a low frequency. Meanwhile, "gamma" brain waves oscillate at 30-100 Hz - much faster. Musical notes also have frequency: Middle C is 261.6 Hz, meaning the air vibrates 261.6 times each second.

**See Also**: Hertz, Spectrum, Fourier transform, Frequency domain

### Frequency Domain
**Category**: 🔢 Math & Statistics

**Simple Definition**: The frequency domain is a way of looking at signals based on which frequencies are present, rather than how the signal changes over time. It shows which rhythms and cycles make up a signal. You convert from time domain to frequency domain using a Fourier transform.

**Example**: In the time domain, an ECG is a wavy line showing voltage over seconds. In the frequency domain, it's a graph showing peaks at different frequencies: the main heartbeat frequency, respiratory variations, and maybe some high-frequency noise. Each view reveals different information about the same signal.

**See Also**: Time domain, Fourier transform, Spectrum, Frequency

---

## G

### Gain
**Category**: 📊 Signals & Measurement

**Simple Definition**: Gain is amplification - it makes signals bigger so they're easier to measure and analyze. The gain setting controls how much the signal is multiplied. Higher gain makes small signals visible but might amplify noise too.

**Example**: Brain signals at the scalp are incredibly tiny - around 50 microvolts (0.00005 volts). That's way too small to see directly! An EEG amplifier uses a gain of 20,000 to multiply these signals, making them big enough to display on screen. The gain turns whispers into shouts.

**See Also**: Amplitude, Signal, Noise, EEG

---

## H

### Heart Rate
**Category**: 🫀 Body & Biology

**Simple Definition**: Heart rate is how many times your heart beats per minute (bpm). It varies based on activity, fitness, emotions, and health. Resting heart rate for adults typically ranges from 60-100 bpm, but athletes might have lower rates.

**Example**: Sitting at rest, your heart rate might be 70 bpm. Start jogging, and it climbs to 140 bpm. Sprint up a hill, and it might hit 180 bpm! Your heart speeds up to deliver more oxygen to working muscles. After exercise, it gradually slows back down to the resting rate.

**See Also**: Cardiac cycle, ECG, PPG, Bradycardia, Tachycardia

### Heart Rate Variability
**Category**: ⚕️ Medical & Clinical

**Simple Definition**: Heart rate variability (HRV) measures how much the time between heartbeats changes. A healthy heart doesn't beat like a metronome - the timing varies slightly with breathing, stress, and other factors. Higher HRV usually indicates better health and fitness.

**Example**: Your heart rate is 60 bpm on average, but that doesn't mean exactly 1.0 seconds between every beat. It might be 0.95 seconds, then 1.05 seconds, then 0.98 seconds, and so on. The variation is your HRV. Low HRV (beats are very regular) can indicate stress or illness. High HRV (more variation) usually means your body is healthy and adaptable.

**See Also**: Heart rate, ECG, Biomarker, Autonomic nervous system

### Hertz
**Category**: 📊 Signals & Measurement

**Simple Definition**: Hertz (Hz) is the unit for frequency, meaning "cycles per second." It's named after physicist Heinrich Hertz. When something repeats 5 times each second, its frequency is 5 Hz. Common signals range from less than 1 Hz to thousands of Hz.

**Example**: Brain waves come in different frequency bands: Delta waves (0.5-4 Hz) during deep sleep, Alpha waves (8-13 Hz) when you're relaxed with eyes closed, and Beta waves (13-30 Hz) when you're alert and thinking. Each band has its own frequency range measured in Hertz.

**See Also**: Frequency, Sample rate, Spectrum, EEG

### Histogram
**Category**: 🔢 Math & Statistics

**Simple Definition**: A histogram is a bar graph that shows how often different values appear in a dataset. The x-axis shows value ranges (called bins), and the height of each bar shows how many data points fall in that range. It visualizes the distribution of data.

**Example**: You collect heart rate data from 100 people. A histogram might show: 5 people with HR 40-50 bpm (short bar), 25 people with 50-60 bpm (medium bar), 40 people with 60-70 bpm (tall bar), 25 people with 70-80 bpm (medium bar), and 5 people with 80-90 bpm (short bar). You can instantly see that most people cluster around 60-70 bpm.

**See Also**: Distribution, Average, Standard deviation, Data point

---

## I

### Impedance
**Category**: 📊 Signals & Measurement

**Simple Definition**: Impedance is electrical resistance between an electrode and the body. High impedance means the signal has trouble getting through, causing noise and weak signals. Good contact with low impedance gives cleaner recordings.

**Example**: You attach EEG electrodes, but forget to add conductive gel. The dry electrodes have high impedance - over 100,000 ohms. The signals look terrible, full of noise. Add gel and gently scrub the skin. Impedance drops to 5,000 ohms. Now the signals are crystal clear. Low impedance = good connection!

**See Also**: Electrode, Signal, Noise, Gain

### Inference
**Category**: 🤖 Computers & AI

**Simple Definition**: Inference is when a trained machine learning model makes predictions on new data it hasn't seen before. After training is complete, the model is deployed for inference. It's like taking a test after studying - you apply what you learned.

**Example**: You train a seizure detection model using 1,000 hours of EEG data. Training takes days. Now the model is ready for inference. A new patient wears an EEG cap, and your model analyzes their brain waves in real-time, looking for seizure patterns. The inference happens in milliseconds - much faster than training!

**See Also**: Prediction, Training, Model, Machine learning

### Interpolation
**Category**: 🔢 Math & Statistics

**Simple Definition**: Interpolation is estimating missing values between known data points. If you know what happened at 1 second and 3 seconds, interpolation can estimate what happened at 2 seconds. It fills gaps in data based on surrounding values.

**Example**: Your heart rate monitor records every second: 72, 75, [missing], 81, 83 bpm. What was the missing value at second 3? Linear interpolation estimates (75+81)÷2 = 78 bpm. More sophisticated methods might consider the pattern and estimate 77 or 79 bpm. Either way, the gap is filled reasonably.

**See Also**: Data point, Time series, Artifact, Noise

---

## L

### Label
**Category**: 🤖 Computers & AI

**Simple Definition**: A label is the correct answer or category assigned to a training example. Labels tell the machine learning model what each example should be classified as. Without labels, supervised learning can't work - the model needs to know the right answers to learn.

**Example**: You're building a sleep stage classifier. Each 30-second EEG epoch needs a label: "Wake," "Light Sleep," "Deep Sleep," or "REM." A sleep expert manually labels 10,000 epochs. Now the computer can learn: "When the EEG looks like THIS, the label is Deep Sleep." Labels are the answer key.

**See Also**: Classification, Training, Dataset, Supervised learning

### Lead
**Category**: 📊 Signals & Measurement

**Simple Definition**: In ECG, a lead is a specific view of the heart's electrical activity, calculated from electrode positions. Standard ECG uses 12 leads, each showing the heart from a different angle. Different leads help doctors see different parts of the heart.

**Example**: Lead II looks at electrical activity from the right shoulder to the left leg - it shows a clear view of heartbeats. Lead V1 is positioned near the right side of the heart. If there's damage to that area, Lead V1 will show abnormalities that other leads might miss. Together, all 12 leads create a complete picture.

**See Also**: ECG, Electrode, Channel, Montage

### Longitudinal
**Category**: ⚕️ Medical & Clinical

**Simple Definition**: Longitudinal means tracking the same person or variable over a long period of time. Longitudinal studies record data at multiple time points to see how things change. They reveal trends, progression, and long-term effects.

**Example**: A longitudinal brain health study follows 100 elderly people for 10 years. They measure EEG and cognitive tests every 6 months. Over time, researchers can see whose brain activity declines and who stays sharp. They discover that people with higher alpha wave power at age 70 have better memory at age 80. Short-term studies would miss this pattern.

**See Also**: Screening, Biomarker, Diagnosis, Time series

---

## M

### Machine Learning
**Category**: 🤖 Computers & AI

**Simple Definition**: Machine learning is a type of AI where computers learn patterns from data instead of following explicit programmed rules. The computer improves its performance through experience. It's like learning to ride a bike - you get better through practice, not by memorizing instructions.

**Example**: Instead of programming every rule for detecting abnormal heartbeats (which would be nearly impossible), you give a machine learning system 10,000 ECG examples labeled "normal" or "abnormal." It learns the patterns that distinguish them. When it sees a new ECG, it can classify it accurately, even though you never explicitly told it the rules.

**See Also**: Artificial intelligence, Deep learning, Training, Neural network

### Median
**Category**: 🔢 Math & Statistics

**Simple Definition**: The median is the middle value when you arrange all numbers from smallest to largest. Half the values are above the median, and half are below. Unlike the average, extreme values don't affect the median much.

**Example**: Five heart rate readings: 45, 68, 70, 72, 150 bpm. The median is 70 (the middle value). The average is 81 - pulled up by that weird 150 reading. The median gives you a better sense of "typical" when you have outliers. It says "half the readings were above 70, half below."

**See Also**: Average, Distribution, Outlier, Data point

### Membrane Potential
**Category**: 🫀 Body & Biology

**Simple Definition**: Membrane potential is the electrical voltage difference between the inside and outside of a cell. Neurons and muscle cells can change their membrane potential to send signals. Resting neurons are typically at -70 millivolts (inside negative).

**Example**: A resting neuron is like a charged battery, sitting at -70 mV. When stimulated enough, gates in the membrane open, ions rush in, and the voltage spikes to +40 mV. This is an action potential! Immediately after, the cell pumps ions back out, resetting to -70 mV. The voltage changes are how neurons communicate.

**See Also**: Action potential, Neuron, Synapse, Neurotransmitter

### Model
**Category**: 🤖 Computers & AI

**Simple Definition**: In machine learning, a model is a trained system that can make predictions or classifications. It's the result of training - a set of learned patterns and rules. You train a model once, then use it many times for inference.

**Example**: You train a model to detect sleep apnea from breathing signals. After training on 1,000 patients, the model has learned what apnea events look like. Now it's a tool that doctors can use: feed in a patient's overnight recording, and the model reports how many apnea events occurred and their severity.

**See Also**: Training, Inference, Machine learning, Neural network

### Montage
**Category**: 📊 Signals & Measurement

**Simple Definition**: In EEG, a montage is the arrangement of electrode connections that determines which signals you calculate and display. Different montages highlight different brain activities. Common types include referential (all electrodes compared to one reference) and bipolar (adjacent electrodes compared to each other).

**Example**: You record EEG with 10 electrodes on the scalp. A referential montage might show all 10 channels comparing each electrode to your earlobe reference. A bipolar montage shows 9 channels, each the difference between neighboring electrodes. The raw data is the same, but different montages reveal different patterns. It's like viewing the same landscape from different angles.

**See Also**: EEG, Channel, Lead, Reference

### Motor Neuron
**Category**: 🫀 Body & Biology

**Simple Definition**: A motor neuron is a nerve cell that controls muscles. It carries signals from your brain and spinal cord to muscles, telling them to contract. When motor neurons fire, muscles move.

**Example**: You decide to wave your hand. Motor neurons in your brain send signals down your spinal cord to motor neurons in your arm. These neurons fire action potentials that travel to hand muscles. The muscles receive the signals and contract, making your hand wave. No motor neurons = no movement!

**See Also**: Neuron, Action potential, EMG, Sensory neuron

### Muscle Fiber
**Category**: 🫀 Body & Biology

**Simple Definition**: A muscle fiber is a single muscle cell. Each fiber is long and thin, containing proteins that slide past each other to create contraction. A whole muscle contains thousands of fibers bundled together. When motor neurons activate fibers, they contract and generate force.

**Example**: Your bicep muscle contains about 250,000 individual muscle fibers. When you curl a light weight, maybe 10,000 fibers contract. Curl a heavy weight, and 100,000+ fibers activate. EMG signals detect the electrical activity when fibers fire. More active fibers = bigger EMG signal.

**See Also**: EMG, Motor neuron, Muscle, Action potential

---

## N

### Nerve
**Category**: 🫀 Body & Biology

**Simple Definition**: A nerve is a bundle of axons (nerve fibers) wrapped together like cables. Nerves carry signals between your brain, spinal cord, and the rest of your body. Some carry sensory information in, while others carry motor commands out.

**Example**: Your sciatic nerve is the longest and thickest nerve, running from your lower back down your leg. It contains thousands of individual axons bundled together. Some carry touch sensation from your foot to your brain. Others carry movement commands from your brain to leg muscles. Damage to this nerve causes pain, numbness, and weakness.

**See Also**: Neuron, Axon, Peripheral nervous system, Action potential

### Neural Network
**Category**: 🤖 Computers & AI

**Simple Definition**: A neural network is a machine learning system inspired by how brains work. It consists of layers of artificial neurons connected together. Each neuron receives inputs, processes them, and passes outputs to the next layer. The network learns by adjusting connection strengths.

**Example**: A heart arrhythmia detector might be a neural network with 3 layers: Input layer (100 neurons receiving ECG data points), hidden layer (50 neurons finding patterns), and output layer (5 neurons for different arrhythmia types). During training, the connections between neurons are adjusted until the network reliably identifies each arrhythmia type.

**See Also**: Deep learning, Spiking neural network, Machine learning, Artificial intelligence

### Neuron
**Category**: 🫀 Body & Biology

**Simple Definition**: A neuron is a specialized cell that carries electrical and chemical signals. Your brain contains about 86 billion neurons. Each has a cell body (soma), branches that receive signals (dendrites), and a long fiber that sends signals (axon). Neurons communicate at connection points called synapses.

**Example**: Think of a neuron as a tiny decision-maker. It receives hundreds of input signals through its dendrites - some saying "fire!" and others saying "don't fire!" The neuron adds them all up. If the total reaches a threshold, it fires an action potential down its axon to tell other neurons what to do. This happens billions of times per second in your brain!

**See Also**: Action potential, Synapse, Axon, Dendrite, Soma

### Neurotransmitter
**Category**: 🫀 Body & Biology

**Simple Definition**: A neurotransmitter is a chemical messenger that carries signals between neurons at synapses. When an action potential reaches the end of an axon, it triggers release of neurotransmitters. These chemicals cross the gap and affect the next neuron. There are dozens of different neurotransmitters.

**Example**: Serotonin is a neurotransmitter that affects mood, sleep, and appetite. When one neuron fires, serotonin molecules are released into the synapse. They float across the tiny gap and attach to receptors on the next neuron, making it more likely to fire. Low serotonin levels are linked to depression. Many antidepressants work by increasing serotonin in synapses.

**See Also**: Synapse, Neuron, Action potential, Receptor

### Noise
**Category**: 📊 Signals & Measurement

**Simple Definition**: Noise is random, unwanted variation in a signal that doesn't carry useful information. It can come from electrical interference, movement, poor connections, or the measurement equipment itself. Reducing noise makes signals clearer and easier to analyze.

**Example**: You're recording ECG, and every signal has a fuzzy, random jitter on top of the heartbeat pattern. That jitter is noise - it doesn't tell you anything about the heart. Some noise comes from muscle tension, some from electrical equipment nearby. Filters and proper electrode placement reduce noise, revealing the clean heart signal underneath.

**See Also**: Artifact, Signal, Filter, Baseline

### Normalization
**Category**: 🔢 Math & Statistics

**Simple Definition**: Normalization scales data to a standard range, making different measurements comparable. Common approaches include scaling to 0-1 range or converting to z-scores. Normalization removes the effects of different units or scales.

**Example**: You want to compare heart rate (60-180 bpm) and breathing rate (10-40 breaths/min) - but they're in different ranges! Normalize both to 0-1 scale: 0 is each person's minimum, 1 is their maximum. Now you can fairly compare patterns. When both values are high (near 1), you know the person is exercising hard, regardless of the original units.

**See Also**: Z-score, Standard deviation, Feature, Distribution

### Normative Data
**Category**: ⚕️ Medical & Clinical

**Simple Definition**: Normative data shows what's typical or normal for a healthy population. It provides reference ranges for comparison. When you measure a patient, you compare to normative data to see if they're within the normal range or abnormal.

**Example**: Normative data for resting heart rate in adults: 60-100 bpm. If your patient has a resting HR of 75, that's normal (within the range). A patient with 45 bpm is below normal (bradycardia), and 120 bpm is above normal (tachycardia). Without normative data, you wouldn't know what's typical and what needs attention.

**See Also**: Distribution, Diagnosis, Screening, Z-score

---

## O

### Outlier
**Category**: 🔢 Math & Statistics

**Simple Definition**: An outlier is a data point that's very different from the others - unusually high or unusually low. Outliers can be real (meaningful extreme values) or errors (measurement mistakes). Identifying outliers is important for quality control and accurate analysis.

**Example**: You measure heart rate 100 times: most readings are 68-75 bpm, but one reading is 450 bpm. That's an outlier - it's impossible to have a heart rate that high! It's probably an error from bad electrode contact. Remove the outlier before calculating the average, or it will make your results wrong.

**See Also**: Median, Artifact, Noise, Distribution

### Overfitting
**Category**: 🤖 Computers & AI

**Simple Definition**: Overfitting happens when a model learns the training data too well, including noise and random patterns. The model performs great on training data but poorly on new data. It's like memorizing test answers instead of understanding the concepts - you fail when questions change slightly.

**Example**: You train a heartbeat detector with 100 ECG examples. After 100 training epochs, it's 99.9% accurate on those 100 - amazing! But test it on new ECGs, and accuracy drops to 60%. The model memorized the specific training examples instead of learning general heartbeat patterns. It's overfitted. The solution: use more training data, train for fewer epochs, or simplify the model.

**See Also**: Training, Validation, Model, Epoch (training)

---

## P

### Parasympathetic
**Category**: 🫀 Body & Biology

**Simple Definition**: The parasympathetic nervous system is the "rest and digest" part of your autonomic nervous system. It slows your heart rate, promotes digestion, and conserves energy. It's active when you're relaxed and safe. The opposite is the sympathetic system.

**Example**: After a stressful day, you settle into a comfortable chair and take deep breaths. Your parasympathetic nervous system kicks in: heart rate drops from 80 to 65 bpm, breathing slows and deepens, muscles relax, and digestion improves. You feel calm. The parasympathetic system is doing its job, helping you recover.

**See Also**: Sympathetic, Autonomic nervous system, Heart rate variability, Heart rate

### Pathology
**Category**: ⚕️ Medical & Clinical

**Simple Definition**: Pathology is the study of disease - what causes it, how it develops, and what effects it has on the body. In biosignals, pathological patterns indicate disease or dysfunction. Normal patterns indicate health.

**Example**: A normal ECG shows a regular P-wave, QRS complex, and T-wave pattern. In pathology, you might see: missing P-waves (atrial fibrillation), wide QRS complexes (bundle branch block), or ST-segment elevation (heart attack). Each pathological pattern has diagnostic meaning. Recognizing these patterns helps identify diseases.

**See Also**: Diagnosis, Biomarker, Waveform morphology, Screening

### Peak
**Category**: 🔢 Math & Statistics

**Simple Definition**: A peak is a high point in a signal where the value is greater than surrounding values. Finding peaks is essential for detecting events like heartbeats, spikes, or breaths. Peak detection algorithms automatically locate these important points.

**Example**: An ECG shows peaks for each heartbeat - the R-wave, the tallest part of the QRS complex. A peak detection algorithm scans the signal looking for sharp, high points. When it finds one, it marks it as a heartbeat. Count all the peaks in 60 seconds, and you have the heart rate. No peaks = no heartbeats = big problem!

**See Also**: Amplitude, Threshold, Signal, Waveform

### Peripheral Nervous System
**Category**: 🫀 Body & Biology

**Simple Definition**: The peripheral nervous system (PNS) includes all nerves outside the brain and spinal cord. It connects the central nervous system to the rest of your body. The PNS has two parts: sensory nerves (carrying information in) and motor nerves (carrying commands out).

**Example**: When you touch a hot stove, sensory nerves in your peripheral nervous system carry "pain!" signals from your hand to your spinal cord. Your CNS processes this and sends back a command through motor nerves: "Move hand away!" The PNS is the communication network between your brain and body.

**See Also**: Central nervous system, Motor neuron, Sensory neuron, Nerve

### PPG
**Category**: 📊 Signals & Measurement

**Simple Definition**: PPG stands for photoplethysmogram. It uses light to measure blood flow in small blood vessels near the skin surface. Shine a light (usually green or red LED) through your skin, and a sensor detects how much light gets absorbed. More blood = more absorption = stronger pulse signal.

**Example**: Your fitness watch has a PPG sensor on the back that glows green. Blood absorbs green light well, so when your heart pumps blood into your wrist, less light reflects back. The sensor sees this as a dip in the signal - that's one heartbeat. By detecting these pulses, the watch calculates your heart rate, even though there are no electrodes!

**See Also**: Heart rate, Pulse, Sensor, Amplitude

### Precision
**Category**: 🤖 Computers & AI

**Simple Definition**: Precision measures how many of the items identified as positive are actually correct. If a model says "this is a heartbeat" 100 times and 90 are real heartbeats (10 are false alarms), precision is 90%. High precision means few false positives.

**Example**: A seizure detector flags 50 events as "seizure." A doctor reviews them: 45 are real seizures, 5 are false alarms (actually just movement artifacts). The precision is 45÷50 = 90%. Good precision means when the detector says "seizure," you can trust it. Low precision would mean lots of false alarms and wasted doctor time.

**See Also**: Accuracy, Recall, Classification, Validation

### Prediction
**Category**: 🤖 Computers & AI

**Simple Definition**: A prediction is a forecast or estimate about what will happen or what category something belongs to. Machine learning models make predictions based on patterns they learned. Predictions can be future events, classifications, or estimated values.

**Example**: You feed 30 seconds of ECG data into a trained arrhythmia detection model. It predicts: "Normal Sinus Rhythm, 85% confidence." That's a classification prediction. Or you input HRV data, and a model predicts: "Tomorrow's stress level: 6 out of 10." That's a future prediction. Both use learned patterns to estimate unknown information.

**See Also**: Inference, Classification, Model, Machine learning

### Prognosis
**Category**: ⚕️ Medical & Clinical

**Simple Definition**: Prognosis is a prediction about how a disease will develop and what the outcome will be. Will the patient recover? How long will it take? Will the condition worsen? Biosignals help doctors make more accurate prognoses.

**Example**: Two patients both have mild cognitive impairment. Patient A's EEG shows strong, organized alpha waves - good prognosis, likely to remain stable. Patient B's EEG shows disorganized, slow waves - poor prognosis, higher risk of progressing to dementia. The EEG biomarkers help predict future outcomes.

**See Also**: Diagnosis, Biomarker, Longitudinal, Pathology

### Pulse
**Category**: 🫀 Body & Biology

**Simple Definition**: A pulse is the rhythmic expansion and contraction of arteries as blood is pumped through them. Each heartbeat creates a pressure wave that travels through your blood vessels. You can feel your pulse wherever arteries are close to the skin surface.

**Example**: Press two fingers against your wrist, just below your thumb. You feel a rhythmic throbbing - that's your pulse! Each throb is one heartbeat pushing blood through the artery. Count the pulses in 15 seconds and multiply by 4 to get your heart rate in beats per minute. Doctors have checked pulses this way for thousands of years.

**See Also**: Heart rate, PPG, Cardiac cycle, Waveform

---

## R

### Recall
**Category**: 🤖 Computers & AI

**Simple Definition**: Recall measures what fraction of actual positive cases were correctly identified. If there are 100 real heartbeats and your detector finds 95 of them (missing 5), recall is 95%. High recall means you don't miss many real events.

**Example**: A patient has 80 sleep apnea events during the night. Your detector flags 72 of them - that's 72÷80 = 90% recall. The detector missed 8 events (10%). High recall is critical for medical devices - missing a real seizure or heart attack could be dangerous. You'd rather have false alarms than miss real problems.

**See Also**: Precision, Accuracy, Classification, Sensitivity

### Receptor
**Category**: 🫀 Body & Biology

**Simple Definition**: A receptor is a protein on a cell surface that detects specific chemical signals. Receptors act like locks that only certain keys (molecules) can open. When the right molecule binds to a receptor, it triggers a response in the cell.

**Example**: At a synapse, the receiving neuron has neurotransmitter receptors. When serotonin molecules (released from the sending neuron) float across and bind to serotonin receptors, channels open and ions rush in. This changes the receiving neuron's membrane potential, potentially triggering an action potential. Receptors are how chemical signals become electrical signals.

**See Also**: Neurotransmitter, Synapse, Neuron, Membrane potential

### Reference
**Category**: 📊 Signals & Measurement

**Simple Definition**: A reference is a common point that voltages are measured against. In biosignal recording, all signals are relative to the reference electrode. Choosing a good reference location is important for clean signals.

**Example**: In EEG, you might place a reference electrode on your earlobe or mastoid (bone behind the ear). All other electrodes measure voltage compared to this reference. If you see "+50 microvolts" at an electrode, it means 50 microvolts higher than the reference point. Different references can make signals look quite different!

**See Also**: Electrode, Montage, Channel, Ground

### Reflex
**Category**: 🫀 Body & Biology

**Simple Definition**: A reflex is an automatic, involuntary response to a stimulus. Reflexes are fast because the signal doesn't have to travel all the way to your brain - your spinal cord handles them. You can't consciously control reflexes.

**Example**: The doctor taps your knee with a rubber hammer. Your leg kicks out automatically - that's the patellar reflex! Here's what happens in milliseconds: tap stretches the tendon, sensory neurons detect this and send signals to the spinal cord, motor neurons immediately fire back to thigh muscles, and your leg extends. Your brain only finds out after it's already happened!

**See Also**: Motor neuron, Sensory neuron, Action potential, Spinal cord

---

## S

### Sample Rate
**Category**: 📊 Signals & Measurement

**Simple Definition**: Sample rate is how many measurements are taken per second, measured in Hertz (Hz) or samples per second. Higher sample rates capture faster changes but create larger files. The sample rate must be at least twice the highest frequency you want to detect (Nyquist theorem).

**Example**: ECG signals contain frequencies up to about 40 Hz (fast components of the QRS complex). To capture these faithfully, you need a sample rate of at least 80 Hz, but 250-500 Hz is typical for clinical ECG. That means taking 250-500 measurements every second. EEG typically uses 250 Hz. Too slow, and you miss important details!

**See Also**: Frequency, Hertz, Data point, Signal

### Screening
**Category**: ⚕️ Medical & Clinical

**Simple Definition**: Screening is testing healthy people to detect disease early, before symptoms appear. Effective screening catches problems when they're easier to treat. Biosignals enable automated, low-cost screening that can be done frequently.

**Example**: A community health program uses 30-second ECG screening for everyone over 65. Most screens are normal, but the system flags 2% with atrial fibrillation - people who had no symptoms but are at high risk for stroke. Thanks to screening, they can start preventive treatment before having a stroke. Early detection saves lives!

**See Also**: Diagnosis, Biomarker, Longitudinal, Sensitivity

### Sensory Neuron
**Category**: 🫀 Body & Biology

**Simple Definition**: A sensory neuron is a nerve cell that detects information from inside or outside your body and sends it to your brain or spinal cord. Sensory neurons respond to touch, temperature, pain, light, sound, and other stimuli.

**Example**: Touch something sharp. Sensory neurons in your fingertip have receptors that detect tissue damage. They fire action potentials that race up your arm, through your spinal cord, to your brain. Your brain interprets these signals as "pain!" The whole process takes less than a tenth of a second. Sensory neurons are your body's alarm system.

**See Also**: Motor neuron, Neuron, Action potential, Receptor

### Sensitivity
**Category**: ⚕️ Medical & Clinical

**Simple Definition**: Sensitivity is the ability of a test to correctly identify people who have a disease (true positive rate). If 100 people have a condition and a test correctly identifies 95 of them, sensitivity is 95%. High sensitivity means few false negatives - you don't miss many real cases.

**Example**: A seizure detection system with 90% sensitivity correctly identifies 90 out of 100 seizures (but misses 10). That's the same as recall in machine learning. For serious conditions, high sensitivity is critical - you'd rather have false alarms than miss a real emergency. Sensitivity answers: "Of all the sick people, what percentage did we catch?"

**See Also**: Specificity, Recall, Screening, Diagnosis

### Sensor
**Category**: 📊 Signals & Measurement

**Simple Definition**: A sensor is a device that detects and responds to physical input from the environment. Biosignal sensors convert body activity (electrical, optical, mechanical) into signals that can be measured and recorded. Each type of sensor is designed for specific measurements.

**Example**: Different sensors for different jobs: ECG electrodes detect tiny voltages on skin (electrical sensors), PPG uses light absorption to measure blood flow (optical sensor), and accelerometers detect movement and vibration (mechanical sensors). Your smartphone contains dozens of sensors working together to understand your environment and activity.

**See Also**: Electrode, Channel, PPG, Impedance

### Signal
**Category**: 📊 Signals & Measurement

**Simple Definition**: A signal is a measurable change that carries information. In biosignal analysis, signals are electrical, optical, or mechanical patterns generated by body processes. Signals vary over time and can be recorded, analyzed, and interpreted.

**Example**: Your heartbeat creates an electrical signal (ECG), a pressure signal (blood pressure pulse), a sound signal (the "lub-dub" you hear with a stethoscope), and an optical signal (changes in blood flow measured by PPG). All these signals carry the same underlying information - your heart rhythm - but in different forms.

**See Also**: Amplitude, Frequency, Waveform, Time series

### Sinoatrial Node
**Category**: 🫀 Body & Biology

**Simple Definition**: The sinoatrial (SA) node is your heart's natural pacemaker. It's a small group of specialized cells in the right atrium that generate electrical impulses. These impulses spread through the heart, causing it to contract rhythmically. The SA node sets your heart rate.

**Example**: The SA node fires about 60-100 times per minute at rest, and each firing triggers one heartbeat. When you exercise, your nervous system tells the SA node to speed up - maybe to 160 beats per minute. When you sleep, it slows down to 50-55 bpm. It automatically adjusts to your body's needs without you thinking about it.

**See Also**: Cardiac cycle, Heart rate, ECG, Autonomic nervous system

### Soma
**Category**: 🫀 Body & Biology

**Simple Definition**: The soma is the cell body of a neuron - the central part containing the nucleus and most cellular machinery. Dendrites extend from the soma to receive signals, and the axon extends from it to send signals. The soma integrates all incoming signals and decides whether to fire an action potential.

**Example**: Picture a neuron like a sun with rays. The soma is the central sphere containing the "brain" of the cell. Dendrite branches extend out like short rays, receiving signals from hundreds of other neurons. One long axon ray extends out the other side to send signals to distant neurons. The soma adds up all the dendrite inputs and makes the fire/don't-fire decision.

**See Also**: Neuron, Dendrite, Axon, Action potential

### Specificity
**Category**: ⚕️ Medical & Clinical

**Simple Definition**: Specificity is the ability of a test to correctly identify people who don't have a disease (true negative rate). If 100 healthy people are tested and 95 are correctly identified as healthy, specificity is 95%. High specificity means few false positives - you don't incorrectly label healthy people as sick.

**Example**: An arrhythmia detector tests 1,000 people with normal heart rhythms. It correctly says "normal" for 950 of them (95% specificity) but falsely flags 50 as "abnormal" (5% false positive rate). Those 50 people get unnecessary worry and follow-up tests. Higher specificity would mean fewer false alarms. Specificity answers: "Of all the healthy people, what percentage did we correctly identify?"

**See Also**: Sensitivity, Precision, Screening, Diagnosis

### Spectrum
**Category**: 📊 Signals & Measurement

**Simple Definition**: A spectrum shows which frequencies are present in a signal and how strong each one is. It's the result of applying a Fourier transform. The x-axis shows frequency, and the y-axis shows power or amplitude. Peaks in the spectrum indicate important frequencies.

**Example**: An EEG spectrum might show: a large peak at 10 Hz (alpha rhythm, indicating relaxed wakefulness), a smaller peak at 20 Hz (beta rhythm, from mental activity), and low power above 30 Hz (minimal high-frequency activity). The spectrum reveals the frequency composition that's hidden in the time-domain signal. It's like seeing the individual colors that make up white light.

**See Also**: Frequency, Fourier transform, Frequency domain, Hertz

### Spiking Neural Network
**Category**: 🤖 Computers & AI

**Simple Definition**: A spiking neural network (SNN) is a type of artificial neural network that closely mimics real neurons. Instead of continuous values, neurons in an SNN communicate with brief spikes (like action potentials). The timing of spikes carries information. SNNs are more biologically realistic and can be very energy-efficient.

**Example**: A traditional neural network might output "0.7" to represent a strong signal. An SNN instead fires 7 spikes in 10 milliseconds - the spike rate carries the same information. SNNs can run on specialized neuromorphic chips that use 1,000 times less power than regular AI chips. This makes them perfect for wearable medical devices with small batteries.

**See Also**: Neural network, Deep learning, Action potential, Neuron

### Standard Deviation
**Category**: 🔢 Math & Statistics

**Simple Definition**: Standard deviation measures how spread out numbers are around the average. Small standard deviation means values cluster tightly around the mean. Large standard deviation means values are scattered widely. It's calculated as the square root of the variance.

**Example**: Person A's heart rate readings: 70, 71, 69, 70, 72 bpm (average 70, standard deviation 1.2 bpm) - very consistent. Person B's readings: 55, 70, 85, 65, 75 bpm (average 70, standard deviation 11.2 bpm) - highly variable. Same average, but B's heart rate jumps around much more, shown by the larger standard deviation.

**See Also**: Variance, Average, Distribution, Z-score

### Sympathetic
**Category**: 🫀 Body & Biology

**Simple Definition**: The sympathetic nervous system is the "fight or flight" part of your autonomic nervous system. It speeds up your heart, increases breathing, and prepares your body for action. It's activated by stress, danger, or excitement. The opposite is the parasympathetic system.

**Example**: You're about to give a presentation to 100 people. Your sympathetic nervous system activates: heart rate jumps from 65 to 120 bpm, breathing quickens, palms sweat, pupils dilate. Your body is flooding with adrenaline, preparing you to either perform brilliantly or run away! After the presentation, the parasympathetic system calms everything back down.

**See Also**: Parasympathetic, Autonomic nervous system, Heart rate, EDA

### Synapse
**Category**: 🫀 Body & Biology

**Simple Definition**: A synapse is the tiny gap between two neurons where they communicate. When an action potential reaches the end of an axon, it triggers release of neurotransmitters. These chemicals cross the synapse and affect the receiving neuron. Each neuron can have thousands of synapses.

**Example**: Imagine two people standing 20 nanometers apart (the width of a synapse) trying to communicate. One person throws chemical message balls (neurotransmitters) across the gap. The other person catches them with special gloves (receptors). If enough balls are caught, the receiving person fires an electrical signal down their arm (action potential). That's how synapses work!

**See Also**: Neuron, Neurotransmitter, Action potential, Receptor

---

## T

### Tachycardia
**Category**: ⚕️ Medical & Clinical

**Simple Definition**: Tachycardia means a faster than normal heart rate, typically above 100 beats per minute in adults at rest. It can be normal during exercise or stress, but resting tachycardia might indicate a heart problem, fever, or other condition.

**Example**: You measure your heart rate while sitting and watching TV. If it's 115 bpm, that's tachycardia - faster than the normal resting range of 60-100 bpm. During a workout, 115 bpm would be normal. But at rest, sustained tachycardia could indicate problems with the heart's electrical system, dehydration, or other issues that need evaluation.

**See Also**: Bradycardia, Heart rate, Arrhythmia, ECG

### Threshold
**Category**: 🔢 Math & Statistics

**Simple Definition**: A threshold is a cutoff value used to make binary decisions. Values above the threshold are classified one way; values below are classified another way. Setting the right threshold balances sensitivity (catching real events) and specificity (avoiding false alarms).

**Example**: A heartbeat detector sets a threshold at 0.5 millivolts. When the ECG signal rises above 0.5 mV, it counts as a heartbeat. Below 0.5 mV, it's not a heartbeat. Set the threshold too low, and noise triggers false heartbeats. Set it too high, and you miss real weak heartbeats. Finding the optimal threshold is crucial for accurate detection.

**See Also**: Peak, Classification, Sensitivity, Specificity

### Time Domain
**Category**: 🔢 Math & Statistics

**Simple Definition**: The time domain is the normal way of viewing signals, with time on the x-axis and amplitude on the y-axis. It shows how a signal changes moment by moment. Most raw biosignals are initially viewed in the time domain.

**Example**: An ECG in the time domain shows heartbeat spikes marching across the screen from left to right as time passes. You can see each P-wave, QRS complex, and T-wave. You can measure how far apart heartbeats are. This is the familiar view that doctors have used for over 100 years to diagnose heart problems.

**See Also**: Frequency domain, Time series, Signal, Waveform

### Time Series
**Category**: 🔢 Math & Statistics

**Simple Definition**: A time series is a sequence of data points measured at successive time intervals. Biosignals are time series - they show how something changes over time. Time series analysis reveals trends, cycles, and patterns.

**Example**: Your continuous heart rate throughout the day is a time series: 65 bpm at 8 AM, 120 bpm at 9 AM (morning jog), 75 bpm at 10 AM (recovered), 80 bpm at noon, etc. Plotting these 24 hours of data shows your daily rhythm - higher during activity, lower during sleep. Time series analysis can predict tomorrow's pattern based on past patterns.

**See Also**: Signal, Data point, Longitudinal, Time domain

### Training
**Category**: 🤖 Computers & AI

**Simple Definition**: Training is the process of teaching a machine learning model to recognize patterns. The model learns from many labeled examples, adjusting its internal parameters to improve performance. Training requires a dataset, computing power, and time.

**Example**: You train an EMG fatigue detector using data from 100 athletes. Initially, the model guesses randomly (50% accuracy). You show it thousands of examples labeled "fresh" or "fatigued," and it gradually learns the patterns. After 20 epochs of training, it reaches 92% accuracy. Training took 2 hours. Now the trained model can detect fatigue in seconds.

**See Also**: Model, Epoch (training), Dataset, Validation, Machine learning

---

## V

### Validation
**Category**: 🤖 Computers & AI

**Simple Definition**: Validation is testing a trained model on data it hasn't seen during training to check if it generalizes well. Validation data is separate from training data. Good performance on validation data means the model learned real patterns, not just memorized training examples.

**Example**: You train a seizure detector using 800 patients' EEG data. Then you validate it on 200 completely different patients that the model never saw during training. If it maintains 90% accuracy on the validation set, great - the model generalizes! If validation accuracy drops to 60%, the model overfit the training data and doesn't work well on new patients.

**See Also**: Training, Overfitting, Accuracy, Dataset

### Variance
**Category**: 🔢 Math & Statistics

**Simple Definition**: Variance measures how far data points spread from their average. It's calculated as the average of squared differences from the mean. Large variance means data is scattered widely. Small variance means data is tightly clustered. Standard deviation is the square root of variance.

**Example**: Dataset A: 70, 70, 70, 70, 70 bpm (variance = 0 - no variation at all). Dataset B: 60, 65, 70, 75, 80 bpm (variance = 62.5 - moderate spread). Dataset C: 40, 50, 70, 90, 100 bpm (variance = 512.5 - high spread). Higher variance means more unpredictability and variability in the measurements.

**See Also**: Standard deviation, Distribution, Heart rate variability, Average

### Ventricle
**Category**: 🫀 Body & Biology

**Simple Definition**: The ventricles are the two lower chambers of the heart that pump blood out to the body and lungs. The right ventricle pumps blood to the lungs to get oxygen. The left ventricle pumps oxygen-rich blood to the rest of the body. Ventricles have thick, muscular walls.

**Example**: During each heartbeat, the ventricles contract powerfully - this is the strong "dub" sound you hear. The left ventricle is especially muscular because it pumps blood all the way from your heart to your toes and back. On an ECG, the large QRS complex represents ventricular contraction - it's the biggest, most visible part of the heartbeat.

**See Also**: Cardiac cycle, Heart rate, ECG, Sinoatrial node

### Vital Signs
**Category**: ⚕️ Medical & Clinical

**Simple Definition**: Vital signs are basic measurements that indicate how well your body is functioning. The traditional vital signs are: heart rate, blood pressure, breathing rate, and body temperature. Modern medicine adds oxygen saturation and sometimes pain level. They're called "vital" because they're essential indicators of health.

**Example**: A nurse checks your vital signs at the hospital: Heart rate 75 bpm (normal), blood pressure 120/80 mmHg (normal), breathing rate 16 breaths/min (normal), temperature 98.6°F (normal), oxygen saturation 98% (normal). All five are in healthy ranges, so your basic body functions are working well. Abnormal vital signs are often the first warning of medical problems.

**See Also**: Heart rate, Screening, Biomarker, Diagnosis

---

## W

### Waveform
**Category**: 📊 Signals & Measurement

**Simple Definition**: A waveform is the shape of a signal when plotted over time. Different biosignals have characteristic waveform shapes that can be recognized visually. Analyzing waveform features helps identify normal and abnormal patterns.

**Example**: An ECG waveform has a distinctive shape: small P-wave (atrial contraction), spike QRS complex (ventricular contraction), and rounded T-wave (recovery). This "P-QRS-T" pattern repeats with each heartbeat. Doctors learn to recognize dozens of abnormal waveforms - wide QRS means conduction problems, tall T-waves might indicate high potassium, missing P-waves suggest atrial fibrillation. The waveform tells the story.

**See Also**: Signal, Amplitude, Peak, Morphology

### Waveform Morphology
**Category**: ⚕️ Medical & Clinical

**Simple Definition**: Morphology means shape and structure. Waveform morphology describes the detailed shape of signal features - their height, width, slopes, and patterns. Changes in morphology indicate changes in the underlying physiology or pathology.

**Example**: Two patients both have heart rates of 75 bpm (same frequency), but their ECG waveform morphologies are different. Patient A has narrow QRS complexes (0.08 seconds wide) - normal morphology. Patient B has wide QRS complexes (0.14 seconds) - abnormal morphology indicating the electrical signal is taking a slow, detoured path through the heart. Morphology reveals problems that timing alone can't show.

**See Also**: Waveform, Signal, Pathology, ECG

### Window
**Category**: 📊 Signals & Measurement

**Simple Definition**: A window is a segment of a signal selected for analysis. Long recordings are often divided into windows (similar to epochs) to analyze smaller chunks. Windows can be sliding (overlapping) or non-overlapping. Window size affects what patterns you can detect.

**Example**: You have 10 minutes of EMG data from an exercise session. Analyzing all 10 minutes together would average out important changes. Instead, you use 5-second windows: analyze seconds 0-5, then 5-10, then 10-15, and so on. Each window gets its own analysis. You can track how muscle fatigue develops window by window throughout the workout.

**See Also**: Epoch, Time series, Signal, Segmentation

---

## Z

### Zero-Crossing
**Category**: 🔢 Math & Statistics

**Simple Definition**: A zero-crossing is a point where a signal changes from positive to negative or vice versa - it crosses the zero line. Counting zero-crossings can estimate frequency or detect events. More zero-crossings per second means higher frequency content.

**Example**: An audio recording of your heartbeat through a stethoscope crosses zero 3 times per second (positive to negative, back to positive, back to negative). That's 3 zero-crossings per second, suggesting frequency content around 1.5 Hz - consistent with a 90 bpm heart rate. Zero-crossing rate is a simple way to estimate signal frequency without complex math.

**See Also**: Baseline, Frequency, Threshold, Signal

### Z-Score
**Category**: 🔢 Math & Statistics

**Simple Definition**: A z-score tells you how many standard deviations away from the mean a value is. Z-score of 0 means exactly average. Positive z-scores are above average; negative are below. Z-scores of +2 or -2 are unusual (only 5% of values). Z-scores of +3 or -3 are very rare.

**Example**: Normal resting heart rate average is 72 bpm with standard deviation of 8 bpm. Your heart rate is 88 bpm. Z-score = (88-72)÷8 = +2.0. You're exactly 2 standard deviations above average - higher than 97.5% of people. A doctor might investigate further. Someone with 60 bpm has z-score = (60-72)÷8 = -1.5, which is still within normal range (1.5 standard deviations below mean).

**See Also**: Standard deviation, Normalization, Average, Distribution

---

## Quick Category Reference

**🫀 Body & Biology (25 terms)**
Action potential, Autonomic nervous system, Axon, Cardiac cycle, Central nervous system, Dendrite, Electrode, Heart rate, Membrane potential, Motor neuron, Muscle fiber, Nerve, Neuron, Neurotransmitter, Parasympathetic, Peripheral nervous system, Pulse, Receptor, Reflex, Sensory neuron, Sinoatrial node, Soma, Sympathetic, Synapse, Ventricle

**📊 Signals & Measurement (25 terms)**
Amplitude, Artifact, Baseline, Channel, ECG/EKG, EDA, EEG, Electrode, EMG, Epoch, Filter, Frequency, Gain, Hertz, Impedance, Lead, Montage, Noise, PPG, Reference, Sample rate, Sensor, Signal, Spectrum, Waveform, Window

**🔢 Math & Statistics (20 terms)**
Algorithm, Average (mean), Correlation, Data point, Distribution, Fourier transform, Frequency domain, Histogram, Interpolation, Median, Normalization, Outlier, Peak, Standard deviation, Threshold, Time domain, Time series, Variance, Z-score, Zero-crossing

**🤖 Computers & AI (20 terms)**
Accuracy, Artificial intelligence, Batch, Classification, Dataset, Deep learning, Epoch (training), Feature, Inference, Label, Machine learning, Model, Neural network, Overfitting, Precision, Prediction, Recall, Spiking neural network, Training, Validation

**⚕️ Medical & Clinical (15 terms)**
Arrhythmia, Biomarker, Bradycardia, Diagnosis, Heart rate variability, Longitudinal, Normative data, Pathology, Prognosis, Screening, Sensitivity, Specificity, Tachycardia, Vital signs, Waveform morphology

---

## How to Learn More

This glossary is part of the Delta-Predictive Biosensing Learning Library. Here are other resources to deepen your understanding:

**For Beginners**: Start with Book 1 (Foundations - "Listening to the Body") to learn the history and basic concepts before diving into technical terms.

**For Visual Learners**: Check out the reference cards and diagrams that show these concepts in pictures and graphs.

**For Hands-On Practice**: Try the beginner Jupyter notebooks that let you work with real biosignals and see these terms in action.

**For Quick Reference**: Use this glossary whenever you encounter an unfamiliar term. The "See Also" links help you explore related concepts.

Remember: Learning this material takes time. Don't worry if some terms are confusing at first. Come back to this glossary often, and the concepts will gradually become clear. Everyone starts as a beginner!

---

*Last updated: 2025-12-18*
*Part of the Delta-Predictive Biosensing Learning Library*
*Target reading level: 8th grade (Ages 13-14)*
