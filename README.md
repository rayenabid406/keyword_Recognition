ESP32 Keyword Recognition – Detailed Technical Documentation
1. Project Overview

This project presents a complete, production-oriented implementation of a keyword recognition system designed specifically for the ESP32 microcontroller platform. The objective is to detect short spoken commands in real time using a highly optimized neural network that operates directly on raw audio samples.

The system recognizes three classes:

“on”

“off”

background noise (used to suppress false activations)

Unlike many traditional speech recognition pipelines, this project intentionally avoids MFCC or spectral feature extraction. Instead, the model operates directly on normalized 16 kHz waveform data, simplifying the signal processing pipeline and enabling tighter integration with TensorFlow Lite Micro.
The resulting system is efficient, lightweight, and suitable for deployment on low-power embedded hardware.

2. System Architecture

The system follows a linear and deterministic processing pipeline:

Audio capture via a digital microphone using the ESP32 I2S peripheral

Buffering of a one-second window (16,000 samples)

Signal normalization and INT8 quantization

Neural network inference

Post-processing for stable keyword detection

The neural network is a lightweight 1D convolutional architecture optimized for temporal pattern recognition. All computations are performed using 8-bit integers, ensuring compatibility with TensorFlow Lite Micro while minimizing RAM usage and inference latency.

3. Dataset Preparation

Training data is organized into class-specific directories, each containing one-second WAV audio recordings:

Sample rate: 16 kHz

Format: mono

Fixed length per sample

Maintaining consistent sample length and format is critical, as the model expects a fixed-size input tensor.

A dedicated background noise class is included to represent silence, ambient sounds, and non-keyword speech. This design choice significantly reduces false positives during real-world deployment and improves overall robustness.

4. Model Architecture and Training Strategy

The neural network consists of a sequence of Conv1D layers with progressively increasing filter counts. Each convolutional block extracts temporal features from the raw waveform, while max pooling layers reduce dimensionality and computational cost.

A Global Average Pooling layer is used instead of flattening to:

Reduce the parameter count

Improve generalization

Improve deployability on constrained hardware

Training is performed using:

Adam optimizer

Categorical cross-entropy loss

The dataset is split into training and validation subsets to monitor generalization performance. Despite the compact model size, the network successfully learns meaningful representations directly from raw audio when provided with sufficient data diversity.

5. INT8 Quantization and TFLite Conversion

After training, the model is converted to TensorFlow Lite using full integer (INT8) quantization. A representative dataset is provided during conversion to compute accurate scale and zero-point parameters.

Both input and output tensors are quantized to int8, enabling execution on microcontrollers without floating-point hardware.

INT8 quantization:

Dramatically reduces memory usage

Improves inference speed

Maintains acceptable classification accuracy

The resulting TFLite model is fully compatible with TensorFlow Lite Micro and can be compiled directly into ESP32 firmware.

6. ESP32 Deployment Considerations

On the ESP32, the TFLite model is embedded as a C array and executed using the MicroInterpreter.

Key deployment parameters:

Tensor arena size: ~40–50 KB

Input preprocessing: normalization + quantization using tensor scale and zero-point

To achieve stable predictions, post-processing techniques such as:

moving averages

confidence thresholds

temporal voting

are applied to reduce jitter and prevent rapid state changes caused by transient noise.

7. Performance and Optimization

The model is designed to balance:

accuracy

inference latency

memory usage

Potential optimization strategies include:

reducing convolutional filter counts

pruning layers

adjusting pooling strategies

Because the system operates directly on raw audio, dataset quality and diversity have a greater impact on performance than architectural complexity.

8. Future Extensions

Possible extensions include:

adding more keywords through dataset expansion and retraining

language-specific models

speaker-dependent models

integration with IoT platforms for voice-controlled automation and embedded HMIs

9. Conclusion

This project demonstrates a complete TinyML pipeline for keyword recognition on the ESP32. By combining raw audio processing, a compact neural network, and full INT8 quantization, it shows that practical speech-based interfaces are achievable on low-power microcontrollers.

The design prioritizes reliability, simplicity, and real-world deployability.

10. Engineering Challenges and Lessons Learned
Model Size vs. Accuracy Trade-off

The first trained neural network achieved high offline accuracy but exceeded ESP32 memory limits. It could not be loaded into TensorFlow Lite Micro due to excessive Flash and RAM usage during tensor allocation.

To resolve this:

The number of convolutional layers was reduced

Filter counts in early layers were decreased

Flattening was replaced with global average pooling

Receptive field size and temporal resolution were carefully balanced

Additional training samples were introduced to compensate for reduced model capacity, allowing acceptable accuracy within ESP32 constraints.

Final Deployed Architecture
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                         ┃ Output Shape                ┃ Param #         ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ conv1d (Conv1D)                      │ (None, 15988, 8)            │ 112             │
│ max_pooling1d (MaxPooling1D)         │ (None, 3997, 8)             │ 0               │
│ conv1d_1 (Conv1D)                    │ (None, 3989, 16)            │ 1,168           │
│ max_pooling1d_1 (MaxPooling1D)       │ (None, 997, 16)             │ 0               │
│ conv1d_2 (Conv1D)                    │ (None, 991, 32)             │ 3,616           │
│ max_pooling1d_2 (MaxPooling1D)       │ (None, 247, 32)             │ 0               │
│ global_average_pooling1d             │ (None, 32)                  │ 0               │
│ dense (Dense)                        │ (None, 16)                  │ 528             │
│ dense_1 (Dense)                     │ (None, 3)                   │ 51              │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
Total parameters: 5,475 (21.39 KB)


This architecture represents a deliberate compromise between expressiveness and deployability.

Dataset and Domain Mismatch

Early training data was recorded using a PC microphone, producing higher-quality audio than the ESP32’s digital MEMS microphone. This caused poor on-device inference despite promising offline results.

Mitigations included:

collecting or augmenting samples to simulate embedded microphone characteristics

expanding background noise samples

aligning normalization and amplitude scaling between Python training and ESP32 inference

Quantization and Numerical Stability

Early inference attempts produced unstable or near-zero outputs due to incorrect handling of quantization parameters.

Key fixes:

using a representative dataset during TFLite conversion

explicitly applying input tensor scale and zero-point on the ESP32

validating quantized outputs against floating-point references

After correction, the quantized model size was reduced to 15,360 bytes.

Tensor Arena and Memory Allocation

TensorFlow Lite Micro requires a statically allocated tensor arena. Undersizing resulted in silent initialization failures.

Through iterative profiling, a stable configuration was achieved using 40–50 KB of RAM, ensuring reliable execution.

Toolchain and Build System Integration

The project was developed using PlatformIO. Challenges included:

managing large C arrays generated from the TFLite model

resolving alignment and memory attribute issues

ensuring consistent builds across iterations

Once resolved, PlatformIO proved robust and scalable for embedded ML development.

11. Experimental Results and Evaluation

Training was conducted over 30 epochs using three classes (“on”, “off”, background noise).

Final metrics:

Test Accuracy: 79.31%

Model Size: 15,360 bytes

Input Size: 16,000 samples

Output Classes: 3

Training observations:

rapid convergence after epoch 6

stable validation accuracy despite model compression

improved robustness after dataset expansion

Final ESP32 deployment output:

=== NEW MODEL (79.31% accuracy) ===
Model: 15360 bytes
Allocating tensors...
Input size: 16000
Output size: 3
✅ Network ready!


These results confirm that meaningful keyword recognition performance can be achieved on low-power microcontrollers using compact neural networks and raw audio inputs.
