ESP32 Keyword Recognition – Detailed Technical Documentation 1. Project Overview This project presents a complete, production-oriented implementation of a keyword recognition system designed specifically for the ESP32 microcontroller platform. The objective is to detect short spoken commands in real time using a highly optimized neural network that operates directly on raw audio samples. The system recognizes three classes: the keywords 'on', 'off', and a background noise class used to suppress false activations. Unlike many traditional speech recognition pipelines, this project intentionally avoids MFCC or spectral feature extraction. Instead, the model operates on normalized 16 kHz waveform data, which simplifies the signal processing pipeline and allows tighter integration with TensorFlow Lite Micro. The resulting system is both efficient and easy to deploy on low-power embedded hardware. 2. System Architecture The system follows a linear and deterministic processing pipeline. Audio is captured from a digital microphone using the ESP32 I2S peripheral. Samples are buffered into a one-second window (16,000 samples), normalized, quantized, and passed directly to the neural network for inference. The output probabilities are post-processed to produce stable and reliable keyword detections. The neural network itself is a lightweight one-dimensional convolutional architecture optimized for temporal pattern recognition. All computations are performed using 8-bit integers, ensuring compatibility with TFLite Micro and minimizing both RAM usage and inference latency. 3. Dataset Preparation Training data is organized into class-specific directories. Each directory contains one-second WAV audio recordings sampled at 16 kHz and stored as mono signals. Maintaining consistent sample length and format is critical, as the model expects a fixed-size input tensor. The inclusion of a dedicated background noise class is a key design decision. This class represents ambient sounds, silence, and non-keyword speech. A well-curated background class significantly reduces false positives during real-world deployment and improves system robustness. 4. Model Architecture and Training Strategy The neural network consists of a sequence of Conv1D layers with progressively increasing filter counts. Each convolutional block extracts temporal features from the waveform while max pooling layers reduce dimensionality and computational cost. A global average pooling layer is used instead of flattening to drastically reduce the parameter count and improve generalization. Training is performed using the Adam optimizer and categorical cross-entropy loss. The dataset is split into training and validation subsets to monitor generalization performance. Despite the small model size, the network is capable of learning meaningful representations directly from raw audio when provided with sufficient data diversity. 5. INT8 Quantization and TFLite Conversion After training, the model is converted to TensorFlow Lite format using full integer quantization. This process requires a representative dataset, which is used to determine accurate scaling parameters for each layer. Both input and output tensors are quantized to int8, enabling execution on microcontrollers without floating-point hardware. INT8 quantization dramatically reduces memory usage and improves inference speed while maintaining acceptable accuracy. The resulting TFLite model is fully compatible with TensorFlow Lite Micro and can be compiled directly into the ESP32 firmware. 6. ESP32 Deployment Considerations On the ESP32, the TFLite model is embedded as a C array and executed using the MicroInterpreter. A tensor arena of approximately 40–50 KB is sufficient for this project. Audio samples must be normalized and quantized using the input tensor's scale and zero-point before inference. To achieve stable predictions, it is recommended to apply post-processing techniques such as moving averages, confidence thresholds, or temporal voting. These techniques reduce jitter and prevent rapid state changes due to transient noise. 7. Performance and Optimization The model is designed to balance accuracy, latency, and memory usage. Further optimizations may include reducing convolutional filter counts, pruning layers, or adjusting pooling strategies. Because the system operates on raw audio, careful dataset design has a greater impact on performance than architectural complexity. 8. Future Extensions This project can be extended in several directions. Additional keywords can be introduced by expanding the dataset and retraining the model. Language-specific models or speaker-dependent systems may also be developed. Integration with IoT platforms allows voice-controlled automation and embedded human-machine interfaces. 9. Conclusion This documentation describes a complete TinyML pipeline for keyword recognition on the ESP32. By combining raw audio processing, a compact neural network, and full INT8 quantization, the system demonstrates that practical speech-based interfaces are achievable on low-power microcontrollers. The design choices prioritize reliability, simplicity, and real-world deployability.
Engineering Challenges and Lessons Learned

Developing a keyword recognition system for deployment on a resource-constrained microcontroller such as the ESP32 introduced a number of practical engineering challenges. Addressing these challenges was a critical part of transforming an initially functional machine learning model into a deployable embedded system.

 Model Size vs. Accuracy Trade-off:

The first trained neural network achieved high classification accuracy during offline evaluation; however, its memory footprint significantly exceeded the limits imposed by the ESP32 platform. Although the model performed well on a desktop environment, it could not be loaded into TensorFlow Lite Micro due to excessive Flash and RAM requirements, particularly during tensor allocation.

To resolve this, the architecture was redesigned with embedded constraints as a primary consideration. This process involved:

Reducing the number of convolutional layers

Decreasing filter counts in early layers

Replacing flattening operations with global average pooling

Carefully balancing receptive field size and temporal resolution

At the same time, additional training samples were introduced to compensate for the reduced model capacity. This combination allowed the final model to retain acceptable accuracy while fitting comfortably within the ESP32 memory budget.

The final deployed architecture is summarized below:
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
│ dense_1 (Dense)                      │ (None, 3)                   │ 51              │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
Total parameters: 5,475 (21.39 KB)
This final architecture demonstrates a deliberate compromise between expressiveness and deployability.

Dataset and Domain Mismatch:

A major early obstacle was the mismatch between training data and deployment conditions. Initial audio samples were recorded using a PC microphone, which produced higher-quality signals than those captured by the ESP32’s digital MEMS microphone. This resulted in poor on-device inference performance despite promising offline accuracy.

To mitigate this issue:

Additional training samples were collected or augmented to simulate embedded microphone characteristics

Background noise samples were expanded to improve generalization

Signal normalization and amplitude scaling were carefully aligned between Python training code and ESP32 inference code

This experience reinforced the importance of matching the training domain as closely as possible to the deployment environment.

Quantization and Numerical Stability:

Although TensorFlow Lite provides automated INT8 quantization, achieving reliable inference required careful handling of scale and zero-point values. Several early inference attempts produced near-zero outputs or unstable predictions due to incorrect input quantization.

The following steps were critical:

Using a representative dataset during TFLite conversion

Explicitly applying the input tensor’s scale and zero-point on the ESP32

Verifying quantized inference against floating-point reference outputs

Once properly configured, INT8 quantization reduced the model size to 15,360 bytes and enabled fast inference without requiring floating-point hardware.

 Tensor Arena and Memory Allocation:

TensorFlow Lite Micro requires a statically allocated tensor arena. Determining the correct arena size involved iterative experimentation, as insufficient allocation resulted in silent failures during tensor initialization.

Through profiling and gradual adjustment, a stable configuration was achieved using approximately 40–50 KB of RAM. This ensured reliable model execution while preserving memory for other system components.

Toolchain and Build System Integration:

The project was developed using PlatformIO, which provided a modern build system, dependency management, and debugging environment for ESP32 development. However, integrating TensorFlow Lite Micro and custom-generated model data required careful configuration of compiler flags, memory sections, and source inclusion.

Key challenges included:

Managing large C arrays generated from the TFLite model

Resolving compilation issues related to alignment and memory attributes

Ensuring consistent builds across development iterations

Once resolved, PlatformIO proved to be a robust and scalable development environment for embedded machine learning projects.

Experimental Results and Evaluation:

Training was conducted over 30 epochs using a three-class dataset (“on”, “off”, and background noise). Performance improved steadily as the model learned temporal audio features directly from waveform data.

Final evaluation metrics:

Test Accuracy: 79.31%

Model Size: 15,360 bytes

Input Size: 16,000 samples

Output Classes: 3

Key training milestones included:

Rapid convergence after epoch 6

Stable validation accuracy despite model compression

Improved robustness after dataset expansion

Final deployment confirmation on ESP32:
=== NEW MODEL (79.31% accuracy) ===
Model: 15360 bytes
Allocating tensors...
Input size: 16000
Output size: 3
✅ Network ready!
These results confirm that meaningful keyword recognition performance can be achieved on low-power microcontrollers using compact neural networks and raw audio inputs.

Summary:

This project demonstrates a complete embedded machine learning workflow, from dataset design and neural network training to INT8 quantization and real-time inference on ESP32 hardware. The challenges encountered and resolved throughout development highlight the practical considerations required to deploy ML systems beyond simulation environments.

The final system achieves a strong balance between accuracy, efficiency, and robustness, validating the feasibility of real-time keyword recognition on constrained devices using TensorFlow Lite Micro and PlatformIO.