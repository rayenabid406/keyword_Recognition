#include <Arduino.h>
#include "driver/i2s.h"
#include "NeuralNetwork.h"
#include <math.h>

// ================= CONFIGURATION =================
#define LED_PIN 2
#define I2S_WS 15
#define I2S_SD 32
#define I2S_SCK 14

// CRITICAL: MUST be 16000 for 1-second audio (matches training)
#define AUDIO_BUFFER_SIZE 16000
#define SAMPLE_RATE 16000
#define PREDICTION_INTERVAL 1500  // ms between predictions

NeuralNetwork *nn = nullptr;

// I2S config for INMP441 microphone
i2s_config_t i2s_config = {
    .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
    .sample_rate = SAMPLE_RATE,
    .bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT,
    .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
    .communication_format = I2S_COMM_FORMAT_I2S,
    .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
    .dma_buf_count = 4,
    .dma_buf_len = 1024,
    .use_apll = false,
    .tx_desc_auto_clear = false,
    .fixed_mclk = 0
};

i2s_pin_config_t pin_config = {
    .bck_io_num = I2S_SCK,
    .ws_io_num = I2S_WS,
    .data_out_num = -1,
    .data_in_num = I2S_SD
};

// Audio buffer for 1 second (16000 samples at 16kHz)
int16_t audio_buffer[AUDIO_BUFFER_SIZE];

void setup() {
    Serial.begin(9600);
    delay(3000);  // Wait for Serial Monitor
    
    Serial.println("\n\n=== ESP32 VOICE COMMAND SYSTEM ===");
    Serial.println("Trained model: 62% accuracy");
    Serial.println("Speak 'on' or 'off' clearly");
    Serial.println("====================================\n");
    
    pinMode(LED_PIN, OUTPUT);
    digitalWrite(LED_PIN, LOW);
    
    // Initialize I2S
    esp_err_t err = i2s_driver_install(I2S_NUM_0, &i2s_config, 0, NULL);
    if(err != ESP_OK) {
        Serial.printf("I2S init failed: %d\n", err);
        while(1);
    }
    
    err = i2s_set_pin(I2S_NUM_0, &pin_config);
    if(err != ESP_OK) {
        Serial.printf("I2S pin config failed: %d\n", err);
        while(1);
    }
    
    Serial.println("I2S initialized");
    
    // Initialize Neural Network (with your converted_model_tflite)
    Serial.println("Loading neural network...");
    nn = new NeuralNetwork();
    
    // Verify model configuration
    int input_size = nn->getInputSize();
    int output_size = nn->getOutputSize();
    
    Serial.printf("\n=== MODEL INFO ===\n");
    Serial.printf("Input size: %d (expected: 16000)\n", input_size);
    Serial.printf("Output size: %d (expected: 3)\n", output_size);
    
    if(input_size == 16000) {
        Serial.println("✅ Model matches training configuration");
    } else {
        Serial.printf("❌ ERROR: Model expects %d inputs, not 16000\n", input_size);
        Serial.println("Voice commands will not work!");
        while(1);
    }
    
    if(output_size == 3) {
        Serial.println("✅ Model has 3 outputs (on/off/background)");
    }
    
    Serial.println("\n=== SYSTEM READY ===\n");
    Serial.println("Instructions:");
    Serial.println("1. Speak clearly, 10-20cm from microphone");
    Serial.println("2. Say 'on' to turn LED ON");
    Serial.println("3. Say 'off' to turn LED OFF");
    Serial.println("4. Watch confidence scores (need >40%)");
    Serial.println("====================================\n");
}

// ================= PROCESS 1 SECOND OF AUDIO =================
void processAudio() {
    static int processing_count = 0;
    
    // Capture EXACTLY 1 second (16000 samples)
    size_t total_bytes = 0;
    unsigned long start_time = millis();
    
    while(total_bytes < sizeof(audio_buffer)) {
        size_t bytes_read;
        i2s_read(I2S_NUM_0,
                (uint8_t*)audio_buffer + total_bytes,
                sizeof(audio_buffer) - total_bytes,
                &bytes_read, 50);
        total_bytes += bytes_read;
        
        // Timeout after 2 seconds
        if(millis() - start_time > 2000) {
            Serial.println("Audio capture timeout!");
            return;
        }
    }
    
    processing_count++;
    
    // Calculate audio energy (loudness check)
    float energy = 0;
    for(int i = 0; i < 100; i++) {
        energy += abs(audio_buffer[i]);
    }
    energy /= 100.0f;
    
    Serial.printf("[%04d] Energy: %5.0f ", processing_count, energy);
    
    // Skip if too quiet (background noise)
    if(energy < 100) {
        Serial.println("(too quiet, skipping)");
        return;
    }
    
    Serial.print("-> Processing... ");
    
    // ===== CRITICAL: Normalize audio to [-1, 1] =====
    float* input = nn->getInputBuffer();
    for(int i = 0; i < 16000; i++) {
        input[i] = audio_buffer[i] / 32768.0f;
    }
    
    // ===== RUN NEURAL NETWORK =====
    nn->predict();
    float* outputs = nn->getOutputBuffer();
    int output_size = nn->getOutputSize();
    
    // Display raw outputs
    Serial.print("Outputs: ");
    for(int i = 0; i < output_size; i++) {
        Serial.printf("[%d]:%.3f ", i, outputs[i]);
    }
    
    // Find best prediction
    int best_class = 0;
    float best_score = outputs[0];
    for(int i = 1; i < output_size; i++) {
        if(outputs[i] > best_score) {
            best_score = outputs[i];
            best_class = i;
        }
    }
    
    // Calculate confidence (simple softmax)
    float confidence = best_score;
    if(output_size > 1) {
        float sum = 0;
        for(int i = 0; i < output_size; i++) {
            sum += outputs[i];
        }
        confidence = best_score / sum;
    }
    
    Serial.printf("-> Class %d (%.0f%%)", best_class, confidence * 100);
    
    // ===== DECISION: Act only with sufficient confidence =====
    // With 62% accuracy model, we use 40% confidence threshold
    if(confidence > 0.40) {
        if(best_class == 0) {  // "on"
            digitalWrite(LED_PIN, HIGH);
            Serial.println(" -> LED ON");
        } else if(best_class == 1) {  // "off"
            digitalWrite(LED_PIN, LOW);
            Serial.println(" -> LED OFF");
        } else {  // "background" or other
            Serial.println(" -> Background noise (ignoring)");
        }
    } else {
        Serial.println(" -> Low confidence (ignoring)");
    }
}

// ================= MAIN LOOP =================
void loop() {
    static unsigned long last_prediction = 0;
    
    // Only process audio every PREDICTION_INTERVAL milliseconds
    unsigned long now = millis();
    if(now - last_prediction < PREDICTION_INTERVAL) {
        delay(10);  // Small delay to prevent busy loop
        return;
    }
    
    last_prediction = now;
    
    // Process 1 second of audio
    processAudio();
}
