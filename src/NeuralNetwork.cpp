#include "NeuralNetwork.h"
#include <Arduino.h>
#include "model_data.h"

// Undefine conflicting Arduino macro BEFORE TensorFlow includes
#undef DEFAULT

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

//  global instances
namespace {
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;
}

NeuralNetwork::NeuralNetwork() {
    Serial.begin(9600);
    delay(100);
      
    Serial.println("\n=== MODEL FINGERPRINT CHECK ===");
    
    // 1. Check size
    Serial.printf("Size claim: %u bytes\n", NEW_MODEL_79_percent_len);
    
    // 2. Check TFLite signature (bytes 4-7 = "TFL3")
    Serial.print("Bytes 4-7 (should be 54 46 4C 33): ");
    for(int i = 4; i < 8; i++) {
        Serial.printf("%02X ", NEW_MODEL_79_percent[i]);
    }
    Serial.println();
    
    // 3. Check for "conv1d" in model (new model signature)
    // Look for operation type 3 (CONV_2D) which should be CONV_1D in new model
    Serial.println("\nScanning for CONV1D operations...");
    
    // Load model temporarily just to check
    const tflite::Model* temp_model = tflite::GetModel(NEW_MODEL_79_percent);
    auto* subgraph = temp_model->subgraphs()->Get(0);
    
    int conv1d_count = 0;
    int expand_dims_count = 0;
    int unknown_count = 0;
    
    for(int i = 0; i < subgraph->operators()->size(); i++) {
        auto* op = subgraph->operators()->Get(i);
        auto* opcode = temp_model->operator_codes()->Get(op->opcode_index());
        
        if(opcode->builtin_code() == 3) {  // CONV_2D (or CONV_1D in new model)
            conv1d_count++;
        } else if(opcode->builtin_code() == 10) {  // EXPAND_DIMS
            expand_dims_count++;
        } else if(opcode->builtin_code() == 70) {  // Unknown
            unknown_count++;
        }
    }
    
    Serial.printf("CONV1D operations: %d\n", conv1d_count);
    Serial.printf("EXPAND_DIMS operations: %d\n", expand_dims_count);
    Serial.printf("Unknown (code 70) operations: %d\n", unknown_count);
    
    // 4. VERDICT (checks if my model changed (for me))
    Serial.println("\n=== VERDICT ===");
    if(unknown_count > 0 || expand_dims_count > 0) {
        Serial.println("❌ STILL OLD MODEL!");
        Serial.println("You're loading the broken model with code 70 operations.");
    } else if(conv1d_count >= 3) {
        Serial.println("✅ NEW 79% MODEL DETECTED!");
        Serial.println("But something else is wrong...");
    } else {
        Serial.println("⚠️  UNKNOWN MODEL TYPE");
    }
    
    // NOW continue with your existing constructor code...
    Serial.println("\n=== Continuing initialization ===");
    Serial.printf("Model: %u bytes\n", NEW_MODEL_79_percent_len);
    
    Serial.println("\n=== NEURAL NETWORK ===");
    Serial.printf("Model: %u bytes\n", NEW_MODEL_79_percent_len);
    
    // 1. Load model
    model = tflite::GetModel(NEW_MODEL_79_percent);
    if (!model) {
        Serial.println("Failed to load model");
        while(1);
    }
    
    // 2. Use the COMPLETE ops resolver (includes ALL operations)
    static tflite::AllOpsResolver resolver;
    
    // 3. Setup error reporter
    static tflite::MicroErrorReporter micro_error_reporter;
    error_reporter = &micro_error_reporter;
    
    // 4. Create interpreter
    interpreter = new tflite::MicroInterpreter(
        model,
        resolver,  // Complete resolver has ALL ops
        tensor_arena,
        sizeof(tensor_arena),
        error_reporter
    );
    
    // 5. Allocate tensors
    Serial.println("Allocating tensors...");
    TfLiteStatus status = interpreter->AllocateTensors();
    if (status != kTfLiteOk) {
        Serial.println("Tensor allocation failed!");
        
        // Show what's in the model
        Serial.println("\n=== MODEL DEBUG ===");
        auto* subgraph = model->subgraphs()->Get(0);
        for(int i = 0; i < subgraph->operators()->size(); i++) {
            auto* op = subgraph->operators()->Get(i);
            auto* opcode = model->operator_codes()->Get(op->opcode_index());
            
            Serial.printf("Op %d: Builtin code %d - ", i, opcode->builtin_code());
            
            // Common codes
            switch(opcode->builtin_code()) {
                case 10: Serial.println("EXPAND_DIMS"); break;
                case 3: Serial.println("CONV_2D"); break;
                case 25: Serial.println("MAX_POOL_2D"); break;
                case 14: Serial.println("FULLY_CONNECTED"); break;
                case 40: Serial.println("SOFTMAX"); break;
                case 33: Serial.println("RELU"); break;
                case 34: Serial.println("RESHAPE"); break;
                default: Serial.printf("Unknown\n");
            }
        }
        
        // If EXPAND_DIMS is code 10, we need to handle it
        Serial.println("\nSOLUTION: If you see EXPAND_DIMS (code 10):");
        Serial.println("1. Retrain model without ExpandDims layer");
        Serial.println("2. Or use newer TensorFlow version");
        
        while(1);
    }
    
    // 6. Get tensors
    input = interpreter->input(0);
    output = interpreter->output(0);
    
    Serial.printf("Input size: %d\n", getInputSize());
    Serial.printf("Output size: %d\n", getOutputSize());
    
    if(getInputSize() == 16000) {
        Serial.println("✅ Input matches training");
    }
    
    Serial.println("✅ Network ready!");
}

NeuralNetwork::~NeuralNetwork() {
    if(interpreter) {
        delete interpreter;
    }
}

void NeuralNetwork::predict() {
    if(interpreter->Invoke() != kTfLiteOk) {
        Serial.println("Predict failed");
    }
}

float* NeuralNetwork::getInputBuffer() {
    return input->data.f;
}

float* NeuralNetwork::getOutputBuffer() {
    return output->data.f;
}

int NeuralNetwork::getInputSize() {
    int size = 1;
    for(int i = 0; i < input->dims->size; i++) {
        size *= input->dims->data[i];
    }
    return size;
}

int NeuralNetwork::getOutputSize() {
    int size = 1;
    for(int i = 0; i < output->dims->size; i++) {
        size *= output->dims->data[i];
    }
    return size;
}
