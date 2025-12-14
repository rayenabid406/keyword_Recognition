import os

print("Converting esp32_simple_model.tflite to C array...")

# Read the new model file
with open('esp32_simple_model.tflite', 'rb') as f:
    data = f.read()

print(f"New model size: {len(data)} bytes")

# Create C header file
with open('model_data_new.h', 'w') as f:
    f.write('#ifndef MODEL_DATA_H\n')
    f.write('#define MODEL_DATA_H\n\n')
    f.write('#include <cstdint>\n\n')
    f.write('// Auto-generated from esp32_simple_model.tflite\n')
    f.write('// Model size: ' + str(len(data)) + ' bytes\n')
    f.write('// Accuracy: 79.31%\n\n')
    f.write('alignas(16) const unsigned char converted_model_tflite[] = {\n')
    
    # Write bytes in hex format
    for i in range(0, len(data), 12):
        line_bytes = data[i:min(i+12, len(data))]
        hex_line = ', '.join([f'0x{b:02x}' for b in line_bytes])
        f.write('    ' + hex_line)
        if i + 12 < len(data):
            f.write(',')
        f.write('\n')
    
    f.write('};\n\n')
    f.write(f'const unsigned int converted_model_tflite_len = {len(data)};\n\n')
    f.write('#endif // MODEL_DATA_H\n')

print("Created model_data_new.h")
print("Replace your existing model_data.cc/.h with this new data!")