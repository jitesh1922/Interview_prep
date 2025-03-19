import torch
import torch.nn as nn
import torch.quantization

# 1. Define a simple model
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(10, 20)  # Input: 10, Output: 20
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(20, 5)   # Input: 20, Output: 5
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# 2. Create and prepare the model
model = SimpleNet()
model.eval()  # Set to evaluation mode

# 3. Specify quantization configuration
model.qconfig = torch.quantization.get_default_qconfig('fbgemm')  # For x86 CPUs
torch.quantization.prepare(model, inplace=True)  # Adds quantization stubs

# 4. Calibrate with sample data
calibration_data = torch.randn(10, 10)  # 10 samples, input size 10
with torch.no_grad():
    model(calibration_data)  # Run forward pass to collect statistics

# 5. Convert to quantized INT8 model
model_int8 = torch.quantization.convert(model, inplace=True)

# 6. Test inference
input_data = torch.randn(1, 10)  # Single input
output = model_int8(input_data)
print("Quantized output:", output)

# 7. Compare model sizes (optional)
fp32_size = sum(p.numel() * p.element_size() for p in SimpleNet().parameters())
int8_size = sum(p.numel() * p.element_size() for p in model_int8.parameters())
print(f"FP32 model size: {fp32_size} bytes")
print(f"INT8 model size: {int8_size} bytes")
