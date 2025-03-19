import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r=8):
        super(LoRALinear, self).__init__()
        
        # Original weight matrix (frozen)
        self.weight = nn.Parameter(torch.randn(out_features, in_features), requires_grad=False)
        self.bias = nn.Parameter(torch.zeros(out_features), requires_grad=False)
        
        # LoRA parameters
        self.A = nn.Parameter(torch.randn(out_features, r))  # Random initialization
        self.B = nn.Parameter(torch.zeros(r, in_features))  # Zero initialization
    
    def forward(self, x):
        # x: (batch_size, in_features)
        
        # Compute original output: W @ x
        Wx = torch.matmul(x, self.weight.t())  # (batch_size, out_features)
        
        # Compute LoRA adaptation: A @ (B @ x)
        Bx = torch.matmul(x, self.B.t())       # (batch_size, r)
        ABx = torch.matmul(Bx, self.A.t())     # (batch_size, out_features)
        
        # Combine and add bias
        return Wx + ABx + self.bias

# Example usage
batch_size, in_features, out_features, r = 32, 64, 128, 8
x = torch.randn(batch_size, in_features)
lora_layer = LoRALinear(in_features, out_features, r)
output = lora_layer(x)
print(output.shape)  # Should be (batch_size, out_features)
