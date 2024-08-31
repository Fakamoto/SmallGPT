from model import SmallGPTLanguageModel
import torch

# Load the model and its weights
model = SmallGPTLanguageModel()
model.load_state_dict(torch.load('darwin_model.pth'))
model.eval()  # Set the model to evaluation mode


token_limit = 300
input_text = "Animals are "
input_text = ""
if input_text:
    generated_text = model.generate_from_text(input_text, max_new_tokens=token_limit)
else:
    generated_text = model.speak(max_new_tokens=token_limit)

print(generated_text)