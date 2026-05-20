import os
import sys
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from groq import Groq

# find src/ from wherever this file lives
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', 'src'))

from model import ChestXrayModel
from dataset import LABELS

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# same preprocessing used during training (no augmentation)
_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


def load_model(checkpoint_path):
    """Load DenseNet model from a training checkpoint."""
    model = ChestXrayModel(num_classes=15, pretrained=False)
    ckpt = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(DEVICE).eval()
    return model, ckpt.get('epoch', '?'), ckpt.get('loss', float('nan'))


def preprocess(image_pil, age_raw, gender_str):
    """
    Convert a PIL image + raw clinical inputs into model-ready tensors.
    Gender encoding matches exploration.ipynb: M=0, F=1
    Age normalization matches exploration.ipynb: age / 100.0
    """
    image  = _TRANSFORM(image_pil.convert('RGB'))
    age    = torch.tensor([age_raw / 100.0], dtype=torch.float32)
    gender = torch.tensor([0.0 if gender_str == 'M' else 1.0], dtype=torch.float32)
    return image, age, gender


def predict(model, image, age, gender):
    """Run inference and return sigmoid probabilities, shape (15,)."""
    with torch.no_grad():
        logits = model(
            image.unsqueeze(0).to(DEVICE),
            age.unsqueeze(0).to(DEVICE),
            gender.unsqueeze(0).to(DEVICE)
        )
    return torch.sigmoid(logits).squeeze().cpu().numpy()


def get_gradcam(model, image, age, gender, class_idx):
    """
    Generate a GradCAM heatmap for the given class index.
    Targets the last DenseNet dense block before global average pooling.
    Returns a (224, 224) float array in [0, 1].
    """
    activations, gradients = {}, {}

    def fwd_hook(m, inp, out):
        activations['v'] = out.detach()

    def bwd_hook(m, grad_in, grad_out):
        gradients['v'] = grad_out[0].detach()

    # hook the final feature layer
    layer = model.features.denseblock4
    h1 = layer.register_forward_hook(fwd_hook)
    h2 = layer.register_full_backward_hook(bwd_hook)

    out = model(
        image.unsqueeze(0).to(DEVICE),
        age.unsqueeze(0).to(DEVICE),
        gender.unsqueeze(0).to(DEVICE)
    )
    model.zero_grad()
    out[0, class_idx].backward()

    acts    = activations['v']                          
    grads   = gradients['v']                            
    weights = grads.mean(dim=(2, 3), keepdim=True)      
    cam     = torch.relu((weights * acts).sum(dim=1).squeeze()).cpu().numpy()

    # normalize and resize to image size
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    cam = np.array(
        Image.fromarray((cam * 255).astype(np.uint8)).resize((224, 224), Image.BILINEAR)
    ) / 255.0

    h1.remove()
    h2.remove()
    return cam


def explain_with_groq(probs, api_key):
    """
    Send model predictions to Groq (llama-3.3-70b) and get a short explanation.
    Returns the explanation string.
    """
    client = Groq(api_key=api_key)

    findings = [
        f"{LABELS[i]} ({probs[i]:.0%})"
        for i in range(len(LABELS)) if probs[i] > 0.5
    ]
    findings_str = ', '.join(findings) if findings else 'No significant findings'
    all_probs    = '\n'.join(f'  - {LABELS[i]}: {probs[i]:.1%}' for i in range(len(LABELS)))

    prompt = f"""A chest X-ray classifier (research/educational demo, NOT clinical) produced:

Findings above 50% threshold: {findings_str}

All class probabilities:
{all_probs}

In 2-3 sentences, briefly explain what these findings could mean, as if talking to a medical student.
Remind the reader that this model is partially trained and results are not clinically valid."""

    response = client.chat.completions.create(
        model='llama-3.3-70b-versatile',
        messages=[{'role': 'user', 'content': prompt}],
        max_tokens=220,
    )
    return response.choices[0].message.content
