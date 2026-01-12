from cookie.model import Model
from cookie.data import corrupt_mnist
import torch
import typer

from torch import nn, optim
import matplotlib.pyplot as plt
import numpy as np

app = typer.Typer()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

def evaluate(model_checkpoint: str) -> None:
    """Evaluate a trained model."""
    print("Evaluating like my life depends on it")


    # Implement evaluation logic here
    model = Model().to(DEVICE)
    model.load_state_dict(torch.load(model_checkpoint))
    model.eval()

    loss_fn = nn.CrossEntropyLoss()

    _, test_set = corrupt_mnist()
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=32)


    statistics={'test_loss':[],'test_accuracy':[]}
    correct, total = 0, 0
    for i,(image, label) in enumerate(test_dataloader):
        img,label=image.to(DEVICE),label.to(DEVICE)

        pred = model(img)
        loss = loss_fn(pred,label)

        statistics['test_loss'].append(loss.item())
        
        correct += (pred.argmax(dim=1) == label).float().sum().item()
        total += label.size(0)
    print(f"Test accuracy: {correct / total}")


if __name__ == "__main__":
    typer.run(evaluate)  # den tager det argument fra teksten i task.py (model path) 
                            # og giver det til funktionen