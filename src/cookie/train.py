from cookie.model import Model
from cookie.data import corrupt_mnist
import torch
import typer
import wandb



from torch import nn, optim
import matplotlib.pyplot as plt

app = typer.Typer()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

def train(lr: float = 1e-3, batch_size: int = 32, epochs: int = 10) -> None:
    """Train a model on MNIST."""
    print("Training day and night")
    print(lr)

    # logging
    project="cookie_mlops"
    config={'epochs':epochs,'lr':lr,'batch size':batch_size}
    wandb.init(project=project, config=config)

    # Implement training loop here
    model = Model().to(DEVICE)
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_set, _ = corrupt_mnist()
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)

    statistics = {"train_loss": [], "train_accuracy": []}

    for e in range(epochs):
        epoch_loss=[]
        epoch_accuracy=[]
        for i,(image, label) in enumerate(train_dataloader):
            img,label=image.to(DEVICE),label.to(DEVICE)
            optimizer.zero_grad()

            pred = model(img)
            loss = loss_fn(pred,label)
            loss.backward()
            optimizer.step()

            statistics["train_loss"].append(loss.item())
            epoch_loss.append(loss.item())
            

            accuracy = (pred.argmax(dim=1) == label).float().mean().item()
            statistics['train_accuracy'].append(accuracy)
            epoch_accuracy.append(accuracy)

            # log
            wandb.log({'loss':loss.item(),'accuracy':accuracy})


        #training_losses.append(sum(running_loss)/len(train_set))
        print(f'Epoch {e+1}: Training Loss {sum(epoch_loss)/len(epoch_loss)}. Training accuracy {sum(epoch_accuracy)/len(epoch_accuracy)}')
    print("Training complete")
    torch.save(model.state_dict(), "models/model.pth")
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(statistics["train_loss"])
    axs[0].set_title("Train loss")
    axs[1].plot(statistics["train_accuracy"])
    axs[1].set_title("Train accuracy")
    fig.savefig("reports/figures/training_statistics.png")



if __name__ == "__main__":
    train()
