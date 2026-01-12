from cookie.model import Model
from cookie.data import corrupt_mnist
import torch
import typer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from torch import nn, optim
import matplotlib.pyplot as plt
import numpy as np

app = typer.Typer()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def visualize(model_checkpoint:str):
    model = Model().to(DEVICE)
    model.load_state_dict(torch.load(model_checkpoint))
    model.eval()
    model.fc1 = torch.nn.Identity()

    _, test_set = corrupt_mnist()

    embeddings,targets=[],[]
    with torch.inference_mode(): # stricter no_grad()
        for batch in torch.utils.data.DataLoader(test_set,batch_size=32):
            img,label=batch
            embedding=model(img)
            embeddings.append(embedding)
            targets.append(label)

        embeddings=torch.cat(embeddings).numpy()
        targets=torch.cat(targets).numpy()


        pca=PCA(n_components=50)
        X=pca.fit_transform(embeddings)
        X=TSNE(n_components=2).fit_transform(X)

        print(X.shape)
        for label in np.unique(targets):
            mask=targets==label
            pc1=X[mask,0]
            pc2=X[mask,1]
            plt.scatter(pc1,pc2,label=str(label),marker='.')
        plt.legend()
        plt.title('Embeddings visualized in two dimension')
        plt.ylabel('PC2')
        plt.xlabel('PC1')
        plt.savefig('reports/figures/PCA.png')










    
    



    


if __name__=="__main__":
    typer.run(visualize)