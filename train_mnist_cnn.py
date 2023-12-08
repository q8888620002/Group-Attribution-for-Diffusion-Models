"""Train MNIST CNN classifier."""

import torch
from lightning.pytorch import seed_everything
from torch import nn, optim

from diffusion.models import CNN
from utils import create_dataloaders


def evaluate_acc(model, dataloader, device):
    """Evaluate a trained model's accuracy."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs, _ = model(images)
            preds = outputs.argmax(1)

            total += labels.size(0)
            correct += (preds == labels).sum().item()
    return correct / total


if __name__ == "__main__":
    seed_everything(42, workers=True)
    num_epochs = 10
    train_dataloader, test_dataloader = create_dataloaders(
        dataset_name="mnist",
        batch_size=128,
        num_workers=4,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cnn = CNN().to(device)

    total_step = len(train_dataloader)
    optimizer = optim.Adam(cnn.parameters(), lr=1e-4)
    loss_func = nn.CrossEntropyLoss()

    # Training loop.
    for epoch in range(num_epochs):
        cnn.train()
        for step, (images, labels) in enumerate(train_dataloader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = cnn(images)[0]
            loss = loss_func(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (step + 1) % 100 == 0:
                print(
                    "Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}".format(
                        epoch + 1, num_epochs, step + 1, total_step, loss.item()
                    )
                )

    # Evaluation.
    train_acc = evaluate_acc(cnn, train_dataloader, device)
    test_acc = evaluate_acc(cnn, test_dataloader, device)

    print(f"Training accuracy: {train_acc:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")

    # Save model weights.
    torch.save(cnn.state_dict(), f"eval/models/epochs={num_epochs}_cnn_weights.pt")
