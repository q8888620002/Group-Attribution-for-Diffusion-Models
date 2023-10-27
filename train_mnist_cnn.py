import torch
from torch import optim
from torch import nn

from model import CNN
from utils import create_mnist_dataloaders









if  __name__ == "__main__":


    training_dataset, testing_dataset = create_mnist_dataloaders(
            batch_size=128,
            num_workers=4,
        )
    device = "cuda:1"
    num_epochs = 10

    cnn = CNN().to(device)

    # Train the model
    total_step = len(training_dataset)


    optimizer = optim.Adam(cnn.parameters(), lr = 1e-4)
    loss_func = nn.CrossEntropyLoss()


    for epoch in range(num_epochs):

        cnn.train()

        for i, (images, labels) in enumerate(training_dataset):

            # gives batch data, normalize x when iterate train_loader
            x = images.to(device)
            y = labels.to(device)

            output = cnn(x)[0]

            loss = loss_func(output, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 100 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                       .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

    cnn.eval()

    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in testing_dataset:

            images = images.to(device)
            labels = labels.to(device)
            outputs, _ = cnn(images)

            _, predicted = outputs.max(1)

            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))

    ## Save model weights

    torch.save(cnn.state_dict(), f"models/epochs={num_epochs}_cnn_weights.pt")

