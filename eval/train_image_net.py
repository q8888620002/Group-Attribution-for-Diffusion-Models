import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import models
from torch import nn, optim


if  __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load the pre-trained ResNet34 model
    model = models.resnet34(pretrained=True)

    num_classes = 10
    num_epochs = 20

    model.fc = nn.Linear(model.fc.in_features, num_classes)

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(), 
        lr=0.001, 
        momentum=0.9
    )

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR10( 
        root = '/projects/leelab/mingyulu/data_att/cifar',
        train=True,
        download=True, 
        transform=transform
    )
    

    trainloader = torch.utils.data.DataLoader(
        trainset, 
        batch_size=4,
        shuffle=True,
        num_workers=2
    )


    testset = torchvision.datasets.CIFAR10( 
        root = '/projects/leelab/mingyulu/data_att/cifar',
        train=False,
        download=True, 
        transform=transform
    )
    

    testloader = torch.utils.data.DataLoader(
        trainset, 
        batch_size=4,
        shuffle=False,
        num_workers=2
    )

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for i, (inputs, labels) in enumerate(trainloader, 0):

            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Print statistics every 2000 mini-batches
            if i % 2000 == 1999:  
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    print('Finished Training')

    # Testing the model
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:

            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

    torch.save(model.state_dict(), f"eval/models/epochs={num_epochs}_cifar_weights.pt")

