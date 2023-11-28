import torch
from torch import optim
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from diffusion.models import CNN
from diffusion.diffusions import DDPM
from ddpm_config import DDPMConfig




def binary_labels(labels):
    # Convert labels to binary: 1 for class 2 and 3, 0 for all other classes
    return (labels == 2)


if __name__ == "__main__":
    batch_size = 128
    num_workers = 4
    device = "cuda:2"
    num_epochs = 20
    

    config = DDPMConfig.mnist_config
    
    model = DDPM(
        timesteps=config['timesteps'],
        base_dim=config['base_dim'],
        channel_mult=config['channel_mult'],
        image_size=config['image_size'],
        in_channels=config['in_channels'],
        out_channels=config['out_channels'],
        attn=config['attn'],
        attn_layer=config['attn_layer'],
        num_res_blocks=config['num_res_blocks'],
        dropout=config['dropout'],
    ).to(device)

    ckpt=torch.load(config['trained_model'])
    model.load_state_dict(ckpt["model"])

    # Load MNIST data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalization for MNIST
    ])

    # Create dataloaders
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Initialize the CNN model
    cnn = CNN(10).to(device)

    # Loss and optimizer
    optimizer = optim.Adam(cnn.parameters(), lr=1e-4)
    # loss_func = nn.BCEWithLogitsLoss()

    loss_func = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(num_epochs):
        cnn.train()
        for i, (images, labels) in enumerate(train_loader):
            # Noisify images using the diffusion model
            images= images.to(device)
            labels = labels.to(device)

            noise = torch.randn_like(images).to(device)
            # Pick random timesteps for the diffusion process
            
            t = torch.randint(0, model.timesteps, (images.shape[0],), device=device)
            # Noisify images
            
            images = model._forward_diffusion(images, t, noise)

            # labels = binary_labels(labels).unsqueeze(1).float().to(device)

            # Forward pass
            outputs = cnn(images)[0]

            loss = loss_func(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')

    # Test the model
    cnn.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            # Noisify images using the diffusion model
            images = images.to(device)
            labels = labels.to(device)


            noise = torch.randn_like(images).to(device)
            # Pick random timesteps for the diffusion process            
            t = torch.randint(0, model.timesteps, (images.shape[0],), device=device)
            # Noisify images
            
            images = model._forward_diffusion(images, t, noise)

            # labels = binary_labels(labels).unsqueeze(1).float().to(device)

            outputs = cnn(images)[0]

            _, predicted = torch.max(outputs.data, 1)
            # predicted = outputs.squeeze() >= 0.5

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Test Accuracy of the model on the 10000 test images: {100 * correct / total} %')

    # Save the model checkpoint
    torch.save(cnn.state_dict(), 'eval/models/cnn_mnist_noisy.pth')
