from torchvision import transforms, datasets
from torch.utils.data import DataLoader


def build_dataloaders_cifar(data_path, train_transforms, test_transforms, num_workers, batch_size):
    """Build dataloaders for CIFAR-10 dataset."""
    train_dataset = datasets.CIFAR10(data_path, train=True, download=True, transform=train_transforms)
    test_dataset = datasets.CIFAR10(data_path, train=False, download=True, transform=test_transforms)
    test_dataset_viz = datasets.CIFAR10(data_path, train=False, download=True)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, persistent_workers=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                 num_workers=num_workers, persistent_workers=True)

    class_names = train_dataset.classes
    return train_dataloader, test_dataloader, test_dataset_viz, class_names


def build_dataloaders_mnist(data_path, train_transforms, test_transforms, num_workers, batch_size):
    """Build dataloaders for MNIST dataset."""
    train_dataset = datasets.MNIST(data_path, train=True, download=True, transform=train_transforms)
    test_dataset = datasets.MNIST(data_path, train=False, download=True, transform=test_transforms)
    test_dataset_viz = datasets.MNIST(data_path, train=False, download=True)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, persistent_workers=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                 num_workers=num_workers, persistent_workers=True)

    class_names = train_dataset.classes
    return train_dataloader, test_dataloader, test_dataset_viz, class_names


def build_dataloaders_fmnist(data_path, train_transforms, test_transforms, num_workers, batch_size):
    """Build dataloaders for Fashion MNIST dataset."""
    train_dataset = datasets.FashionMNIST(data_path, train=True, download=True, transform=train_transforms)
    test_dataset = datasets.FashionMNIST(data_path, train=False, download=True, transform=test_transforms)
    test_dataset_viz = datasets.FashionMNIST(data_path, train=False, download=True,
                                              transform=transforms.Compose([transforms.ToTensor()]))

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, persistent_workers=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                 num_workers=num_workers, persistent_workers=True)

    class_names = train_dataset.classes
    return train_dataloader, test_dataloader, test_dataset_viz, class_names
