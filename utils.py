import torch
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import argparse

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--x_size', dest='x_size', type=int, default='784')
	parser.add_argument('--h_dim', dest='h_dim', type=int, default='400')
	parser.add_argument('--latent_dim', dest='latent_dim', type=int, default='20')
	parser.add_argument('--user_epochs', dest='user_epochs', type=int, default='5')
	parser.add_argument('--lr', dest='lr', type=float, default='1e-3')
	parser.add_argument('--batch_size', dest='batch_size', type=int, default='100')
	parser.add_argument('--weight_decay', dest='weight_decay', type=float, default='1e-5')
	parser.add_argument('--lowest_val_loss', dest='lowest_val_loss', type=float, default='1e10')
	parser.add_argument('--val_output', dest='val_output', action='store_true', default=True)
	parser.add_argument('--grey_scale', dest='grey_scale', action='store_true')
	parser.add_argument('--data_folder', dest='data_folder', type=str, default='/.data')
	parser.add_argument('--seed', dest='seed', type=int, default='3435')
	parser.add_argument('--val_subset', dest='val_subset', type=int, default='10000')
	parser.add_argument('--val_imgs', dest='val_imgs', type=int, default='3435')
	args = parser.parse_args()

	return args

def to_device(tens):
    if torch.cuda.is_available():
        tens = tens.cuda()
    return tens

# This implements either no grayscale or the standard grayscale transform
def process_data_grayscale_v1(args):
    # Download and tranform the data
    if (args.grey_scale == True):
        transform_set = transforms.Compose(
                          [transforms.Grayscale(),
                           transforms.ToTensor()])
    else:
        transform_set = transforms.ToTensor()
	
    train_dataset = datasets.MNIST(root=args.data_folder,
                                    train=True,
                                    transform=transform_set,
                                    download=True)

    val_dataset = datasets.MNIST(root=args.data_folder,
                                   train=False,
                                   transform=transform_set)

    print (train_dataset.train_data.shape, val_dataset.test_data.shape)

    train_iter = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_iter = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)

    return train_iter, val_iter

# This implements the grayscale transform from the original repo
def process_data_grayscale_v2(args):
    # Download the dataset and initialize train and test objects
    train = datasets.MNIST(root=args.data_folder,
                                    train=True,
                                    transform=transforms.ToTensor(),
                                    download=True)

    test = datasets.MNIST(root=args.data_folder,
                                   train=False,
                                   transform=transforms.ToTensor())

    train_stack = torch.stack([torch.bernoulli(d[0]) for d in train])
    train_label = torch.LongTensor([d[1] for d in train])

    # Test datset not used in this version
    test_stack = torch.stack([torch.bernoulli(d[0]) for d in test])
    test_label = torch.LongTensor([d[1] for d in test])

    # Generate an object with the user-defined number of examplars for validation
    val_stack = train_stack[-args.val_subset:].clone()
    val_label = train_label[-args.val_subset:].clone()

    train_stack = train_stack[:-args.val_subset]
    train_label = train_label[:-args.val_subset]

    # Convert to tensor and DataLoader
    train_tens = torch.utils.data.TensorDataset(train_stack, train_label)
    val_tens = torch.utils.data.TensorDataset(val_stack, val_label)
    test_tens = torch.utils.data.TensorDataset(test_stack, test_label)

    train_dl = torch.utils.data.DataLoader(train_tens, batch_size=args.batch_size, shuffle=True)
    val_dl = torch.utils.data.DataLoader(val_tens, batch_size=args.batch_size, shuffle=True)

    return train_dl, val_dl

