# ==== Part 0: import libs
import argparse  # argparse is used to conveniently set our configurations
import cv2
import json
import os
import string
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


# ==== Part 1: data loader

# construct a dataset and a data loader, more details can be found in
# https://pytorch.org/tutorials/beginner/basics/data_tutorial.html?highlight=dataloader
class ListDataset(Dataset):
    def __init__(self, im_dir, file_path, norm_size=(32, 32)):
        """
        :param im_dir: path to directory with images
        :param file_path: json file containing image names and labels
        :param norm_size: image normalization size, (width, height)
        """

        # this time we will try to recognize 26 English letters (case-insensitive)
        letters = string.ascii_letters[-26:]  # ABCD...XYZ
        self.alphabet = {letters[i]: i for i in range(len(letters))}

        # get image paths and labels from json file
        with open(file_path, 'r') as f:
            imgs = json.load(f)
            im_names = list(imgs.keys())

            self.im_paths = [os.path.join(im_dir, im_name) for im_name in im_names]
            self.labels = list(imgs.values())

        self.norm_size = norm_size

    def __len__(self):
        return len(self.im_paths)

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'

        # read an image and convert it to grey scale
        im_path = self.im_paths[index]
        im = cv2.imread(im_path)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

        # image pre-processing, after pre-processing, the values of image pixels are converted to [0,1]
        im = cv2.resize(im, self.norm_size)
        # convert numpy image to pytorch tensor, and normalize values to [-1, 1]
        im = (torch.from_numpy(im).float() - 127.5) / 127.5
        # add the first channel dimension
        im = im.unsqueeze(0)

        # get the label of the current image
        # upper() is used to convert a letter into uppercase
        label = self.labels[index].upper()

        # convert an English letter into a number index
        label = self.alphabet[label]

        return im, label


def dataLoader(im_dir, file_path, norm_size, batch_size, workers=0):
    """
    :param im_dir: path to directory with images
    :param file_path: file with image paths and labels
    :param norm_size: image normalization size, (height, width)
    :param batch_size: batch size
    :param workers: number of workers for loading data in multiple threads
    :return: a data loader
    """

    dataset = ListDataset(im_dir, file_path, norm_size)
    return DataLoader(dataset,
                      batch_size=batch_size,
                      shuffle=True if 'train' in file_path else False,  # shuffle images only when training
                      num_workers=workers)


# ==== Part 2: construct a model
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.ReLU())
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU())
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU())
        self.pool3 = nn.AdaptiveAvgPool2d(1)
        self.layer4 = nn.Linear(in_features=32, out_features=26)  # 26 letters

    def forward(self, x, return_features=False):
        # x: input image with shape [b, 1, h, w]
        f1 = self.layer1(x)  # [b, 8, h, w]
        p1 = self.pool1(f1)  # [b, 8, h//2, w//2]
        f2 = self.layer2(p1)  # [b, 16, h//2, w//2]
        p2 = self.pool2(f2)  # [b, 16, h//4, w//4]
        f3 = self.layer3(p2)  # [b, 32, h//4, w//4]
        p3 = self.pool3(f3)  # [b, 32, 1, 1]
        out = self.layer4(p3.view(-1, 32))  # [b, 26]

        if return_features:
            return out, f1, f2, f3
        return out


# ==== Part 3: training and validation
def train_val(im_dir, train_file_path, val_file_path,
              norm_size, n_epochs, batch_size,
              lr, valInterval, device='cpu'):
    """
    The main training procedure
    ----------------------------
    :param im_dir: path to directory with images
    :param train_file_path: file list of training image paths and labels
    :param val_file_path: file list of validation image paths and labels
    :param norm_size: image normalization size, (height, width)
    :param n_epochs: number of training epochs
    :param batch_size: batch size of training and validation
    :param lr: learning rate
    :param valInterval: the frequency of validation, e.g., if valInterval = 5, then do validation after each 5 training epochs
    :param device: 'cpu' or 'cuda', we can use 'cpu' for our homework if GPU with cuda support is not available
    """

    # training and validation data loader
    trainloader = dataLoader(im_dir, train_file_path, norm_size, batch_size)
    valloader = dataLoader(im_dir, val_file_path, norm_size, batch_size)

    # initialize a model
    model = SimpleCNN()
    # put the model on CPU or GPU
    model = model.to(device)

    # loss function and optimizer
    ce_loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr)

    # training
    for epoch in range(n_epochs):
        # set the model in training mode
        model.train()

        # to save total loss in one epoch
        total_loss = 0.
        for step, (ims, labels) in enumerate(trainloader):  # get a batch of data

            # set data type and device
            ims, labels = ims.to(device), labels.to(device)

            # clear gradients in the optimizer
            optimizer.zero_grad()

            # run the forward process
            out = model(ims)

            # compute the cross entropy loss, and call backward propagation function
            loss = ce_loss(out, labels)
            loss.backward()

            # sum up of total loss, loss.item() return the value of the tensor as a standard python number
            # this operation is not differentiable
            total_loss += loss.item()

            # call a function to update the parameters of the models
            optimizer.step()

        # average of the total loss for iterations
        avg_loss = total_loss / len(trainloader)
        print('Epoch {:02d}: loss = {:.3f}'.format(epoch + 1, avg_loss))

        # validation
        if (epoch + 1) % valInterval == 0:

            # set the model in evaluation mode
            model.eval()

            n_correct = 0.  # number of images that are correctly classified
            n_ims = 0.  # number of total images

            with torch.no_grad():  # we do not need to compute gradients during validation

                for ims, labels in valloader:
                    ims, labels = ims.to(device), labels.to(device)
                    out = model(ims)
                    predictions = out.argmax(1)

                    # sum up the number of images correctly recognized
                    n_correct += torch.sum(predictions == labels)
                    # sum up the total image number
                    n_ims += ims.size(0)

            # show prediction accuracy
            print('Epoch {:02d}: validation accuracy = {:.1f}%'.format(epoch + 1, 100 * n_correct / n_ims))

    # save model parameters in a file
    if not os.path.exists('saved_models'):
        os.makedirs('saved_models')
    model_save_path = 'saved_models/recognition.pth'
    torch.save({'state_dict': model.state_dict()}, model_save_path)
    print('[Info] Model saved in {}\n'.format(model_save_path))


# ==== Part 4: test
def test(model_path, im_dir='data/images',
         test_file_path='data/test.json',
         norm_size=(32, 32), batch_size=8,
         device='cpu'):
    """
    Test procedure
    ---------------
    :param model_path: path of the saved model
    :param im_dir: path to directory with images
    :param test_file_path: file with test image paths and labels
    :param norm_size: image normalization size, (height, width)
    :param batch_size: test batch size
    :param device: 'cpu' or 'cuda'
    """

    # load configurations from saved model
    checkpoint = torch.load(model_path)

    # initialize the model
    model = SimpleCNN()
    # load model parameters we saved in model_path
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)
    print('[Info] Load model from {}'.format(model_path))

    # enter the evaluation mode
    model.eval()

    # test loader
    testloader = dataLoader(im_dir, test_file_path, norm_size, batch_size)

    # run the test process
    n_correct = 0.
    n_ims = 0.

    with torch.no_grad():  # we do not need to compute gradients during test stages
        for ims, labels in testloader:
            ims, labels = ims.to(device), labels.to(device)
            out = model(ims)
            predictions = out.argmax(1)
            n_correct += torch.sum(predictions == labels)
            n_ims += ims.size(0)

    print('[Info] Test accuracy = {:.1f}%'.format(100 * n_correct / n_ims))


def predict(model_path, im_path, norm_size=(32, 32), device='cpu'):
    # read image and preprocess
    assert os.path.exists(im_path), '{} not exists!'.format(im_path)
    im = cv2.imread(im_path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im = cv2.resize(im, norm_size)
    im = (torch.from_numpy(im).float() - 127.5) / 127.5
    im = im.view(1, 1, norm_size[1], norm_size[0])

    # load configurations from saved model
    checkpoint = torch.load(model_path)

    # initialize the model
    model = SimpleCNN()
    # load model parameters we saved in model_path
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)
    print('[Info] Load model from {}'.format(model_path))

    # run the forward process
    model.eval()
    with torch.no_grad():
        out = model(im)
    prediction = out[0].argmax().item()
    prediction = chr(prediction + ord('A'))

    print('{}: {}'.format(os.path.basename(im_path), prediction))


if __name__ == '__main__':
    # set random seed for reproducibility
    seed = 2022
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    # set configurations
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', help='train or test')
    parser.add_argument('--im_dir', type=str, default='data/images',
                        help='path to directory with images')
    parser.add_argument('--train_file_path', type=str, default='data/train.json',
                        help='file list of training image paths and labels')
    parser.add_argument('--val_file_path', type=str, default='data/validation.json',
                        help='file list of validation image paths and labels')
    parser.add_argument('--test_file_path', type=str, default='data/test.json',
                        help='file list of test image paths and labels')
    parser.add_argument('--batchsize', type=int, default=8, help='batch size')
    parser.add_argument('--device', type=str, default='cpu', help='cpu or cuda')

    # configurations for training and test
    parser.add_argument('--norm_size', type=str, default='32,32',
                        help='image normalization size, height,width, splitted by comma')
    parser.add_argument('--epoch', type=int, default=30, help='number of training epochs')
    parser.add_argument('--valInterval', type=int, default=10, help='the frequency of validation')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--model_path', type=str, default='saved_models/recognition.pth',
                        help='path of a saved model')

    # configurations for prediction
    parser.add_argument('--im_path', type=str, default='', help='path of an image to be recognized')

    opt = parser.parse_args()

    # -- run the code for training and validation
    if opt.mode == 'train':
        train_val(im_dir=opt.im_dir,
                  train_file_path=opt.train_file_path,
                  val_file_path=opt.val_file_path,
                  norm_size=(int(opt.norm_size.split(',')[0]), int(opt.norm_size.split(',')[1])),
                  n_epochs=opt.epoch,
                  batch_size=opt.batchsize,
                  lr=opt.lr,
                  valInterval=opt.valInterval,
                  device=opt.device)

    # -- test the saved model
    elif opt.mode == 'test':
        test(model_path=opt.model_path,
             im_dir=opt.im_dir,
             test_file_path=opt.test_file_path,
             norm_size=(int(opt.norm_size.split(',')[0]), int(opt.norm_size.split(',')[1])),
             batch_size=opt.batchsize,
             device=opt.device)

    elif opt.mode == 'predict':
        predict(model_path=opt.model_path,
                im_path=opt.im_path,
                norm_size=(int(opt.norm_size.split(',')[0]), int(opt.norm_size.split(',')[1])),
                device=opt.device)

    else:
        raise NotImplementedError('mode should be train or test')
