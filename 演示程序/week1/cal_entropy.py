import argparse
import cv2
import json
import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from recognition import dataLoader, SimpleCNN


def im_entropy(im):
    """
    Calculate entropy of one single image
    :param im: a greyscale image in numpy format
    :return: entropy of the image
    """

    h, w = im.shape
    hist = np.histogram(im.reshape(-1), bins=256)[0]
    probs = hist / (h * w)
    probs = probs[probs > 0]
    ent = np.sum(-probs * np.log(probs))
    return ent


def im_entropy_dataset(im_dir, file_path, norm_size):
    """
    Calculate the average entropy of images in a dataset
    :param im_dir: path to directory with images
    :param file_path: json file containing image names and labels
    :param norm_size: image normalization size, (width, height)
    :return: the average entropy
    """

    # get image paths from json file
    with open(file_path, 'r') as f:
        imgs = json.load(f)
        im_names = list(imgs.keys())
        im_paths = [os.path.join(im_dir, im_name) for im_name in im_names]

    ent = 0.
    # calculate entropy of each image
    for im_path in im_paths:
        im = cv2.imread(im_path)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        im = cv2.resize(im, norm_size)
        ent += im_entropy(im)

    return ent / len(im_paths)


def label_entropy_random(n_class=26):
    """
    We randomly guess a label for each input.
    :param n_class: number of class
    :return: the entropy
    """
    return -np.log(1 / n_class)


def label_entropy_statistics(file_path):
    """
    We use the statistics results for prediction.
    :param file_path: json file containing image names and labels
    :return: the entropy
    """

    # get labels from json file
    with open(file_path, 'r') as f:
        imgs = json.load(f)
        labels = list(imgs.values())
    # convert labels to int numbers
    labels = [ord(label.upper()) - ord('A') for label in labels]

    # calculate entropy
    hist = np.histogram(np.array(labels), bins=26)[0]
    probs = hist / len(labels)
    probs = probs[probs > 0]
    ent = np.sum(-probs * np.log(probs))

    return ent


def label_entropy_model(im_dir, file_path, norm_size, batch_size, model_path, device='cpu'):
    """
    We use the trained model for prediction.
    :param im_dir: path to directory with images
    :param file_path: json file containing image names and labels
    :param norm_size: image normalization size, (width, height)
    :param batch_size: batch size
    :param model_path: path of the saved model
    :param device: 'cpu' or 'cuda'
    :return: the entropy
    """

    # initialize dataloader and model
    dataloader = dataLoader(im_dir, file_path, norm_size, batch_size)
    checkpoint = torch.load(model_path)
    model = SimpleCNN()
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)

    # extract features
    outs = []
    model.eval()
    with torch.no_grad():
        for ims, _ in dataloader:
            ims = ims.to(device)
            out = model(ims)
            outs.append(out)

    # calculate entropy
    probs = torch.cat(outs, 0).softmax(1)  # [n_ims, 26], probabilities of predicted characters
    probs = probs.cpu().numpy()
    ent = 0.
    for prob in probs:
        prob = prob[prob > 0]
        ent -= np.sum(prob * np.log(prob))
    ent /= len(probs)

    return ent


def feature_entropy_channel(x, n_bins, ignore_lowest):
    """ The entropy of a single channel feature map
    :param x: feature map with shape [h, w] in pytorch tensor form
    :param n_bins: the bins to be divided
    :param ignore_lowest: whether to ignore the lowest value when calculating the entropy
    :return: the entropy
    """

    x = x.view(-1)
    if ignore_lowest:
        assert x.max() > x.min(), 'the feature map is identical, cannot ignore the lowest value'
        x = x[x > x.min()]

    hist = np.histogram(x.cpu().numpy(), bins=n_bins)[0]
    probs = hist / len(x)
    probs = probs[probs > 0]
    ent = np.sum(-probs * np.log(probs))

    return ent


def feature_entropy(x, n_bins, ignore_lowest, reduction='mean'):
    """ The entropy of feature maps
    :param x: feature map with shape [c, h, w] in pytorch tensor form
    :param n_bins: the bins to be divided
    :param ignore_lowest: whether to ignore the lowest value when calculating the entropy
    :param reduction: 'mean' or 'sum', the way to reduce results of c channels
    :return: the entropy
    """

    ent = 0.
    for f in x:
        if ignore_lowest and f.max() == f.min():
            continue
        ent += feature_entropy_channel(f, n_bins, ignore_lowest)

    assert reduction in ['mean', 'sum'], 'reduction should be mean or sum'
    if reduction == 'mean':
        ent /= x.size(0)
    return ent


def feature_entropy_dataset(n_bins, ignore_lowest, reduction, im_dir, file_path,
                            norm_size, batch_size, model_path, device='cpu'):
    """
    Calculate entropy of features extracted by our model.
    :param n_bins: the bins to be divided
    :param ignore_lowest: whether to ignore the lowest value when calculating the entropy
    :param reduction: 'mean' or 'sum', the way to reduce results of c channels
    :param im_dir: path to directory with images
    :param file_path: json file containing image names and labels
    :param norm_size: image normalization size, (width, height)
    :param batch_size: batch size
    :param model_path: path of the saved model
    :param device: 'cpu' or 'cuda'
    :return: the entropy of features
    """

    # initialize dataloader and model
    dataloader = dataLoader(im_dir, file_path, norm_size, batch_size)
    checkpoint = torch.load(model_path)
    model = SimpleCNN()
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)

    # extract features and calculate entropy
    ent1, ent2, ent3 = 0., 0., 0.
    n_ims = 0
    model.eval()
    with torch.no_grad():
        for ims, _ in dataloader:
            ims = ims.to(device)
            _, feats1, feats2, feats3 = model(ims, return_features=True)
            n_ims += ims.size(0)
            for f1, f2, f3 in zip(feats1, feats2, feats3):
                ent1 += feature_entropy(f1, n_bins, ignore_lowest, reduction)
                ent2 += feature_entropy(f2, n_bins, ignore_lowest, reduction)
                ent3 += feature_entropy(f3, n_bins, ignore_lowest, reduction)

    return ent1 / n_ims, ent2 / n_ims, ent3 / n_ims


def entropy_single_input(im_path, norm_size, model_path,
                         n_bins, ignore_lowest, reduction,
                         device='cpu'):
    """
    Calculate entropy of a single image and its prediction
    :param im_path: path to an image file
    :param norm_size: image normalization size, (width, height)
    :param model_path: path of the saved model
    :param n_bins: the bins to be divided
    :param ignore_lowest: whether to ignore the lowest value when calculating the entropy
    :param reduction: 'mean' or 'sum', the way to reduce results of c channels
    :param device: 'cpu' or 'cuda'
    :return: image entropy and predicted probability entropy
    """

    # read image and calculate image entropy
    im = cv2.imread(im_path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im = cv2.resize(im, norm_size)
    ent_im = im_entropy(im)
    # preprocess
    im = (torch.from_numpy(im).float() - 127.5) / 127.5
    im = im.view(1, 1, norm_size[1], norm_size[0])
    im = im.to(device)

    # initialize the model
    checkpoint = torch.load(model_path)
    model = SimpleCNN()
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)

    # calculate prediction entropy
    model.eval()
    with torch.no_grad():
        out, f1, f2, f3 = model(im, return_features=True)
    ent_f1 = feature_entropy(f1[0], n_bins, ignore_lowest, reduction)
    ent_f2 = feature_entropy(f2[0], n_bins, ignore_lowest, reduction)
    ent_f3 = feature_entropy(f3[0], n_bins, ignore_lowest, reduction)
    pred = out[0].argmax().item()
    pred = chr(pred + ord('A'))
    prob = out[0].softmax(0).cpu().numpy()
    confidence = prob.max()
    prob = prob[prob > 0]
    ent_pred = np.sum(-prob * np.log(prob))

    return pred, confidence, ent_im, ent_f1, ent_f2, ent_f3, ent_pred, f1[0], f2[0], f3[0]


def visualize_features(x, rows, cols):
    """
    Visualize feature maps.
    :param x: feature maps in pytorch tensor, [c, h, w]
    :param rows: rows of subplots
    :param cols: columns of subplots
    """

    c = x.size(0)
    f, axs = plt.subplots(rows, cols)
    assert rows * cols >= c, 'not enough space to draw all feature maps'
    for i, f in enumerate(x):
        row = i // cols
        col = i % cols
        axs[row][col].imshow(f.cpu())
        axs[row][col].set_xticks([])
        axs[row][col].set_yticks([])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='dataset',
                        help='if mode is dataset, then calculate average entropy of the whole dataset; \
                              if mode is single, then calculate the entropy of a single image')
    parser.add_argument('--im_dir', type=str, default='data/images',
                        help='path to directory with images')
    parser.add_argument('--train_file_path', type=str, default='data/train.json',
                        help='file list of training image paths and labels')
    parser.add_argument('--test_file_path', type=str, default='data/test.json',
                        help='file list of test image paths and labels')
    parser.add_argument('--batchsize', type=int, default=8, help='batch size')
    parser.add_argument('--n_bins', type=int, default=10, help='the bins to be divided')
    parser.add_argument('--ignore_lowest', action='store_true',
                        help='whether to ignore the lowest value when calculating the entropy for feature maps')
    parser.add_argument('--reduction', type=str, default='mean',
                        help='mean or sum, the way to reduce results of c channels for feature maps')
    parser.add_argument('--device', type=str, default='cpu', help='cpu or cuda')
    parser.add_argument('--norm_size', type=str, default='32,32',
                        help='image normalization size, width,height, split by comma')
    parser.add_argument('--model_path', type=str, default='saved_models/recognition.pth',
                        help='path of a saved model')
    parser.add_argument('--im_path', type=str, default='', help='path of an image file')
    parser.add_argument('--visualize', action='store_true', help='whether to visualize feature maps')
    parser.add_argument('--subplots1', type=str, default='2,4',
                        help='rows and columns of visualization features 1, split by comma')
    parser.add_argument('--subplots2', type=str, default='4,4',
                        help='rows and columns of visualization features 2, split by comma')
    parser.add_argument('--subplots3', type=str, default='8,4',
                        help='rows and columns of visualization features 3, split by comma')

    opt = parser.parse_args()

    w, h = map(int, opt.norm_size.split(','))
    if opt.mode == 'dataset':
        print('\ncalculating entropy of dataset...')

        # image entropy
        ent_im = im_entropy_dataset(opt.im_dir, opt.test_file_path, (w, h))
        # feature entropy
        ent_f1, ent_f2, ent_f3 = feature_entropy_dataset(opt.n_bins, opt.ignore_lowest,
                                                         opt.reduction, opt.im_dir,
                                                         opt.test_file_path, (w, h),
                                                         opt.batchsize, opt.model_path,
                                                         opt.device)
        # label entropy
        ent_rand = label_entropy_random()
        ent_statistics = label_entropy_statistics(opt.train_file_path)
        ent_model = label_entropy_model(opt.im_dir, opt.test_file_path, (w, h),
                                        opt.batchsize, opt.model_path, opt.device)
        
        print('Entropy of input test images = {:.2f}'.format(ent_im))
        print('Entropy of features = {:.2f}, {:.2f}, {:.2f}'.format(ent_f1, ent_f2, ent_f3))
        print('Entropy of random guess = {:.2f}'.format(ent_rand))
        print('Entropy of symbols in text labels = {:.2f}'.format(ent_statistics))
        print('Entropy of using trained model = {:.2f}'.format(ent_model))

    elif opt.mode == 'single':
        print('\n{}:'.format(os.path.basename(opt.im_path)))
        pred, confidence, ent_im, ent_f1, ent_f2, ent_f3, ent_pred, f1, f2, f3 = \
            entropy_single_input(opt.im_path, (w, h), opt.model_path, opt.n_bins,
                                 opt.ignore_lowest, opt.reduction, opt.device)
        print('Recognition result: {} (confidence = {:.2f})'.format(pred, confidence))
        print('Entropy of input image = {:.2f}'.format(ent_im))
        print('Entropy of features = {:.2f}, {:.2f}, {:.2f}'.format(ent_f1, ent_f2, ent_f3))
        print('Entropy of prediction = {:.2f}'.format(ent_pred))

        if opt.visualize:
            r1, c1 = map(int, opt.subplots1.split(','))
            r2, c2 = map(int, opt.subplots2.split(','))
            r3, c3 = map(int, opt.subplots3.split(','))
            visualize_features(f1, r1, c1)
            visualize_features(f2, r2, c2)
            visualize_features(f3, r3, c3)
            plt.show()

    else:
        raise NotImplementedError('mode should be dataset or single')
