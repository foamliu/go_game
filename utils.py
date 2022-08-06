import argparse
import logging
import math

import cv2 as cv
import numpy as np
import torch


def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.
    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def save_checkpoint(epoch, epochs_since_improvement, model, optimizer, acc, is_best):
    state = {'epoch': epoch,
             'epochs_since_improvement': epochs_since_improvement,
             'loss': acc,
             'state_dict': model.module.state_dict(),
             'optimizer': optimizer}
    # filename = 'checkpoint_' + str(epoch) + '_' + str(loss) + '.tar'
    filename = 'checkpoint.pth'
    torch.save(state, filename)
    # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    if is_best:
        torch.save(state, 'BEST_checkpoint.pth')


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, shrink_factor):
    """
    Shrinks learning rate by a specified factor.
    :param optimizer: optimizer whose learning rate must be shrunk.
    :param shrink_factor: factor in interval (0, 1) to multiply learning rate with.
    """

    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))


def set_learning_rate(optimizer, learning_rate):
    """
    Set learning rate by a specified value.
    :param optimizer: optimizer whose learning rate must be set.
    :param learning_rate: specified learning rate.
    """

    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))


def get_learning_rate(optimizer):
    return optimizer.param_groups[0]['lr']


def get_logger():
    logger = logging.getLogger()
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s %(levelname)s \t%(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    return logger


def draw_landmarks(img, landmarks):
    img = np.array(img)
    for i in range(68):
        x = int(landmarks[i][0])
        y = int(landmarks[i][1])
        cv.circle(img, (x, y), 1, (0, 255, 0), 2)

    return img


def accuracy(scores, targets, k=1):
    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() * (100.0 / batch_size)


def parse_args():
    parser = argparse.ArgumentParser(description='train alphago network')
    # general
    parser.add_argument('--model', type=str, default='alphago zero', help='resnet-50 or mobilenet-v2')
    parser.add_argument('--end-epoch', type=int, default=1000, help='training epoch size.')
    parser.add_argument('--lr', type=float, default=0.1, help='start learning rate')
    parser.add_argument('--optimizer', default='adam', help='optimizer')
    parser.add_argument('--weight-decay', type=float, default=0, help='weight decay')
    parser.add_argument('--mom', type=float, default=0.9, help='momentum')
    parser.add_argument('--batch-size', type=int, default=128, help='batch size in each context')
    parser.add_argument('--checkpoint', type=str, default=None, help='checkpoint')
    parser.add_argument('--pretrain', type=str, default=None, help='pretrain')
    parser.add_argument('--robust', type=bool, default=False, help='robust')
    args = parser.parse_args()
    return args


def rotate_image(image, landmarks):
    # landmarks = np.array([float(l) for l in facial_landmarks.split(';')]).reshape((-1, 2))
    left_eye_x = left_eye_y = right_eye_x = right_eye_y = 0
    for (x, y) in landmarks[36:41]:
        left_eye_x += x
        left_eye_y += y
    for (x, y) in landmarks[42:47]:
        right_eye_x += x
        right_eye_y += y
    left_eye_x /= 6
    left_eye_y /= 6
    right_eye_x /= 6
    right_eye_y /= 6
    theta = math.degrees(math.atan((right_eye_y - left_eye_y) / (right_eye_x - left_eye_x)))
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv.getRotationMatrix2D(image_center, theta, 1.0)
    result = cv.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv.INTER_LINEAR)
    # print(left_eye_x, left_eye_y, right_eye_x, right_eye_y,theta,img.shape)
    return result


def ensure_folder(folder):
    import os
    if not os.path.isdir(folder):
        os.mkdir(folder)


# adapted from https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
def set_parameter_requires_grad(model, requires_grad):
    for param in model.parameters():
        param.requires_grad = requires_grad
