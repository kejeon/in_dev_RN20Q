# -*- coding: utf-8 -*-
"""ResNetTrainer.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Nw6TNDrlzw6Dw_zPEXU1TvTs5uKSp3Ui
"""

import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
import wandb
from mylib.KL_div import avg_kl_div_loss

import os
from tqdm import tqdm

class ResNetTrainer():
  def __init__(self, dataset, model, arch_tag, eta,
               criterion = nn.CrossEntropyLoss(), device = 'cuda', lr = 0.1, batch_size = 128):
    self.dataset = dataset
    self.batch_size = batch_size
    self.model = model
    self.lr = lr
    self.device = device
    self.best_acc = 0
    self.eta = eta

    if dataset == 'CIFAR10':
      self.train_loader, self.test_loader = self._load_CIFAR10(self.batch_size)
    elif dataset == 'CIFAR100':
      raise ValueError('CIFAR100 not supported yet')
    else:
      raise ValueError('Dataset not supported')
    
    self.criterion = nn.CrossEntropyLoss()
    self._init_opt()

  def _init_opt(self):
    self.model.to(self.device)
    self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr,
                               momentum=0.9, nesterov=True, weight_decay=5e-4)
    self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200)
    # self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=list(range(100,200,20)), gamma=0.1)

    if self.device == 'cuda':
        self.model = torch.nn.DataParallel(self.model)
        # cudnn.benchmark = True

  def _load_CIFAR10(self, batch_size):
    transform_train = transforms.Compose([
      transforms.RandomCrop(32, padding=4),  #resises the image so it can be perfect for our model.
      transforms.RandomHorizontalFlip(), # FLips the image w.r.t horizontal axis
      transforms.ToTensor(), # comvert the image to tensor so that it can work with torch
      transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)) #Normalize all the images
    ])
    
    transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', 
                                            train=True, 
                                            download=True, 
                                            transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, 
                                              batch_size=batch_size,
                                              shuffle=True, 
                                              num_workers=2)
    testset = torchvision.datasets.CIFAR10(root='./data', 
                                          train=False,
                                          download=True, 
                                          transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)
    # classes = ('plane', 'car', 'bird', 'cat', 'deer',
    #           'dog', 'frog', 'horse', 'ship', 'truck')
    return train_loader, test_loader

  def train_script(self, num_it):
    best_acc = 0
    # start_epoch = epoch + 1
    start_epoch = 0

    for epoch in range(start_epoch, start_epoch+num_it):
        self._train(epoch)
        self._test(epoch)
        self.scheduler.step()
    return

  def compute_grad(self, epoch=0):
    self.model.train()
    self.optimizer.zero_grad()
    pbar_train_loader = tqdm(self.train_loader)
    for batch_idx, (inputs, targets) in enumerate(pbar_train_loader):
        pbar_train_loader.set_description('Epoch ' + str(epoch))
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)
        loss.backward()

        pbar_train_loader.set_postfix()
    return 

  def _train(self, epoch):
    print('\nEpoch: %d' % epoch)
    self.model.train()
    train_loss = 0
    correct = 0
    total = 0
    pbar_train_loader = tqdm(self.train_loader)
    for batch_idx, (inputs, targets) in enumerate(pbar_train_loader):
        pbar_train_loader.set_description('Epoch ' + str(epoch))

        inputs, targets = inputs.to(self.device), targets.to(self.device)
        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        kl_div_loss = avg_kl_div_loss(self.model)
        wc_loss = self.eta*kl_div_loss
        xent_loss = self.criterion(outputs, targets)
        loss = xent_loss + wc_loss
        loss.backward()
        self.optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        pbar_train_loader.set_postfix(xentropy='%.3f' % (train_loss / (batch_idx + 1)),
                                      train_acc='%.3f' % (correct / total))

        # progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #              % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    wandb.log({"xent_loss": xent_loss, 
               "kl_div_loss": kl_div_loss,
               "wc_loss": wc_loss, 
               "train_loss": train_loss / total, 
               "train_acc": (correct / total),
               "lr": self.scheduler.get_lr()[0]} )
    # print('Accuracy of the model on the train images: {} %'.format(100 * correct / total))

  def _test(self, epoch):
    self.model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(self.test_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    tqdm.write('test_acc: %.3f' % (correct/total))

            # progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #              % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # print('Accuracy of the model on the test images: {} %'.format(100 * correct / total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > self.best_acc:
        print('Saving..')
        state = {
            'net': self.model.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        self.best_acc = acc
        artifact = wandb.Artifact('model', type='model')
        artifact.add_file('./checkpoint/ckpt.pth')
        wandb.log_artifact(artifact)
    wandb.log({"test_loss": test_loss / total, "test_acc": (correct / total)})
    return acc