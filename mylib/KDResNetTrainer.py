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
import torch.nn.functional as F
import wandb

import os
from tqdm import tqdm

def distillation(y, labels, teacher_scores, T, alpha):
    # distillation loss + classification loss
    # y: student
    # labels: hard label
    # teacher_scores: soft label
  return nn.KLDivLoss()(F.log_softmax(y/T, dim=1), F.softmax(teacher_scores/T, dim=1)) * (T*T * alpha) + F.cross_entropy(y,labels) * (1.-alpha)

class ResNetTrainer():
  def __init__(self, dataset, student_model, teacher_model, arch_tag, 
               T, alpha, device = 'cuda', lr = 0.1, batch_size = 128):
    self.dataset = dataset
    self.batch_size = batch_size
    self.student_model = student_model
    self.teacher_model = teacher_model
    self.lr = lr
    self.device = device
    self.best_acc = 0
    self.T = T
    self.alpha = alpha

    if dataset == 'CIFAR10':
      self.train_loader, self.test_loader = self._load_CIFAR10(self.batch_size)
    elif dataset == 'CIFAR100':
      raise ValueError('CIFAR100 not supported yet')
    else:
      raise ValueError('Dataset not supported')
    
    self.criterion = nn.CrossEntropyLoss()
    self._init_opt()

  def _init_opt(self):
    self.student_model.to(self.device)
    self.teacher_model.to(self.device)
    self.optimizer = optim.SGD(self.student_model.parameters(), lr=self.lr,
                               momentum=0.9, nesterov=True, weight_decay=5e-4)
    # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200)
    # self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[40, 80, 120, 160], gamma=0.1)
    self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=list(range(80,200,40)), gamma=0.1)



    if self.device == 'cuda':
        self.student_model = torch.nn.DataParallel(self.student_model)
        self.teacher_model = torch.nn.DataParallel(self.teacher_model)

        # cudnn.benchmark = True

  def _load_CIFAR10(self, batch_size):
    transform_train = transforms.Compose([
      transforms.RandomCrop(32, padding=4),  #resises the image so it can be perfect for our student_model.
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

    # knowledge distillation loss
  
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
    self.student_model.train()
    self.optimizer.zero_grad()
    pbar_train_loader = tqdm(self.train_loader)
    for batch_idx, (inputs, targets) in enumerate(pbar_train_loader):
        pbar_train_loader.set_description('Epoch ' + str(epoch))
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        outputs = self.student_model(inputs)
        loss = self.criterion(outputs, targets)
        loss.backward()

        pbar_train_loader.set_postfix()
    return 

  def _train(self, epoch):
    print('\nEpoch: %d' % epoch)
    self.student_model.train()
    train_loss = 0
    correct = 0
    total = 0
    pbar_train_loader = tqdm(self.train_loader)
    for batch_idx, (inputs, targets) in enumerate(pbar_train_loader):
        pbar_train_loader.set_description('Epoch ' + str(epoch))

        inputs, targets = inputs.to(self.device), targets.to(self.device)
        self.optimizer.zero_grad()
        outputs = self.student_model(inputs)
        # loss = self.criterion(outputs, targets)
        loss = distillation(outputs, targets, self.teacher_model(inputs), T=self.T, alpha=self.alpha)
        crossEnt = self.criterion(outputs, targets)
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
    wandb.log({"train_loss": train_loss / total, 
               "train_acc": (correct / total),
               "cross_entropy": crossEnt,
               "lr": self.scheduler.get_lr()[0]} )
    # print('Accuracy of the student_model on the train images: {} %'.format(100 * correct / total))

  def _test(self, epoch):
    self.student_model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(self.test_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            outputs = self.student_model(inputs)
            loss = self.criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    tqdm.write('test_acc: %.3f' % (correct/total))

            # progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #              % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # print('Accuracy of the student_model on the test images: {} %'.format(100 * correct / total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > self.best_acc:
        print('Saving..')
        state = {
            'net': self.student_model.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        self.best_acc = acc
        artifact = wandb.Artifact('student_model', type='student_model')
        artifact.add_file('./checkpoint/ckpt.pth')
        wandb.log_artifact(artifact)
    wandb.log({"test_loss": test_loss / total, "test_acc": (correct / total)})
    return acc