import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import math
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.models as models
import torchvision.datasets as datasets
# import SMD_opt_new
import SMD_opt

class CIFAR10RandomLabels(datasets.CIFAR10):
    def __init__(self, corrupt_prob=0.0, num_classes=10,**kwargs):
        super(CIFAR10RandomLabels, self).__init__(**kwargs)
        self.n_classes = num_classes
        if corrupt_prob > 0:
            self.corrupt_labels(corrupt_prob)

    def corrupt_labels(self, corrupt_prob):
        labels = np.array(self.targets)
        mask = np.random.rand(len(labels)) <= corrupt_prob
        rnd_labels = np.random.choice(self.n_classes, mask.sum())
        labels[mask] = rnd_labels
        # we need to explicitly cast the labels from npy.int64 to
        # builtin int type, otherwise pytorch will fail...
        labels = [int(x) for x in labels]
        self.targets = labels


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18():
    return ResNet(BasicBlock, [2,2,2,2])

def ResNet34():
    return ResNet(BasicBlock, [3,4,6,3])

def ResNet50():
    return ResNet(Bottleneck, [3,4,6,3])

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

num_epochs = 3500
num_classes = 10
batch_size = 128

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
])

trainset = torch.load('./cifar_40_percent_corruption.pth') #already transformed


class MyDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.cifar10 = torch.load('./cifar_40_percent_corruption.pth') #already transformed
        
    def __getitem__(self, index):
        data, target = self.cifar10[index]
        
        return data, target, index

    def __len__(self):
        return len(self.cifar10)

dataset = MyDataset()
trainloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True, num_workers=4)


testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')

model = ResNet18()


model = model.to(device)
if device == 'cuda':
    model = torch.nn.DataParallel(model) # make parallel
    cudnn.benchmark = True



def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.01)
#         print(1)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(0.0, 0.01)
        m.bias.data.fill_(0)
#         print(2)
    elif classname.find('Linear') != -1:
        m.weight.data.uniform_(-0.01, 0.01)
        m.bias.data.uniform_(-0.1, 0.1)
    
model.apply(weights_init)

free_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(free_params)

z = torch.normal(0.0, 0.0000005, size=(50000, 1)).to(device) #small z




per_sample_loss = torch.zeros(50000,1).to(device)
total_loss = torch.zeros(num_epochs,1).to(device)

q = 2
learning_rate =  10**-1
criterion = nn.CrossEntropyLoss() #losses are averaged in minibatch
per_sample_criterion = nn.CrossEntropyLoss(reduction='none') #per sample losses
optimizer = SMD_opt.SMD_qnorm(model.parameters(), lr=learning_rate, q=q)
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
# optimizer = SMD_opt_new.SMD_qnorm_new_alg(model.parameters(), lr=learning_rate, q=2)

total_step = len(trainloader)
print(total_step)

lmbda = 4

model.load_state_dict(torch.load('1initial.pth'))
# torch.save(model.state_dict(), './2norm_initial2.pth') 

w_initial = torch.nn.utils.parameters_to_vector(model.parameters())
print(torch.sum(torch.abs(w_initial)**q))
print(torch.sum(z))

history = 500
new_constraint = 0
old_constraint = 10**6
count = 0



print('Training:')
# Training
for epoch in range(num_epochs):
    total = 0
    correct = 0
#     scheduler.step()

    for i, (images, labels, idx) in enumerate(trainloader):
        
        images = images.to(device)
        labels = labels.to(device)
        images = Variable(images)
        labels = Variable(labels)
        # Run the forward pass
        outputs = model(images)
        
#         print('initial')
#         print(z[idx])
#         batch_sum_z = sum(z[idx]**2)/(2*(len(idx)))
        batch_sum_z = sum(z[idx])/((len(idx)))    #correct algo
    
#         print(batch_sum_z)
        loss = criterion(outputs, labels)
        
        total_loss[epoch] = total_loss[epoch] + loss.clone().detach().item() * len(idx)
#         print(loss.type())
        
#         newlr = learning_rate * (-batch_sum_z + loss.clone().detach().item())
#         print(batch_sum_z)
#         print(math.sqrt(2*loss.clone().detach().item()))
        newlr = learning_rate * (-batch_sum_z + math.sqrt(2*loss.clone().detach().item())) / math.sqrt(2*loss.clone().detach().item()) #correct algo
#         print(newlr)
        newlr = newlr.clone().detach().item()
        
        for g in optimizer.param_groups:
            g['lr'] = newlr           

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        
#         print(per_sample_loss[idx])
        temp1 = per_sample_criterion(outputs, labels).unsqueeze(1).clone().detach()
        
        per_sample_loss[idx] = temp1
        
        temp = z[idx]
#         z[idx] = temp - (-newlr / lmbda)* temp
#         print(temp)
#         print((-newlr / lmbda) * math.sqrt(2*loss.clone().detach().item()))
        z[idx] = temp - (-newlr / lmbda) * math.sqrt(2*loss.clone().detach().item()) #correct algo
        
         # Track the accuracy
        total = labels.size(0) + total
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels).sum().item() + correct
        
        
        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),
                          (correct / total) * 100))
#     print(per_sample_loss)
#     print('sqrt(2*per_sample_loss) : {}'.format(torch.sqrt(2*per_sample_loss)))
    print('sum(sqrt(2*per_sample_loss)) : {}'.format(torch.sum(torch.sqrt(2*per_sample_loss))))
#     print('z vector: {}'.format(z))
    print('z vector sum: {}'.format(torch.sum(z)))

    per_sample_loss = z - torch.sqrt(2*per_sample_loss)
    print('Constraints: {}'.format(per_sample_loss))
          
          
    new_constraint = torch.sum(torch.abs(per_sample_loss))
    improvement = (old_constraint - new_constraint) / old_constraint
    print('Improvement: {}'.format(improvement))
    w = torch.nn.utils.parameters_to_vector(model.parameters())
    print('Weight Norm: {}'.format(torch.sum(torch.abs(w)**q)))
    print('Constraint Sum: {}'.format(new_constraint))
    
    if (improvement >= 0.0001):
        old_constraint = new_constraint 
        count = 0
        print(epoch)
        torch.save(model.state_dict(), './rmd_40_percent_lmb_4_0_lr_0_1_long_hist.pth')
        torch.save(z, './z_40_percent_lmb_4_0_lr_0_1_long_hist.pth')
    else:
        count = count + 1 
    
#     if (count == history):
#         print(old_constraint)
#         break
        
print('Finished Training')


model.load_state_dict(torch.load('./rmd_40_percent_lmb_4_0_lr_0_1_long_hist.pth'))
# Test the model
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in testloader:
        images = images.to(device)
        labels = labels.to(device)
        
        images = Variable(images)
        labels = Variable(labels)
        
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model on the 10000 test images: {} %'.format((correct / total) * 100))

