import os
import cv2
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
import torch.nn.init as init
from torchvision import models, transforms
from torch.nn.parallel import DataParallel

from dataset import CustomDataset, proprecess
from tqdm import tqdm
from utils import plot_loss


class FeatureExtractor(nn.Module):
    """Feature extractor, which process the input img (3, 448, 448) into a 512 dim vector.
    It bases on pretrained VGG16 using weight: models.VGG16_Weights.DEFAULT

    Args:
        
    """
    def __init__(self, model):
        super(FeatureExtractor, self).__init__()
        # Extract VGG-16 Feature Layers
        self.features = list(model.features)
        self.features = nn.Sequential(*self.features)
        # Extract VGG-16 Average Pooling Layer
        self.pooling = model.avgpool
        # Convert the image into one-dimensional vector
        self.flatten = nn.Flatten()
        # Extract the first part of fully-connected layer from VGG16
        # self.fc = model.classifier[0]
        self.fc = nn.Linear(25088, 512, bias=True)

    def forward(self, x):
        # It will take the input 'x' until it returns the feature vector called 'out'
        out = self.features(x)
        out = self.pooling(out)
        out = self.flatten(out)
        out = self.fc(out)
        return out


class BaseClassifier(nn.Module):
    """Basic classifier.

    Args:
        input_size (_type_): 1024 dim
        
        hinden_size1 (int, optional): first-order hinden layer. Defaults to 200.
        
        hinden_size2 (int, optional): second-order hinden layer. Defaults to 50.
    """
    def __init__(self, input_size, hinden_size1: int = 200, hinden_size2: int = 50):
        super(BaseClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hinden_size1)
        self.fc2 = nn.Linear(hinden_size1, hinden_size2)
        self.fc3 = nn.Linear(hinden_size2, 1)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        init.kaiming_normal_(
            self.fc1.weight,  mode="fan_in", nonlinearity="relu")
        init.kaiming_normal_(
            self.fc2.weight,  mode="fan_in", nonlinearity="relu")
        init.zeros_(self.fc3.weight)

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=1)
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.sigmoid(self.fc3(x))

        return x


class Classifier(nn.Module):
    """Foreost, use extractor to acquire features, then use base_classifier output same or different

    Args:
        input_size (_type_): (batch_size, 6, 448, 448)
        
        batch_size (int, optional): . Defaults to 2.
        
        hinden_size1 (int, optional): used on base Classifier. Defaults to 200.
        
        hinden_size2 (int, optional): used on base Classifier. Defaults to 50.
        
        pTrained (bool, optional): used on VGG16. Defaults to False.
    """
    def __init__(self, input_size, batch_size: int = 2, hinden_size1: int = 200, hinden_size2: int = 50, pTrained=False):
        super(Classifier, self).__init__()
        self.batch_size = batch_size
        self.base_classifier = BaseClassifier(
            input_size, hinden_size1, hinden_size2)

        if pTrained == True:
            base_model = models.vgg16(
                weights=models.VGG16_Weights.DEFAULT)
        else:
            base_model = models.vgg16()
        self.extractor = FeatureExtractor(base_model)

    def forward(self, x):
        # Bacth size must be even
        num_rows = x.size(1)
        indices1 = torch.arange(num_rows//2)
        indices2 = torch.arange(num_rows//2, num_rows)

        # print("Pro: ", indices2)
        x1 = torch.index_select(x, dim=1, index=indices1)
        x2 = torch.index_select(x, dim=1, index=indices2)
        # print("Processed1: ", x1.shape)
        # print("Processed2: ", x2.shape)

        f1 = self.extractor(x1)
        f2 = self.extractor(x2)

        out = self.base_classifier(f1, f2)

        return out


def end_process(path, loss_record, it, ep, model, optm, lr):
    plot_loss(loss_record, it)
    torch.save({
        "learning_rate": lr,
        'epoch': ep,
        "iter": it,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optm.state_dict(),
    }, path)


def train(epoch, batch_size, save_path, resume: bool = False, resume_path: str = None):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)

    # Use VGG16 as base model

    classifier = Classifier(1024, pTrained=True)
    classifier = classifier.to(device)

    # Define the dataset and use the Dataloader provided by torch
    dataset = CustomDataset(path="./datapaths/train_resampled.txt")
    train_loader = data.DataLoader(dataset, batch_size, shuffle=True)
    print("Batch size: ", batch_size)

    # optimizer
    optimizer = torch.optim.Adam(
        classifier.parameters(), lr=1e-2)
    # loss = NetLoss()
    loss = torch.nn.BCELoss()
    loss_record = [0]

    if resume is True:
        checkpoint = torch.load(resume_path)
        classifier.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        ep = checkpoint['epoch']

    model_parallel = DataParallel(classifier)

    print("*************************************************************************************")
    for ep in range(epoch):
        i = 0           # record the iteration
        for path1, path2, target in tqdm(train_loader):
            i += 1
            processed_batch = proprecess(path1, path2, device, batch_size)
            # print("Processed: ", processed_batch.shape)
            target = torch.tensor([int(i) for i in target]).reshape(
                batch_size, 1).float()

            # pred = classifier.forward(processed_batch)
            pred = model_parallel(processed_batch)

            ls = loss(pred, target)
            outcome = []
            for p in pred:
                outcome.append(1 if p > 0.5 else 0)      
            print("\n\tTarget:   ", target.view(-1).numpy())
            print("\tPredict:  ", pred.detach().view(-1).numpy())
            print("\tOutcome:  ", outcome)
            print("\n\tIteration: %d  Mean loss: %0.3f" % (i, ls))
            print(
                "-------------------------------------------------------------------------------------")

            # print("\nloss: ", ls, end='  ')
            # print("size: %s value: %.4f" % (ls.shape, ls.item()))
            # if ls.mean() - loss_record[-1] < 1e-6:
            #     break

            loss_record.append(ls.item())
            # break

            # Compute and updata params
            optimizer.zero_grad()
            ls.backward()
            optimizer.step()

            # Save and record
            save_it = 100
            if i % save_it == 0:
                beta_path = os.path.join(
                    save_path, "epoch%d-it%d.pth" % (ep, i))
                end_process(beta_path, loss_record, i, ep, classifier, optimizer)
                # Acquire learning rate
                lr = optimizer.param_groups[0]['lr']
                print('Adam optimizer learning rate: {:.6f}'.format(lr))

            if ls < 0.10:
                pt = "./checkpoints/success/succ{}.pth".format(i)
                end_process(pt, loss_record, 1+i, ep, classifier, optimizer)
        end_process("./checkpoints/lastiter.pth", loss_record, 0, ep, classifier, optimizer)


if __name__ == "__main__":
    train(epoch=1, batch_size=8, save_path="./checkpoints",
          resume=False, resume_path="./checkpoints/success/succ98-7.pth")

    # pred("ROI/001_1_h_l_01_ROI.jpeg", "ROI/001_1_h_r_10_ROI.jpeg")
