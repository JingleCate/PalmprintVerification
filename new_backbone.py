import os
import cv2
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
import torchvision.models as models
from torch.nn.parallel import DataParallel
from torch.optim.lr_scheduler import StepLR

from tqdm import tqdm
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score


from tqdm import tqdm
from dataset import CustomDataset, proprecess
from backbone import end_process
from utils import plot_evaluate

from facenet_pytorch import MTCNN, InceptionResnetV1
  

class Backbone(nn.Module):
    def __init__(self, batch_size):
        super(Backbone, self).__init__()
        self.batch_size = batch_size
        # pretrained='casia-webface'
        self.resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        # # 替换最后一层全连接层
        # num_features = self.resnet.fc.in_features
        # self.resnet.fc = nn.Linear(num_features, 2)
        # # 冻结所有模型参数
        # for param in self.resnet.parameters():
        #     param.requires_grad = False
        self.threshold = 0.5
    
    def forward(self, x):
        # Bacth size must be even
        num_rows = x.size(1)
        indices1 = torch.arange(num_rows//2)
        indices2 = torch.arange(num_rows//2, num_rows)

        # print("Pro: ", indices2)
        x1 = torch.index_select(x, dim=1, index=indices1)
        x2 = torch.index_select(x, dim=1, index=indices2)
        
        f1 = self.resnet(x1)
        f2 = self.resnet(x2)
        
        # # Use L2 to normalize the feature
        # f1 = F.normalize(f1, p=2, dim=1)
        # f2 = F.normalize(f2, p=2, dim=1)
        
        # f size (batch_size, 512), Compute cosin similarity and similarity probility
        cos_sim = F.cosine_similarity(f1, f2, dim=1).reshape((-1, 1))
        sim_prob = (cos_sim + 1) / 2
        return sim_prob


def evaluate(test_path, batch_size, checkpoints_path):
    """Evaluate the model
    """
    print("#############################  Evaluate  ##################################")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    if checkpoints_path and os.path.isfile(checkpoints_path): 
        checkpoints = torch.load(checkpoints_path)
    else:
        print("Arguments loaded error, please select feasible checkpoint.")
        return
    
    classifier = Backbone(batch_size)
    classifier.load_state_dict(checkpoints['model_state_dict'])
    
    dataset = CustomDataset(path=test_path)
    test_loader = data.DataLoader(dataset, batch_size, shuffle=False)
    
    
    predicted_labels = []
    true_labels = []
    with torch.no_grad():
        i = 0           # record the iteration
        for path1, path2, target in tqdm(test_loader):
            i += 1
            processed_batch = proprecess(path1, path2, device, batch_size)
            target = torch.tensor([int(i) for i in target]).reshape(
                batch_size, 1).float()

            pred = classifier.forward(processed_batch)
            outcome = []
            for p in pred:
                outcome.append(1 if p > 0.5 else 0) 
            target = target.view(-1).numpy()
            pred = pred.view(-1).numpy()
                 
            # print("\n\n\tTarget:   ", target)
            # print("\tPredict:  ", pred)
            # print("\tOutcome:  ", outcome)
            
            true_labels.extend(target)
            predicted_labels.extend(outcome)
    print("\nConfusion matrix: ")
    classes = list(range(2))
    cmatrix = confusion_matrix(true_labels, predicted_labels, labels=classes)
    print(cmatrix)
    
    acc = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, average='macro')
    recall = recall_score(true_labels, predicted_labels, average='macro')
    f1 = f1_score(true_labels, predicted_labels, average='macro')
    print(f'Test Accuracy: {acc:.4f}')
    print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')
    print("#############################  Evaluate  ##################################")
    
    return [acc, precision, recall, f1]


def train(epoch, batch_size, save_path, resume: bool = False, resume_path: str = None):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)
    
    # Define classifier
    classifier = Backbone(batch_size=batch_size).to(device)
    
    # Load lr
    if resume is True:
        checkpoint = torch.load(resume_path)
        learning_rate = checkpoint['learning_rate']
        it_prime = checkpoint["iter"]
        # learning_rate = 5e-5
        # it_prime = 3800
    else:
        learning_rate = 1e-4
        it_prime = 0
        
    # Define optimizer
    optimizer = torch.optim.SGD(classifier.parameters(), lr=learning_rate)
    
    # Load train dataset and use dataloader
    dataset = CustomDataset(path="./datapaths/train_resampled.txt")
    train_loader = data.DataLoader(dataset, batch_size, shuffle=False)
    
    loss_func = nn.BCELoss()                        # Loss function
    model_parallel = DataParallel(classifier)       # parallel compute to accelerate
    scheduler = StepLR(optimizer, step_size=50, gamma=0.90)     # update lr
    
    # resume the params and optimizer
    if resume is True:
        classifier.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        ep_prime = checkpoint['epoch'] # Remember to modify 1
    
    i = it_prime                # record the iteration
    loss_record = []            # record the loss iteration
    accuracy, precision, recall, f1 = [], [], [], []
    for ep in range(ep_prime, epoch + 1):
        print('\nEpoch {}/{}'.format(ep, epoch))
        print('-' * 10)
        for path1, path2, target in tqdm(train_loader):
            i += 1
            # Proprecess the input
            processed_batch = proprecess(path1, path2, device, batch_size)
            # print("Processed: ", processed_batch.shape)
            target = torch.tensor([int(i) for i in target]).reshape(
                batch_size, 1).float()

            # forward and caculate the loss
            # pred = classifier.forward(processed_batch)
            pred = model_parallel(processed_batch)
            ls = loss_func(pred, target)
            outcome = []                    # Record the outcome
            for p in pred:
                outcome.append(1 if p > 0.5 else 0)   

            pred = pred.view(-1).detach().numpy()
            print("\n\tTarget:   ", target.view(-1).numpy())
            print("\tPredict:  ", pred)
            
            print("\tOutcome:  ", outcome)
            print("\tIteration: ep%d %d     Mean loss: %0.3f" % (ep, i, ls))
            print("-------------------------------------------------------------------------------------")   

            loss_record.append(ls.item())       # record loss of each bacth

            # Compute and updata params
            optimizer.zero_grad()
            ls.backward()
            optimizer.step()
            scheduler.step()

            # Save and record
            save_it = 500
            if i % save_it == 0:
                beta_path = os.path.join(save_path, "epoch%d-it%d.pth" % (ep, i))
                # Get the learning rate and save
                lr = optimizer.param_groups[0]['lr']
                end_process(beta_path, loss_record, i, ep, classifier, optimizer, lr)
                
                # each 100 compute the following indexes on validation dataset
                acc, prec, rec, f1_prime= evaluate("./datapaths/valid_resampled.txt", batch_size, beta_path)
                accuracy.append(acc)
                precision.append(prec) 
                recall.append(rec) 
                f1.append(f1_prime)
                # Acquire learning rate
                print('Adam optimizer learning rate: {:.6f}'.format(lr))
                
                

            # Save the potential minimizer
            if ls < 0.10:
                pt = "./checkpoints/success/succ{}.pth".format(i)
                end_process(pt, loss_record, i, ep, classifier, optimizer)
        # end_process("./checkpoints/lastiter.pth", loss_record, 0, ep, classifier, optimizer)
        plot_evaluate(accuracy, precision, recall, f1, ep + 1)
        break
     
        
if __name__ == "__main__":
    train(epoch=10, batch_size=8, save_path="./checkpoints",
          resume=True, resume_path="./checkpoints/epoch2-it5900.pth")
        
