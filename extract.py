import os
import torch
from tqdm import tqdm

from new_backbone import Backbone
from dataset import CustomDataset, proprecess
from torch.utils.data import DataLoader


# def evaluate(batch_size, checkpoints_path):
#     """Evaluate the model
#     """
#     print("#############################  Evaluate  ##################################")
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
#     if checkpoints_path and os.path.isfile(checkpoints_path): 
#         checkpoints = torch.load(checkpoints_path)
#     else:
#         print("Arguments loaded error, please select feasible checkpoint.")
#         return
    
#     classifier = Backbone(batch_size)
#     classifier.load_state_dict(checkpoints['model_state_dict'])
    
#     dataset = CustomDataset(path="./datapaths/test_resampled.txt")
#     test_loader = DataLoader(dataset, batch_size, shuffle=False)
    
    
#     predicted_labels = []
#     true_labels = []
#     with torch.no_grad():
#         i = 0           # record the iteration
#         for path1, path2, target in tqdm(test_loader):
#             i += 1
#             processed_batch = proprecess(path1, path2, device, batch_size)
#             target = torch.tensor([int(i) for i in target]).reshape(
#                 batch_size, 1).float()

#             pred = classifier.forward(processed_batch)
#             outcome = []
#             for p in pred:
#                 outcome.append(1 if p > 0.5 else 0) 
#             target = target.view(-1).numpy()
#             pred = pred.view(-1).numpy()
                 
#             print("\n\tTarget:   ", target)
#             print("\tPredict:  ", pred)
#             print("\tOutcome:  ", outcome)
            
#             true_labels.extend(target)
#             predicted_labels.extend(outcome)
    
#     classes = list(range(2))
#     cmatrix = confusion_matrix(true_labels, predicted_labels, labels=classes)
#     print(cmatrix)
    
#     acc = accuracy_score(true_labels, predicted_labels)
#     precision = precision_score(true_labels, predicted_labels, average='macro')
#     recall = recall_score(true_labels, predicted_labels, average='macro')
#     f1 = f1_score(true_labels, predicted_labels, average='macro')
#     print(f'Test Accuracy: {acc:.4f}')
#     print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')
#     print("#############################  Evaluate  ##################################")
    
            
        
def predict(path1: str, path2: str):
    """Predict the two images

    Args:
        path1 (str): _description_
        path2 (str): _description_
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoints_path = "./checkpoints/epoch2-it5900.pth"
    if checkpoints_path and os.path.isfile(checkpoints_path): 
        checkpoints = torch.load(checkpoints_path)
    else:
        print("Arguments loaded error, please select feasible checkpoint.")
        return
    
    classifier = Backbone(1024)
    classifier.load_state_dict(checkpoints['model_state_dict'])
    
    input = proprecess(path1, path2, device, batch_size=1)
    with torch.no_grad():
        pred = classifier.forward(input)
        outcome = []
        for p in pred:
            pred = pred.view(-1).numpy()
            outcome.append(1 if p > 0.5 else 0) 
        
        print("\tPredict:  ", pred)
        print("\tOutcome:  ", outcome)
            
            
if __name__ == "__main__":
    predict("", "")    
            

    



