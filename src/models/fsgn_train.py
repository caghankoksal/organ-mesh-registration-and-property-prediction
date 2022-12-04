import sys
sys.path.append('/u/home/koksal/organ-mesh-registration-and-property-prediction/')

import torch
from fsgn_model import MeshSeg
from src.data.organs_dataset import OrganMeshDataset
from torch_geometric.data import DataLoader
from tqdm import tqdm
from time import sleep
import mlflow

def train(net, train_data, optimizer, loss_fn, device):
    """Train network on training dataset."""
    net.train()
    cumulative_loss = 0.0
    for data in train_data:
        data = data.to(device)
        optimizer.zero_grad()
        out = net(data)
        #print('Out shape ', out.shape)
        #print('data y shape', data.y.shape)
        loss = loss_fn(out, data.y.float())
        loss.backward()
        cumulative_loss += loss.item()
        optimizer.step()
    return cumulative_loss / len(train_data)

def calculate_val_loss(net, val_data, loss_fn, device):
    net.eval()
    cumulative_loss = 0.0
    for data in val_data:
        data = data.to(device)
        out = net(data)
        #print('Out shape ', out.shape)
        #print('data y shape', data.y.shape)
        loss = loss_fn(out, data.y.float())
        cumulative_loss += loss.item()
    return cumulative_loss / len(val_data)



def accuracy(predictions, gt_seg_labels):
    """Compute accuracy of predicted segmentation labels.

    Parameters
    ----------
    predictions: [|V|, num_classes]
        Soft predictions of segmentation labels.
    gt_seg_labels: [|V|]
        Ground truth segmentations labels.
    Returns
    -------
    float
        Accuracy of predicted segmentation labels.    
    """
    #predicted_seg_labels = predictions.argmax(dim=-1, keepdim=True)
    predicted_seg_labels = torch.nn.Sigmoid()(predictions)
    #predicted_seg_labels[predicted_seg_labels>0.5] = 1
    #predicted_seg_labels[predicted_seg_labels<0.5] = 0
    predicted_seg_labels = torch.round(predicted_seg_labels)
    #predicted_seg_labels = predictions.argmax(dim=-1, keepdim=True)
    #print('predicted_seg_labels : ',predicted_seg_labels.shape)
    if predicted_seg_labels.shape != gt_seg_labels.shape:
        raise ValueError("Expected Shapes to be equivalent")
    correct_assignments = (predicted_seg_labels == gt_seg_labels).sum()
    num_assignemnts = predicted_seg_labels.shape[0]
    return float(correct_assignments / num_assignemnts)


def evaluate_performance(dataset, net, device):
    """Evaluate network performance on given dataset.

    Parameters
    ----------
    dataset: DataLoader
        Dataset on which the network is evaluated on.
    net: torch.nn.Module
        Trained network.
    device: str
        Device on which the network is located.

    Returns
    -------
    float:x
        Mean accuracy of the network's prediction on
        the provided dataset.
    """
    prediction_accuracies = []
    for data in dataset:
        data = data.to(device)
        predictions = net(data)
        prediction_accuracies.append(accuracy(predictions, data.y))
    return sum(prediction_accuracies) / len(prediction_accuracies)

@torch.no_grad()
def test(net, train_data, test_data, device):
    net.eval()
    train_acc = evaluate_performance(train_data, net, device)
    test_acc = evaluate_performance(test_data, net, device)
    return train_acc, test_acc




if __name__ == '__main__':

    model_params = dict(
    in_features=3,
    encoder_features=128,
    conv_channels=[32, 64, 128, 64],
    encoder_channels=[128],
    decoder_channels=[32],
    num_classes=1,
    num_heads=12,
    apply_batch_norm=True,
)   
    device = torch.device('cuda:6')
    net = MeshSeg(**model_params).to(device)

    root = '/vol/chameleon/projects/mesh_gnn/organ_meshes'


    basic_feat_path = '/vol/chameleon/projects/mesh_gnn/basic_features.csv'
    bridge_path = '/vol/chameleon/projects/mesh_gnn/Bridge_eids_60520_87802.csv'
    split_path = '/u/home/koksal/organ-mesh-registration-and-property-prediction/data/'


    train_dataset = OrganMeshDataset(root, basic_feat_path, bridge_path, num_samples=3000, mode='train', split_path=split_path )
    val_dataset = OrganMeshDataset(root, basic_feat_path, bridge_path,  num_samples=500, mode='val', split_path=split_path )
    test_dataset = OrganMeshDataset(root, basic_feat_path, bridge_path,  num_samples=500, mode='test', split_path=split_path )

    train_loader = DataLoader(train_dataset,  shuffle=True)
    test_loader = DataLoader(test_dataset, shuffle=False)

    lr = 0.001
    num_epochs = 300
    best_test_acc = 0.0

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    with tqdm(range(num_epochs), unit="Epoch") as tepochs:
        for epoch in tepochs:
            train_loss = train(net, train_loader, optimizer, loss_fn, device)
            val_loss = calculate_val_loss(net, test_loader, loss_fn, device)
            mlflow.log_metric('train_loss',train_loss)
            mlflow.log_metric('val_loss',val_loss)
            train_acc, test_acc = test(net, train_loader, test_loader, device)
            mlflow.log_metric('train_acc',train_acc)
            mlflow.log_metric('test_acc',test_acc)

            
            tepochs.set_postfix(
                train_loss=train_loss,
                val_loss = val_loss,
                train_accuracy=100 * train_acc,
                test_accuracy=100 * test_acc,
            )
            sleep(0.1)

            if test_acc > best_test_acc:
                best_test_acc = test_acc
                torch.save(net.state_dict(), f"checkpoint_best_testacc_{test_acc}")


    

