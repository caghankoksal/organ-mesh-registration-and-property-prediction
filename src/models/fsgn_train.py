import sys
sys.path.append('/u/home/koksal/organ-mesh-registration-and-property-prediction/')

import torch
from fsgn_model import MeshSeg
from src.data.organs_dataset import OrganMeshDataset
from torch_geometric.data import DataLoader
from tqdm import tqdm
from time import sleep
import mlflow
from baseline_model import GCN
import argparse

def train(net, train_data, optimizer, loss_fn, device):
    """Train network on training dataset."""
    net.train()
    cumulative_loss = 0.0
    for data in train_data:
        data = data.to(device)
        optimizer.zero_grad()
        out = net(data)
        #print('Out shape ', out.shape, out)
        #print('data y shape', data.y.shape)
        loss = loss_fn(out.squeeze(1), data.y.float())
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
        loss = loss_fn(out.squeeze(1), data.y.float())
        cumulative_loss += loss.item()

        #print(f'Val loss is being calculated  val loss: {cumulative_loss}, len : {len(val_data)}')
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
   # print('Predictions', predictions.shape)
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
        prediction_accuracies.append(accuracy(predictions.squeeze(1), data.y))
    return sum(prediction_accuracies) / len(prediction_accuracies)

@torch.no_grad()
def test(net, train_data, test_data, device):
    net.eval()
    train_acc = evaluate_performance(train_data, net, device)
    test_acc = evaluate_performance(test_data, net, device)
    return train_acc, test_acc




def build_args():
    parser = argparse.ArgumentParser(description='GNN for Organ Meshes')
    parser.add_argument("--model", type=str, default="fsgnet")
    parser.add_argument("--device", type=int, default=5)
    parser.add_argument("--max_epoch", type=int, default=50,
                        help="number of training epochs")
    parser.add_argument("--enc_feats", type=int, default=128,
                        help="Encoder features")        
    parser.add_argument("--num_heads", type=int, default=12,
                        help="number of hidden attention heads")

    parser.add_argument("--hidden_channels", type=int, default=512,
                        help="Hidden dim of baseline")

    parser.add_argument("--num_train_samples", type=int, default=3000,
                        help="Number of training samples")  
    parser.add_argument("--num_test_samples", type=int, default=300,
                            help="Number of training samples")                        

    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--layer", type=str, default="gcn")
    parser.add_argument("--use_input_encoder", type=bool, default=True)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = build_args()
    print('Args : ',args)
    device = args.device if args.device >= 0 else "cpu"
    model = args.model
    max_epoch = args.max_epoch
    num_heads = args.num_heads
    enc_feats = args.enc_feats
    hidden_channels = args.hidden_channels
    batch_size = args.batch_size
    layer = args.layer
    num_train_samples = args.num_train_samples
    num_test_samples = args.num_test_samples
    use_input_encoder = args.use_input_encoder
    num_layers = args.num_layers 
    

    if model=='fsgnet':

        model_params = dict(
        use_input_encoder = use_input_encoder,
        in_features=3,
        encoder_features=enc_feats,
        conv_channels=[256, 256, 256, 256],
        encoder_channels=[enc_feats],
        decoder_channels=[256],
        num_classes=1,
        num_heads=num_heads,
        apply_batch_norm=True,  
    )   

        net = MeshSeg(**model_params).to(device)

    elif model=='baseline':
        print('Baseline Model is initialized')
        model_params = dict(
        num_classes=1,
        in_features=3, 
        hidden_channels=args.hidden_channels,
        layer = args.layer,
        use_input_encoder = args.use_input_encoder,
        input_encoder_dim = args.enc_feats,
        num_layers = args.num_layers,
        )
        net = GCN(**model_params).to(device)
    mlflow.start_run()
    for k,v in model_params.items():
        mlflow.log_param(k, v)

    


    root = '/vol/chameleon/projects/mesh_gnn/organ_meshes'


    basic_feat_path = '/vol/chameleon/projects/mesh_gnn/basic_features.csv'
    bridge_path = '/vol/chameleon/projects/mesh_gnn/Bridge_eids_60520_87802.csv'
    split_path = '/u/home/koksal/organ-mesh-registration-and-property-prediction/data/'
    train_dataset = OrganMeshDataset(root, basic_feat_path, bridge_path, num_samples=num_train_samples, mode='train', split_path=split_path )
    val_dataset = OrganMeshDataset(root, basic_feat_path, bridge_path,  num_samples=num_test_samples, mode='val', split_path=split_path )
    test_dataset = OrganMeshDataset(root, basic_feat_path, bridge_path,  num_samples=num_test_samples, mode='test', split_path=split_path )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,  shuffle=False)

    lr = 0.0001
    num_epochs = 50
    best_test_acc = 0.0

    mlflow.log_param("lr", lr)
    mlflow.log_param("num_epochs", num_epochs) 


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
                torch.save(net.state_dict(), f"/u/home/koksal/organ-mesh-registration-and-property-prediction/models/ckpts_enc_channels_{model_params['in_features']}_best_testacc_{test_acc:.2f}")

    print('Best Test Accuracy is ',best_test_acc)

    

