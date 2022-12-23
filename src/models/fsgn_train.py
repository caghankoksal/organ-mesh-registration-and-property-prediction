import sys
import wandb
sys.path.append('/u/home/koksal/organ-mesh-registration-and-property-prediction/')

import torch
from fsgn_model import MeshSeg
from src.data.organs_dataset import OrganMeshDataset
from torch_geometric.data import DataLoader
from tqdm import tqdm
from time import sleep
import mlflow
from baseline_model import GNN
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


def build_optimizer(network, optimizer, learning_rate):
    if optimizer == "sgd":
        optimizer = torch.optim.SGD(network.parameters(),
                              lr=learning_rate, momentum=0.9)
    elif optimizer == "adam":
        optimizer = torch.optim.Adam(network.parameters(),
                               lr=learning_rate)
    return optimizer


def build_network(configs):
    if configs.model=='fsgnet':

        model_params = dict(
        use_input_encoder = configs.use_input_encoder,
        num_classes=configs.num_classes,
        in_features=configs.in_features, 
        encoder_features=configs.enc_feats,
        conv_channels=[256, 256, 256, 256],
        encoder_channels=[configs.enc_feats],
        decoder_channels=[256],
        num_heads=num_heads,
        apply_batch_norm=True,  
    ) 

        net = MeshSeg(**model_params)

    elif configs.model=='baseline':
        
        print('Baseline Model is initialized')
        model_params = dict(
        use_input_encoder = configs.use_input_encoder,
        num_classes=configs.num_classes,
        in_features=configs.in_features, 
        encoder_features = configs.enc_feats,
        hidden_channels=configs.hidden_channels,
        layer = configs.layer,
        num_layers = configs.num_layers,
        )
        net = GNN(**model_params)

    return net


def build_dataset(config):
        # Build Dataset
    root = '/vol/chameleon/projects/mesh_gnn/organ_meshes'
    basic_feat_path = '/vol/chameleon/projects/mesh_gnn/basic_features.csv'
    bridge_path = '/vol/chameleon/projects/mesh_gnn/Bridge_eids_60520_87802.csv'
    split_path = '/u/home/koksal/organ-mesh-registration-and-property-prediction/data/'

    train_dataset = OrganMeshDataset(root, basic_feat_path, bridge_path, split_path=split_path,
                                    num_samples=config.num_train_samples, mode='train', organ=config.organ)

    val_dataset = OrganMeshDataset(root, basic_feat_path, bridge_path, split_path=split_path,
                                    num_samples=config.num_test_samples, mode='val', organ=config.organ)

    test_dataset = OrganMeshDataset(root, basic_feat_path, bridge_path,  split_path=split_path,
                                    num_samples=config.num_test_samples, mode='test', organ=config.organ)
                                    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size,  shuffle=False)

    return train_loader, test_loader

def training_function(config=None):
    print('Input cfg to training function ',config)
    
    # note that we define values from `wandb.config` instead of 
    # defining hard values
    config = wandb.config
    print('Training function config ',config)
    device = config.device

    #Data Loader
    train_loader, test_loader = build_dataset(config)

    #Network
    net = build_network(config).to(device)
    
    #Optimizer
    optimizer = build_optimizer(net, config.optimizer, config.lr)

    #Loss Function
    loss_fn = torch.nn.BCEWithLogitsLoss()

    best_test_acc = 0

    with tqdm(range(config.max_epoch), unit="Epoch") as tepochs:
        for epoch in tepochs:
            train_loss = train(net, train_loader, optimizer, loss_fn, device)
            val_loss = calculate_val_loss(net, test_loader, loss_fn, device)
            #mlflow.log_metric('train_loss',train_loss)
            #mlflow.log_metric('val_loss',val_loss)
            wandb.log({'train_loss': train_loss, 'val_loss':val_loss, 'epoch': epoch})
            train_acc, test_acc = test(net, train_loader, test_loader, device)
            #mlflow.log_metric('train_acc',train_acc)
            #mlflow.log_metric('test_acc',test_acc)
            wandb.log({'train_acc': train_acc, 'test_acc':test_acc, 'epoch': epoch})

            
            tepochs.set_postfix(
                train_loss=train_loss,
                val_loss = val_loss,
                train_accuracy=100 * train_acc,
                test_accuracy=100 * test_acc,
            )
            sleep(0.1)

            wandb.watch(net)

            if test_acc > best_test_acc:
                best_test_acc = test_acc
                wandb.run.summary["best_test_acc"] = 100 *best_test_acc
                wandb.run.summary["best_train_acc"] = 100 * train_acc
                savedir = '/u/home/koksal/organ-mesh-registration-and-property-prediction/models/'
                torch.save(net.state_dict(), f"{savedir}organ_{config.organ}_enc_channels_{config.hidden_channels}_best_testacc_{test_acc:.2f}")

    print('Best Test Accuracy is ',best_test_acc)


def build_args():
    parser = argparse.ArgumentParser(description='GNN for Organ Meshes')
    parser.add_argument("--model", type=str, default="baseline")
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
    parser.add_argument("--in_features", type=int, default=3)
    parser.add_argument("--num_classes", type=int, default=1)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0.0001)

    parser.add_argument("--layer", type=str, default="gcn")
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--use_input_encoder", type=bool, default=True)
    parser.add_argument("--hparam_search", type=bool, default=False)
    parser.add_argument("--organ", type=str, default="liver")
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
    lr = args.lr
    hparam_search = args.hparam_search

    print('Hyperparameter  Searching  ',hparam_search)
    if hparam_search is True:
        print('Hyperparameter search starts')

        run = wandb.init(project='meshgnn-hparam-search')

        sweep_config = {
        'method': 'random',
        'name': 'sweep',
        'metric': {'goal': 'maximize', 'name': 'val_acc'},
        'parameters': 
        {
            'batch_size': {'value': 32},
            'max_epoch': {'value': 50},
            'model': {'value': 'baseline'},
            'device': {'value': device},
            'optimizer': {'value': 'adam'},
            'lr': {'max': 0.1, 'min': 0.0001},
            'num_layers': {'values': [3, 4, 5, 6, 7, 8]},
            'use_input_encoder': {'value':True},
            'num_train_samples': {'value':3000},
            'num_test_samples': {'value':300},
            'layer': {'values':['sageconv','gcn']},
            'hidden_channels': {'values': [32, 64, 128, 256]},
            'enc_feats': {'values': [32, 64, 128, 256, 512]},
            'num_heads': {'values': [4, 8, 12]},
        }
    }
        # 🐝 Step 3: Initialize sweep by passing in config
        sweep_id = wandb.sweep(sweep=sweep_config, project='meshgnn-hparam-search')
        import pprint
        wandb.agent(sweep_id, training_function, count=10)

    else:
        print('Usual training starts')
        run = wandb.init(
        project="mesh_gnn_organ",
        notes="tweak baseline",
        tags=[ "gnn"],
        config=args,
        )

        training_function()

    

