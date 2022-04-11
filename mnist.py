import torch
import argparse
import numpy as np
import time
from DCN import DCN
from torchvision import datasets, transforms
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import normalized_mutual_info_score
from cluster_acc import acc
from utils import plot_losses, plot_acc, plot_pretrain_loss, plot_tSNE

class AddGaussianNoise(object):
  def __init__(self, mean=0, std=1):
    self.mean = mean
    self.std = std

  def __call__(self, tensor):
    shape = tensor.shape
    tensor += torch.normal(mean=self.mean*torch.ones(shape), std=self.std*torch.ones(shape))
    return torch.clamp(tensor, 0.0, 1.0)

# testing
def evaluate(e, model, test_loader, tsne_interval):
    y_test = []
    y_pred = []

    for idx, (data, target) in enumerate(test_loader):
        batch_size = data.size()[0]
        data = data.to(model.device)
        latent_X = model.autoencoder(data, latent=True)
        latent_X = latent_X.detach().cpu().numpy()
        if idx == 3 and e%tsne_interval == 0:
            plot_tSNE(latent_X, target, name=str(e))

        y_test.append(target.view(-1, 1).numpy())
        y_pred.append(model.kmeans.update_assign(latent_X).reshape(-1, 1))

    y_test = np.vstack(y_test).reshape(-1)
    y_pred = np.vstack(y_pred).reshape(-1)
    return (normalized_mutual_info_score(y_test, y_pred),
            adjusted_rand_score(y_test, y_pred),
            acc(y_test, y_pred))


def solver(args, model, train_loader, noisy_train_loader, test_loader):

    rec_loss_list_pretrain = model.pretrain(noisy_train_loader, args.pre_epoch)
    nmi_list = []
    ari_list = []
    acc_list = []
    train_loss_list = []
    total_rec_loss_list = []
    total_dist_loss_list = []
    tsne_interval = np.ceil(args.epoch/10.)

    for e in range(args.epoch):
        model.train()
        total_train_loss, total_rec_loss, total_dist_loss = model.fit(e, train_loader)

        model.eval()
        NMI, ARI, ACC = evaluate(e, model, test_loader, tsne_interval)  # evaluation on test_loader
        nmi_list.append(NMI)
        ari_list.append(ARI)
        acc_list.append(ACC)
        train_loss_list.append(total_train_loss)
        total_rec_loss_list.append(total_rec_loss)
        total_dist_loss_list.append(total_dist_loss)

        print('\nEpoch: {:02d} | NMI: {:.3f} | ARI: {:.3f} | ACC: {:.3f}\n'.format(
            e, NMI, ARI, ACC))

    return rec_loss_list_pretrain, train_loss_list, total_rec_loss_list, total_dist_loss_list, nmi_list, ari_list, acc_list


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Deep Clustering Network')

    # Dataset parameters
    parser.add_argument('--dir', default='../Dataset/mnist',
                        help='dataset directory')
    parser.add_argument('--input-dim', type=int, default=28*28,
                        help='input dimension')
    parser.add_argument('--n-classes', type=int, default=10,
                        help='output dimension')

    # Training parameters
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate (default: 1e-4)')
    parser.add_argument('--wd', type=float, default=5e-4,
                        help='weight decay (default: 5e-4)')
    parser.add_argument('--batch-size-train', type=int, default=256,
                        help='input batch size for training')    
    parser.add_argument('--batch-size-test', type=int, default=1000,
                        help='input batch size for testing')
    parser.add_argument('--epoch', type=int, default=100,
                        help='number of epochs to train')
    parser.add_argument('--pre-epoch', type=int, default=50, 
                        help='number of pre-train epochs')
    parser.add_argument('--pretrain', type=bool, default=True,
                        help='whether use pre-training')

    # Model parameters
    parser.add_argument('--lamda', type=float, default=1,
                        help='coefficient of the reconstruction loss')
    parser.add_argument('--beta', type=float, default=1,
                        help=('coefficient of the regularization term on '
                              'clustering'))
    parser.add_argument('--latent_dim', type=int, default=10,
                        help='latent space dimension')
    parser.add_argument('--kernel_sizes', type=int, default=[2, 2, 3],
                        help='kernel size dimensions for convolutional layers')
    parser.add_argument('--n-clusters', type=int, default=10,
                        help='number of clusters in the latent space')
    parser.add_argument('--input_channels', type=int, default=1,
                        help='number of channels in the input data')
    parser.add_argument('--hidden_channels', type=int, default=[32, 64, 128],
                        help='number of channels in the convolutional layer outputs')

    # Utility parameters
    parser.add_argument('--n-jobs', type=int, default=1,
                        help='number of jobs to run in parallel')
    parser.add_argument('--cuda', type=bool, default=True,
                        help='whether to use GPU')
    parser.add_argument('--log-interval', type=int, default=100,
                        help=('how many batches to wait before logging the '
                              'training status'))

    args = parser.parse_args()

    # Load data
    transformer = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize((0.1307,),
                                                           (0.3081,))])
    transformer_with_noise = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize((0.1307,),
                                                           (0.3081,)),
                                      AddGaussianNoise(0.0, 0.5)])

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(args.dir, train=True, download=True,
                       transform=transformer),
        batch_size=args.batch_size_train, shuffle=False)

    noisy_train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(args.dir, train=True, download=True,
                       transform=transformer_with_noise),
        batch_size=args.batch_size_train, shuffle=False)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(args.dir, train=False, transform=transformer),
        batch_size=args.batch_size_test, shuffle=True)

    # Main body
    start_time = time.time()
    model = DCN(args)
    rec_loss_list_pretrain, train_loss_list, total_rec_loss_list, total_dist_loss_list, nmi_list, ari_list, acc_list = solver(
        args, model, train_loader, noisy_train_loader, test_loader)
    if args.pretrain:
        plot_pretrain_loss(rec_loss_list_pretrain, name="pre-training")
    plot_acc(nmi_list, ari_list, acc_list, name="test")
    plot_losses(total_rec_loss_list, total_dist_loss_list, train_loss_list, name="train")
    print(f"Run-time  = {time.time()-start_time}")

