import torch
import numbers
import numpy as np
import torch.nn as nn
from kmeans import batch_KMeans
from autoencoder import AutoEncoder


class DCN(nn.Module):

    def __init__(self, args):
        super(DCN, self).__init__()
        self.args = args
        self.beta = args.beta  # coefficient of the clustering term
        self.lamda = args.lamda  # coefficient of the reconstruction term
        self.device = torch.device('cuda' if args.cuda else 'cpu')

        # Validation check
        if not self.beta > 0:
            msg = 'beta should be greater than 0 but got value = {}.'
            raise ValueError(msg.format(self.beta))

        if not self.lamda > 0:
            msg = 'lambda should be greater than 0 but got value = {}.'
            raise ValueError(msg.format(self.lamda))

        if len(self.args.hidden_channels) == 0:
            raise ValueError('No hidden layer specified.')

        if len(self.args.kernel_sizes) == 0:
            raise ValueError('No kernel sizes specified for convolutional layers.')

        self.kmeans = batch_KMeans(args)
        self.autoencoder = AutoEncoder(args).to(self.device)

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(),
                                          lr=args.lr,
                                          weight_decay=args.wd)

    """ Compute the Equation (5) in the original paper on a data batch """
    def _loss(self, X, cluster_id):
        batch_size = X.size()[0]
        rec_X = self.autoencoder(X)
        latent_X = self.autoencoder(X, latent=True)

        # Reconstruction error
        rec_loss = self.lamda * self.criterion(X, rec_X)

        # Regularization term on clustering
        dist_loss = torch.tensor(0.).to(self.device)
        clusters = torch.FloatTensor(self.kmeans.clusters).to(self.device)
        for i in range(batch_size):
            diff_vec = latent_X[i] - clusters[cluster_id[i]]
            sample_dist_loss = torch.matmul(diff_vec.view(1, -1),
                                            diff_vec.view(-1, 1))
            dist_loss += 0.5 * self.beta * torch.squeeze(sample_dist_loss)

        return (rec_loss + dist_loss,
                rec_loss.detach().cpu().numpy(),
                dist_loss.detach().cpu().numpy())

    # Pre-training
    def pretrain(self, train_loader, epoch=100, verbose=True):

        if not self.args.pretrain:
            return

        if not isinstance(epoch, numbers.Integral):
            msg = '`epoch` should be an integer but got value = {}'
            raise ValueError(msg.format(epoch))

        if verbose:
            print('========== Start pretraining ==========')

        rec_loss_list = []

        self.train()
        for e in range(epoch):
            for batch_idx, (data, _) in enumerate(train_loader):
                batch_size = data.size()[0]
                data = data.to(self.device)
                rec_X = self.autoencoder(data)
                loss = self.criterion(data, rec_X)

                if verbose and batch_idx % self.args.log_interval == 0:
                    msg = 'Epoch: {:02d} | Batch: {:03d} | Rec-Loss: {:.3f}'
                    print(msg.format(e, batch_idx,
                                     loss.detach().cpu().numpy()))
                    rec_loss_list.append(loss.detach().cpu().numpy())

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        self.eval()

        if verbose:
            print('========== End pretraining ==========\n')

        # Initialize clusters in self.kmeans after pre-training
        batch_X = []
        for batch_idx, (data, _) in enumerate(train_loader):
            batch_size = data.size()[0]
            data = data.to(self.device)
            latent_X = self.autoencoder(data, latent=True)
            batch_X.append(latent_X.detach().cpu().numpy())
        batch_X = np.vstack(batch_X)
        self.kmeans.init_cluster(batch_X)

        return rec_loss_list

    def fit(self, epoch, train_loader, verbose=True):
        train_loss = 0.0
        total_rec_loss = 0.0
        total_dist_loss = 0.0

        for batch_idx, (data, _) in enumerate(train_loader):
            batch_size = data.size()[0]
            data = data.to(self.device)

            # Get the latent features
            with torch.no_grad():
                latent_X = self.autoencoder(data, latent=True)
                latent_X = latent_X.cpu().numpy()

            # [Step-1] Update the assignment results
            cluster_id = self.kmeans.update_assign(latent_X)

            # [Step-2] Update clusters in batch Kmeans
            elem_count = np.bincount(cluster_id,
                                     minlength=self.args.n_clusters)
            for k in range(self.args.n_clusters):
                # avoid empty slicing
                if elem_count[k] == 0:
                    continue
                self.kmeans.update_cluster(latent_X[cluster_id == k], k)

            # [Step-3] Update the network parameters
            loss, rec_loss, dist_loss = self._loss(data, cluster_id)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_loss += loss.detach().cpu().numpy()/batch_size
            total_rec_loss += rec_loss/batch_size
            total_dist_loss += dist_loss/batch_size

            if verbose and batch_idx % self.args.log_interval == 0:
                msg = 'Epoch: {:02d} | Batch: {:03d} | Loss: {:.3f} | Rec-' \
                      'Loss: {:.3f} | Dist-Loss: {:.3f}'
                print(msg.format(epoch, batch_idx,
                                 loss.detach().cpu().numpy(),
                                 rec_loss, dist_loss))

        return train_loss, total_rec_loss, total_dist_loss
