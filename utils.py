import matplotlib.pyplot as plt  
import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


def plot_losses(rec_loss, dist_loss, loss, name):
	fig = plt.figure(figsize = (12, 8))
	epochs = np.arange(1, len(rec_loss)+1, 1).tolist()
	plt.plot(epochs, rec_loss, label="Reconstruction loss")
	plt.plot(epochs, dist_loss, label="Clustering loss")
	plt.plot(epochs, loss, label="Total loss")
	plt.legend()
	Y_low, Y_high = get_axis_limits(rec_loss, dist_loss, loss)
	plt.axis([1, len(rec_loss), Y_low, Y_high])
	plt.xlabel("Number of epochs")
	plt.ylabel("Loss")
	plt.title("Loss : "+name)
	fig.savefig("losses_"+name+".jpeg", bbox_inches = "tight")


def plot_acc(NMI, ARI, ACC, name):
	fig = plt.figure(figsize = (12, 8))
	epochs = np.arange(1, len(NMI)+1, 1).tolist()
	plt.plot(epochs, NMI, label = 'NMI')
	plt.plot(epochs, ARI, label = 'ARI')
	plt.plot(epochs, ACC, label = 'ACC')
	plt.legend()
	Y_low, Y_high = get_axis_limits(NMI, ARI, ACC)
	plt.axis([1, len(NMI), Y_low, Y_high])
	plt.xlabel("Number of epochs")
	plt.ylabel("Metric")
	plt.title("Accuracy of clustering : "+name)
	fig.savefig("Accuracy_"+name+".jpeg", bbox_inches = "tight")

def plot_tSNE(X, y, name):
	tsne = make_pipeline(StandardScaler(), TSNE(n_components=2, init='pca', random_state=42))
	tsne.fit(X, y)
	X_tsne = tsne.fit_transform(X)
	fig = plt.figure(figsize = (12, 8))
	plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, s=30, cmap='tab10')
	plt.title(name)
	plt.colorbar()
	fig.savefig("t-SNE_epoch_"+name+".jpeg", format = "jpeg", bbox_inches = "tight")

def plot_pretrain_loss(rec_loss_pretrain, name):
	fig = plt.figure(figsize = (12, 8))
	epochs = np.arange(1, len(rec_loss_pretrain)+1, 1).tolist()
	plt.plot(epochs, rec_loss_pretrain, label="Reconstruction loss")
	plt.legend()
	Y_low, Y_high = get_axis_limits(rec_loss_pretrain, rec_loss_pretrain, rec_loss_pretrain)
	plt.axis([1, len(rec_loss_pretrain), Y_low, Y_high])
	plt.xlabel("Number of epochs")
	plt.ylabel("Reconstruction Loss")
	plt.title("Loss : "+name)
	fig.savefig("pretrain_losses_"+name+".jpeg", bbox_inches = "tight")	

def get_axis_limits(X, Y, Z):
	lower = min(min(X), min(Y), min(Z))
	upper = max(max(X), max(Y), max(Z))
	lower = lower - 0.02*(upper-lower)
	upper = upper + 0.02*(upper-lower)
	return lower, upper


