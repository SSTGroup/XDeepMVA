import os
import numpy as np
import pickle as pkl
import tensorflow as tf
from sklearn.manifold import TSNE
from sklearn.cluster import SpectralClustering
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
from tqdm.notebook import tqdm
from utils import CCA

class experiment():
    def __init__(self, log_dir, dataprovider, model, dim, cca_reg, lambda_rec, lambda_l2, eval_epochs):
        self.dataprovider = dataprovider
        self.model = model
        self.cca_reg = cca_reg
        self.lambda_rec = lambda_rec
        self.lambda_l2 = lambda_l2
        self.eval_epochs = eval_epochs
        self.best_acc = 0.0
        
        self.dim = dim
        self.tb_writer = TensorboardWriter(log_dir)
        self.epoch = 1
        self.optimizer = tf.keras.optimizers.Adam()
            
    def train(self, num_epochs):
        """
        Main training function
        """
        # Load training data once
        training_data = self.dataprovider.training_data

        # Iterate over epochs
        for epoch in tqdm(range(num_epochs), desc='Epochs', leave=False):
            # Train one epoch
            self.train_single_epoch(training_data)

    def train_single_epoch(self, training_data):
        """
        Fuction for a single epoch, includes forward and backward path as well as metric logging
        """
        with tf.GradientTape() as tape:
            # Feed forward
            network_output = self.model(training_data, training=True, cca_reg=self.cca_reg)
            # Compute loss
            loss = self.compute_loss(network_output, training_data)

        # Compute gradients
        gradients = tape.gradient(loss, self.model.trainable_variables)
        # Apply gradients
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        # Log metrics
        _ = self.compute_metrics(network_output)

        # Increase epoch counter
        self.epoch += 1
        
    def compute_loss(self, network_output, data):
        """
        Compute the single loss terms and add them up at the end
        """
        # CCA loss
        cca_loss = -1*tf.reduce_sum(network_output['ccor']) / self.dim
        
        # Reconstruction loss
        rec_loss_1 = tf.square(tf.norm(tf.cast(data['nn_input_0'], tf.float32) - network_output['reconst_view_0'], axis=0))
        rec_loss_2 = tf.square(tf.norm(tf.cast(data['nn_input_1'], tf.float32) - network_output['reconst_view_1'], axis=0))
        rec_loss = tf.reduce_mean(tf.add(rec_loss_1, rec_loss_2))
        rec_loss *= self.lambda_rec
        
        # L2 loss
        l2_loss = self.model.get_l2()
        l2_loss *= self.lambda_l2
        
        loss = cca_loss + rec_loss + l2_loss
        
        # Every 10-th epoch we write a summary for the losses to get visualization in the Tensorboard
        if self.epoch%10==0:
            self.tb_writer.write_scalar_summary(
                epoch=self.epoch, 
                list_of_tuples=[
                    (loss, 'Loss/Total'),
                    (cca_loss, 'Loss/CCA'),
                    (rec_loss, 'Loss/Reconstruction'),
                    (l2_loss, 'Loss/L2')
                ]
            )
        
        return loss
        
    def compute_metrics(self, network_output):
        """
        Compute and log all metrics of interest during training
        """
        metrics = dict()

        # Just eval after 'eval_epochs' many epochs
        if self.epoch % self.eval_epochs == 0:
            # We save the current canonical correlation values
            metrics['ccor'] = network_output['ccor']

            # We compute the clustering result on view 1
            clust_acc = self.eval_clustering(view=1)

            # If we have a new best accuracy, save the model and update the metric
            if clust_acc > self.best_acc:
                self.save_weights()
                self.best_acc = clust_acc

            self.tb_writer.write_scalar_summary(
                epoch=self.epoch, 
                list_of_tuples=[(network_output['ccor'][i], 'Correlations/'+str(i)) for i in range(self.dim)] +
                [
                    (clust_acc, 'Accuracy/Clustering'),
                ]
            )
            
    def eval_clustering(self, view=1):
        """
        Evaluate the clustering performance on training data
        """
        netw_output = self.model(self.dataprovider.training_data.copy())
        labels = np.squeeze(self.dataprovider.training_data['labels'].numpy())

        # Use the CCA view of either view 0 or 1
        if view==0:
            latent_view = netw_output['cca_view_0']
        else:
            latent_view = netw_output['cca_view_1']
        
        # Do the acutal clustering
        acc, cmatrix, clust_labels, prediction = self.compute_clustering_accuracy(latent_view, labels)
        
        return acc

    def compute_clustering_accuracy(self, pointcloud, labels):
        """
        Cluster a given pointcloud an compute accuracy as well as confusion matrix
        """
        
        clust_labels = SpectralClustering(
            n_clusters=2,
            assign_labels='kmeans',
            affinity='nearest_neighbors',
            random_state=33,
            n_init=10).fit_predict(pointcloud)

        # Assing cluster labels to our labels by majority voting
        prediction = np.zeros_like(clust_labels)
        for i in range(2):
            ids = np.where(clust_labels == i)[0]
            prediction[ids] = np.argmax(np.bincount(labels[ids]))

        # Compute the accuracy
        acc = accuracy_score(labels, prediction)
                                 
        # Compute the confusion matrix
        cmatrix = tf.math.confusion_matrix(
            labels,
            prediction,
        )

        return acc, cmatrix, clust_labels, prediction

    def analyse_subspace(self, views=[1], method='DCCAE', latent_dim=3, save_path=None):
        """
        Main function for analysing a trained model. Supports evaluation of:
         - The trained DCCAE method (ignores the latent_dim argmuent)
         - PCA 
         - CCA
        DCCAE method just uses the trained model and evaluates the clustering performance.
        PCA and CCA are computed with latentent_dimension given to create a baseline.
        
        We can either evaluate views on their own or concatenated: [0], [1], [0,1]
        """
        data = self.dataprovider.training_data.copy()
        labels = np.squeeze(self.dataprovider.training_data['labels'].numpy())

        assert method in ['DCCAE', 'CCA', 'PCA']

        # Switch between methods
        if method == 'DCCAE':
            # Use the trained model to predict the views
            netw_output = self.model(data)
            view_0 = netw_output['cca_view_0'].numpy()
            view_1 = netw_output['cca_view_1'].numpy()
            # We also compute the canonical correlation value for the views
            _, _, _, _, D = CCA(view_0, view_1, num_shared_dim=latent_dim, rx=self.cca_reg, ry=self.cca_reg)
            print('CCorValue: '+str(D.numpy()))
        elif method == 'PCA':
            # We use the result of the PCA of the raw input
            view0_raw = data['nn_input_0']
            pca = PCA(n_components=latent_dim)
            pca.fit(view0_raw)
            view_0 = pca.transform(view0_raw)

            view1_raw = data['nn_input_1']
            pca = PCA(n_components=latent_dim)
            pca.fit(view1_raw)
            view_1 = pca.transform(view1_raw)
        elif method == 'CCA':
            # We first do PCA on the raw input
            view0_raw = data['nn_input_0']
            pca = PCA(n_components=latent_dim)
            pca.fit(view0_raw)
            view_0 = pca.transform(view0_raw)

            view1_raw = data['nn_input_1']
            pca = PCA(n_components=latent_dim)
            pca.fit(view1_raw)
            view_1 = pca.transform(view1_raw)

            # And then do CCA to get the final view with canonical correlation values
            A, B, view_0, view_1, D = CCA(view_0, view_1, num_shared_dim=latent_dim)
            print('CCorValue: '+str(D.numpy()))
            view_0, view_1 = view_0.numpy().T, view_1.numpy().T
        else:
            raise NotImplementedError

        assert isinstance(views, list)
        assert len(views) in [1,2]

        # Here we choose which view to use for the evaluation
        if len(views) == 2:
            latent_view = np.concatenate([view_0, view_1], axis=1)
        else:
            if views[0] == 0:
                latent_view = view_0
            else:
                latent_view = view_1

        dimensionality = latent_view.shape[1]
        num_classes = np.max(labels)+1
        
        # Do the clustering
        acc, cmatrix, clust_labels, prediction = self.compute_clustering_accuracy(latent_view, labels)
        print("Clustering Accuracy: "+str(acc))        
        print("Confusion matrix: \n"+str(cmatrix.numpy()))
        
        # Here follows a lot of figure creation and plotting for different dimensionalities
        if dimensionality <= 2:
            dims_to_extend = 2 - latent_view.shape[1]
            if dims_to_extend > 0:
                vis_latent_view = np.concatenate([latent_view, np.zeros((latent_view.shape[0], dims_to_extend))], axis=1)
            else:
                vis_latent_view = latent_view

            labels_str = labels.astype(str)
            labels_str[labels_str=='0'] = 'No seizure'
            labels_str[labels_str=='1'] = 'Seizure'

            clust_labels_str = clust_labels.astype(str)
            clust_labels_str[clust_labels_str=='0'] = 'Cluster 1'
            clust_labels_str[clust_labels_str=='1'] = 'Cluster 2'

            fig = px.scatter(
                x=vis_latent_view[:,0], 
                y=vis_latent_view[:,1],  
                color=labels_str,
                color_discrete_sequence=['rgb(230,0,0)', 'rgb(0,230,0)'],
                symbol=clust_labels_str,
                labels={"Group", "Cluster"},
                size=labels+1,
                size_max=10,
                opacity=1
            )
            fig.update_layout(
                legend_orientation='h',
                showlegend=True)
            
            fig.update_traces(marker=dict(line=dict(width=0,)))
            fig.update_xaxes(visible=False)
            fig.update_yaxes(visible=False)
            fig.show()

        elif dimensionality == 3:
            labels_str = labels.astype(str)
            labels_str[labels_str=='0'] = 'No seizure'
            labels_str[labels_str=='1'] = 'Seizure'

            clust_labels_str = clust_labels.astype(str)
            clust_labels_str[clust_labels_str=='0'] = 'Cluster 1'
            clust_labels_str[clust_labels_str=='1'] = 'Cluster 2'
                
            fig = px.scatter_3d(
                x=latent_view[:,0], 
                y=latent_view[:,1], 
                z=latent_view[:,2], 
                color=labels_str,
                color_discrete_sequence=['rgb(230,110,0)', 'rgb(0,0,0)'],
                symbol=clust_labels_str,
                labels={"Group", "Cluster"}
            )
            
            fig.update_layout(
                legend_orientation='h',
                scene = dict(
                    xaxis = dict(showticklabels=False, showgrid=False,),
                    yaxis = dict(showticklabels=False, showgrid=False,),
                    zaxis = dict(showticklabels=False, showgrid=False,),
                ),
                legend_title_text='Group, Cluster',
                showlegend=True
            )
            fig.show()

        else:
            data_embedded = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(latent_view)
            
            labels_str = labels.astype(str)
            labels_str[labels_str=='0'] = 'No seizure'
            labels_str[labels_str=='1'] = 'Seizure'

            clust_labels_str = clust_labels.astype(str)
            clust_labels_str[clust_labels_str=='0'] = 'Cluster 1'
            clust_labels_str[clust_labels_str=='1'] = 'Cluster 2'
            fig = px.scatter(
                x=data_embedded[:,0], 
                y=data_embedded[:,1],  
                color=labels_str,
                color_discrete_sequence=['rgb(230,0,0)', 'rgb(0,230,0)'],
                symbol=clust_labels_str,
                labels={"Group", "Cluster"},
                size=np.ones_like(labels),
                size_max=5,
                opacity=1
            )
            fig.update_layout(
                legend_orientation='h',
                showlegend=True)
            fig.update_layout(
                {'plot_bgcolor': 'rgba(0, 0, 0, 0)'}
            )
            fig.update_traces(marker=dict(line=dict(width=0,)))
            fig.update_xaxes(visible=False)
            fig.update_yaxes(visible=False)
            fig.show()

            fig.write_image("clustering.svg")

        # We also do the individual analysis here
        windows_per_patient = int(prediction.shape[0]/42)
        total_windows = prediction.shape[0]
        patients_accuracies = list()
        # We compute accuracies just for a single participant
        for j in range(int(total_windows/windows_per_patient)):
            start_idx = j*windows_per_patient
            end_idx = (j+1)*windows_per_patient
            patient_prediction = prediction[start_idx:end_idx]
            patient_labels = labels[start_idx:end_idx]
            patients_accuracies.append(accuracy_score(patient_labels, patient_prediction))
            
        patients_accuracies = np.asarray(patients_accuracies)

        # Plotting
        share_50p = len(patients_accuracies[patients_accuracies>0.5]) / len(patients_accuracies)

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=np.arange(1,22),
                y=patients_accuracies[:21], 
                mode='markers',
                marker=dict(color='rgb(230,0,0)', size=8)
            )
        )
        fig.add_trace(
            go.Scatter(
                x=np.arange(22,43),
                y=patients_accuracies[21:], 
                mode='markers',
                marker=dict(color='rgb(0,230,0)', size=8)
            )
        )
        fig.update_xaxes(showgrid=False, title='Patient number', tickvals=[1, 7, 14, 21, 28, 35, 42], range=[0.5,42.5])
        fig.update_yaxes(zeroline=True, zerolinecolor='rgb(150,150,150)', title="Accuracy", tickvals=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], range=[-0.05, 1.05])
        fig.layout.yaxis.gridcolor='rgb(150,150,150)'
        fig.add_hline(y=0.5, line_dash="dot",
              annotation_text="{:.2f}".format(share_50p), 
              annotation_position="bottom right")
        fig.update_layout(
            {'plot_bgcolor': 'rgba(0, 0, 0, 0)'}
        )
        fig.update_layout(
            showlegend=False
        )
        fig.show()
        
        fig.write_image("accuracies.svg")
        
        print(np.histogram(patients_accuracies, bins=10))
        
    def save_weights(self):
        matrices_file = os.path.join(self.tb_writer.dir, 'matrices.pkl')
        with open(matrices_file, 'wb') as f:
            pkl.dump(dict(B1=self.model.B1, B2=self.model.B2), f)
        self.model.save_weights(filepath=self.tb_writer.dir)

    def load_best(self):
        matrices_file = os.path.join(self.tb_writer.dir, 'matrices.pkl')
        with open(matrices_file, 'rb') as f:
            B_dict = pkl.load(f)
            B1 = B_dict['B1']
            B2 = B_dict['B2']
        self.model.load_weights(filepath=self.tb_writer.dir)
        self.model.B1 = B1
        self.model.B2 = B2

    def save(self):
        self.model.save(self.tb_writer.dir)
        
class TensorboardWriter():
    def __init__(self, root_dir):
        folders = list()
        for file in os.listdir(root_dir):
            file_path = os.path.join(root_dir, file)
            if os.path.isdir(file_path):
                folders.append(file_path)

        curr_number = 0
        while True:
            num_str = str(curr_number)
            if len(num_str) == 1:
                num_str = "0"+num_str

            folder = os.path.join(root_dir, num_str)
            if not os.path.exists(folder):
                break
            else:
                curr_number = curr_number + 1

        os.makedirs(folder)

        self.writer = tf.summary.create_file_writer(folder)
        self.dir = folder

    def write_scalar_summary(self, epoch, list_of_tuples):
        with self.writer.as_default():
            for tup in list_of_tuples:
                tf.summary.scalar(tup[1], tup[0], step=epoch)
        self.writer.flush()
