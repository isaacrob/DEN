from sklearn.base import BaseEstimator
from sklearn.metrics import normalized_mutual_info_score, mutual_info_score, silhouette_score, davies_bouldin_score, calinski_harabasz_score, v_measure_score, adjusted_mutual_info_score, log_loss
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model, naive_bayes
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from scipy.linalg import eigh
from scipy.optimize import linear_sum_assignment
from scipy.stats import entropy as get_entropy
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset
from torch.nn.utils.rnn import pad_sequence
import progressbar
import copy
import matplotlib.pyplot as plt
import shap
import contextlib
import sys
from collections import defaultdict, Counter

from models import *
from losses import f_loss, reg_betainc
from plots import plot_2d, plot_3d


simple_classifier = linear_model.LogisticRegression(solver = 'lbfgs', n_jobs = -1)


def change_cluster_labels_to_sequential(clusters):
    labels = np.unique(clusters)
    clusters_to_labels = {cluster:i for i, cluster in enumerate(labels)}
    seq_clusters = np.array([clusters_to_labels[cluster] for cluster in clusters])

    return seq_clusters

def make_cost_matrix(c1, c2):
    c1 = change_cluster_labels_to_sequential(c1)
    c2 = change_cluster_labels_to_sequential(c2)
    
    uc1 = np.unique(c1)
    uc2 = np.unique(c2)
    l1 = uc1.size
    l2 = uc2.size
    assert(l1 == l2 and np.all(uc1 == uc2)), str(uc1) + " vs " + str(uc2)

    m = np.ones([l1, l2])
    for i in range(l1):
        it_i = np.nonzero(c1 == uc1[i])[0]
        for j in range(l2):
            it_j = np.nonzero(c2 == uc2[j])[0]
            m_ij = np.intersect1d(it_j, it_i)
            m[i,j] =  -m_ij.size

    return m

def get_accuracy(clusters, labels):
    cost = make_cost_matrix(clusters, labels)
    row_ind, col_ind = linear_sum_assignment(cost)
    to_labels = {i: ind for i, ind in enumerate(col_ind)}
    clusters_as_labels = list(map(to_labels.get, clusters))
    acc = np.sum(clusters_as_labels == labels) / labels.shape[0]

    return acc

def tokens_to_tfidf(x):
    list_of_strs = [' '.join(str(token) for token in item if token != 0) for item in x]
    out = TfidfVectorizer().fit_transform(list_of_strs)

    return out

class DummyFile(object):
    def write(self, x): pass
    def flush(self): pass

@contextlib.contextmanager
def nostdout():
    save_stdout = sys.stdout
    sys.stdout = DummyFile()
    yield
    sys.stdout = save_stdout

class DEN(BaseEstimator):
    def __init__(
        self,
        n_components = 2,
        model = 'auto',
        min_neighbors = 1,
        max_neighbors = 10,
        snn = True,
        batch_size = 256,
        ignore = 1,
        metric = 'euclidean',
        neighbors_preprocess = None,
        use_gpu = True,
        learning_rate = 1e-3,
        optimizer_override = None,
        epochs = 10,
        verbose_level = 3,
        random_seed = 37,
        gamma = 1,
        semisupervised = False,
        cluster_subnet_dropout_p = .3,
        is_tokens = False,
        cluster_subsample_n = 1000,
        initial_zero_cutoff = 1e-2,
        minimum_zero_cutoff = 1e-7,
        update_zero_cutoff = False,
        internal_dim = 128,
        cluster_subnet_training_epochs = 50,
        semisupervised_weight = None,
        l2_penalty = 0,
        prune_graph = False,
        fine_tune_end_to_end = True,
        fine_tune_epochs = 50,
        simple_classifier = simple_classifier,
        final_training_epochs = 20,
        final_dropout_p = .3,
        min_p = 0,
        max_correlation = 0
    ):
        self.n_components = n_components
        self.model = model
        self.min_neighbors = min_neighbors
        self.max_neighbors = max_neighbors
        self.snn = snn
        self.batch_size = batch_size
        self.ignore = ignore
        self.metric = metric
        self.neighbors_preprocess = neighbors_preprocess
        self.use_gpu = use_gpu
        self.learning_rate = learning_rate
        self.optimizer_override = optimizer_override
        self.epochs = epochs
        self.verbose_level = verbose_level
        self.random_seed = random_seed
        self.gamma = gamma
        self.semisupervised = semisupervised
        self.cluster_subnet_dropout_p = cluster_subnet_dropout_p
        self.is_tokens = False # forces TF-IDF preprocessing if preprocessing unspecified
        self.cluster_subsample_n = cluster_subsample_n
        self.initial_zero_cutoff = initial_zero_cutoff
        self.minimum_zero_cutoff = minimum_zero_cutoff
        self.update_zero_cutoff = update_zero_cutoff
        self.internal_dim = internal_dim
        self.cluster_subnet_training_epochs = cluster_subnet_training_epochs
        self.semisupervised_weight = semisupervised_weight
        self.l2_penalty = l2_penalty
        self.prune_graph = prune_graph
        self.fine_tune_end_to_end = fine_tune_end_to_end
        self.fine_tune_epochs = fine_tune_epochs
        self.simple_classifier = simple_classifier
        self.final_training_epochs = final_training_epochs
        # self.final_model = final_model
        self.final_dropout_p = final_dropout_p
        self.min_p = min_p
        self.max_correlation = max_correlation

        self.best_full_net = None
        self.best_embedding_net = None
        self.optimizer = None
        self.semisupervised_model = None
        # self.final_model = None

    def find_differentiating_features(self, sample, context, n_context_samples = 400, feature_names = None):
        assert self.best_full_net is not None, "have not trained a prediction network yet!"

        if n_context_samples < context.shape[0]:
            context_subsample_inds = np.random.choice(context.shape[0], n_context_samples, replace = False)
            context_subsample = context[context_subsample_inds]
        else:
            context_subsample = context

        e = shap.DeepExplainer(self.best_full_net, context_subsample)

        if sample.shape[0] == 1:
            # only one sample so assuming need to add batch dimension
            sample = sample.unsqueeze(0)

        with nostdout():
            shap_values, indexes = e.shap_values(sample, ranked_outputs = 1)

        if type(sample) is not np.ndarray:
            sample = sample.cpu().numpy()

        if len(context.shape) == 4:
            # assuming image
            shap_values = [np.swapaxes(np.swapaxes(s, 2, 3), 1, -1) for s in shap_values]
            if sample.shape[1] == 1:
                # need valid image shape for matplotlib
                sample = sample.squeeze(1)
            shap.image_plot(shap_values, -sample)
        else:
            # assuming not image
            shap.force_plot(e.expected_value[0], shap_values[0], sample, feature_names = feature_names, matplotlib = True)

    def summerize_differentiating_features(self, X, n_samples = 200, n_context_samples = 400):
        # split the dataset into clusters and average differentiating features in each cluster
        n_samples = min(n_samples, X.shape[0])
        sample_inds = np.random.choice(X.shape[0], n_samples, replace = False)
        samples = X[sample_inds]
        clusters = self.predict(samples)

        n_context_samples = min(n_context_samples, X.shape[0])
        context_sample_inds = np.random.choice(X.shape[0], n_context_samples, replace = False)
        context_samples = X[context_sample_inds]
        context_samples = context_samples.to(self.device)

        e = shap.DeepExplainer(self.best_full_net, context_samples)

        summerizations = defaultdict(lambda : np.zeros(X.shape[1:]))
        average_samples = defaultdict(lambda : np.zeros(X.shape[1:]))
        counts = Counter(clusters)

        self._print_with_verbosity("finding differentiating features across the dataset...", 1)

        for cluster, sample in self._progressbar_with_verbosity(zip(clusters, samples), 1, max_value = n_samples):
            with nostdout():
                shap_values, indexes = e.shap_values(sample.unsqueeze(0), ranked_outputs = 1)
            summerizations[cluster] += shap_values[0].squeeze(0) / counts[cluster]
            average_samples[cluster] += sample.cpu().numpy() / counts[cluster]

        # recall that dictionaries are ordered in Python3
        summery = np.array(list(summerizations.values())).squeeze(1)
        averages = np.array(list(average_samples.values())).squeeze(1)

        shap.image_plot(summery, -averages)


    def _print_with_verbosity(self, message, level, strict = False):
        if level <= self.verbose_level and (not strict or level == self.verbose_level):
            print(message)

    def _progressbar_with_verbosity(self, data, level, max_value = None, strict = False):
        if level <= self.verbose_level and (not strict or level == self.verbose_level):
            for datum in progressbar.progressbar(data, max_value = max_value):
                yield datum
        else:
            for datum in data:
                yield datum

    def _select_model(self, X, n_outputs = None, dropout_p = 0):
        if n_outputs is None:
            n_outputs = self.n_components
        # not sure if allowed to modify model attribute under sklearn rules
        if type(X) is tuple or type(X) is list:
            self._print_with_verbosity("assuming token-based data, using bag-of-words model", 1)
            self.is_tokens = True
            vocab = set()
            for x in X:
                vocab.update(x)
            vocab_size = len(vocab)
            to_model = BOWNN(n_outputs, vocab_size, internal_dim = self.internal_dim)
        else:
            n_dims = len(X.shape)
            if n_dims == 2:
                self._print_with_verbosity("using fully connected neural network", 1)
                to_model = FFNN(n_outputs, X.shape[1], internal_dim = self.internal_dim)
            elif n_dims == 4:
                self._print_with_verbosity("using convolutional neural network", 1)
                n_layers = int(np.log2(min(X.shape[2], X.shape[3])))
                to_model = CNN(n_outputs, n_layers, internal_dim = self.internal_dim, p = dropout_p)
                # self.model = ClusterNet(X.shape[-1]*X.shape[-2], self.n_components)
            else:
                assert False, "not sure which neural network to use based off data provided"

        return to_model

    def _get_near_and_far_pairs_mem_efficient_chunks(self, X, block_size = 512, return_sorted = True):
        n_neighbors = self.max_neighbors

        closest = []
        furthest = []
        if type(X) is np.ndarray:
            splits = np.array_split(X, max(X.shape[0] // block_size, 1))
            max_value = len(splits)
        else:
            inds = list(range(0, X.shape[0], block_size))
            inds.append(None)
            splits = (X[inds[i]:inds[i+1]] for i in range(len(inds) - 1))
            max_value = len(inds) - 1

        self._print_with_verbosity(f"using metric {self.metric} to build nearest neighbors graph", 2)

        for first in self._progressbar_with_verbosity(splits, 2, max_value = max_value):
            dists = pairwise_distances(first, X, n_jobs = -1, metric = self.metric)
            # dists = cdist(first, X, metric = metric)
            this_closest = np.argpartition(dists, n_neighbors + 1)[:, :n_neighbors+1]
            if return_sorted:
                original_set = set(this_closest[-1])
                relevant = dists[np.arange(this_closest.shape[0])[:, None], this_closest]
                sorted_inds = np.argsort(relevant)
                this_closest = this_closest[np.arange(sorted_inds.shape[0])[:, None], sorted_inds]
                assert set(this_closest[-1]) == original_set, "something went wrong with sorting"
                this_closest = this_closest[:, 1:]
            closest.append(this_closest)

            probs = dists / np.sum(dists, axis = 1)[:, None]
            this_furthest = np.array([np.random.choice(len(probs[i]), n_neighbors, False, probs[i]) for i in range(len(probs))])
            furthest.append(np.array(this_furthest))

        closest = np.concatenate(closest)
        furthest = np.concatenate(furthest)

        return closest, furthest

    def _build_dataset(self, X, y = None):
        # returns Dataset object 
        neighbors_X = X.view(X.shape[0], -1).cpu().numpy()

        if self.is_tokens and self.neighbors_preprocess is None:
            self._print_with_verbosity("using tokenized data without neighbors preprocessing so using TF-IDF transform", 2)
            self.neighbors_preprocess = tokens_to_tfidf
            self.metric = 'cosine'

        if self.neighbors_preprocess is not None:
            neighbors_X = self.neighbors_preprocess(neighbors_X)

        closest, furthest = self._get_near_and_far_pairs_mem_efficient_chunks(neighbors_X)

        samples = []
        paired = []

        # for semisupervised version
        # assuming y has positive integer class labels
        # and -1 if there is no label
        first_label = []
        second_label = []

        self._print_with_verbosity("building dataset from nearest neighbors graph", 1)
        
        already_paired = set()
        for first, seconds in enumerate(closest):
            represented = 0
            for ind, second in enumerate(seconds[::-1]): # matters if sorted and min_neighbors so closest are last
                if self.snn:
                    if first not in closest[second]:
                        n_left = len(seconds) - ind
                        if n_left > self.min_neighbors - represented:
                            continue

                if self.semisupervised and self.prune_graph:
                    if y[first] != y[second] and y[first] != -1 and y[second] != -1:
                        continue

                represented += 1

                if tuple(sorted([first, second])) not in already_paired and first != second:
                    first_data = X[first]
                    second_data = X[second]
                    stack = torch.stack([first_data, second_data])

                    samples.append(stack)
                    paired.append(1)
                    already_paired.add(tuple(sorted([first, second])))

                    if y is not None:
                        first_label.append(y[first])
                        second_label.append(y[second])
                    else:
                        first_label.append(-1)
                        second_label.append(-1)

        already_paired = set()
        for first, seconds in enumerate(furthest):
            for second in seconds:
                if self.semisupervised and self.prune_graph:
                    if y[first] == y[second] and y[first] != -1 and y[second] != -1:
                        continue

                if tuple(sorted([first, second])) not in already_paired and first != second:
                    first_data = X[first]
                    second_data = X[second]
                    stack = torch.stack([first_data, second_data])

                    samples.append(stack)
                    paired.append(0)
                    already_paired.add(tuple(sorted([first, second])))

                    if y is not None:
                        first_label.append(y[first])
                        second_label.append(y[second])
                    else:
                        first_label.append(-1)
                        second_label.append(-1)

        samples = torch.stack(samples)
        paired = torch.Tensor(np.array(paired)) 
        first_label = torch.Tensor(np.array(first_label))
        second_label = torch.Tensor(np.array(second_label))

        dataset = TensorDataset(samples, paired, first_label.long(), second_label.long())

        return dataset

    def _orthgonality_regularizer(self, x):
        diff = 0
        for i in range(x.shape[1]):
            for j in range(i+1, x.shape[1]):
                diff = diff + torch.abs(F.cosine_similarity(x[:, i], x[:, j], dim = 0))
        diff = diff / ((x.shape[1]*(x.shape[1]-1))/2)

        return diff

    def _train_siamese_one_epoch(self, data_loader):
        epoch_loss = 0
        self.model.train()
        for data, target, first_label, second_label in self._progressbar_with_verbosity(data_loader, 1):
            self.optimizer.zero_grad()

            data = data.to(self.device)
            target = target.to(self.device)
            first_label = first_label.to(self.device)
            second_label = second_label.to(self.device)

            output_1 = self.model(data[:, 0])
            output_2 = self.model(data[:, 1])

            loss = f_loss(output_1, output_2, target, ignore = self.ignore, device = self.device, min_p = self.min_p)

            if self.semisupervised:
                which = first_label != -1
                first_label_pred = self.semisupervised_model.cluster_net(output_1[which])
                loss = loss + F.cross_entropy(first_label_pred, first_label[which])*self.semisupervised_weight
                which = second_label != -1
                second_label_pred = self.semisupervised_model.cluster_net(output_2[which])
                loss = loss + F.cross_entropy(second_label_pred, second_label[which])*self.semisupervised_weight

            if self.l2_penalty > 0:
                loss = loss + self.l2_penalty*torch.mean((torch.norm(output_1, p = 2, dim = 1) + torch.norm(output_2, p = 2, dim = 1)))

            if self.max_correlation is not None:
                diff = (self._orthgonality_regularizer(output_1) + self._orthgonality_regularizer(output_2)) / 2
                loss = loss + torch.max(torch.Tensor([self.max_correlation]).to(self.device), diff) - self.max_correlation

            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()

            del output_1
            del output_2

        self._print_with_verbosity(f"training loss: {epoch_loss / len(data_loader)}", 1)

    def _train_one_epoch(self, model, data_loader, optimizer, crit):
        model.train()
        for data, target in data_loader:
            optimizer.zero_grad()

            data = data.to(self.device)
            target = target.to(self.device)

            pred = model(data)

            loss = crit(pred, target)

            loss.backward()
            optimizer.step()

    def transform(self, X, to_numpy = True, batch_size = 4096, model = None):
        if self.is_tokens:
            X = pad_sequence(X, padding_value = 0, batch_first = True)

        if model is None:
            assert self.best_embedding_net is not None, "no embedding model trained yet!"
            model = self.best_embedding_net
        # embeds the data
        dataset = TensorDataset(X)
        embed_loader = DataLoader(dataset, shuffle = False, batch_size = batch_size)

        embeddings = []
        model.eval()
        with torch.no_grad():
            for data in embed_loader:
                data = data[0].to(self.device)
                embedding = model(data).cpu()
                if to_numpy:
                    embedding = embedding.numpy()
                embeddings.append(embedding)

        if to_numpy:
            embeddings = np.concatenate(embeddings)
            embeddings = embeddings.reshape(len(X), -1)
        else:
            embeddings = torch.cat(embeddings)
            embeddings = embeddings.view(len(X), -1)
    
        return embeddings    

    def _get_exp_dist(self, data_loader):
        # sets self.exp_dist based off mean of means dist between positive pairs
        self.model.eval()

        cumulative_dist = 0
        with torch.no_grad():
            for data, target, first_label, second_label in data_loader:
                data = data.to(self.device)
                target = target.to(self.device)

                should_be_close = target == 1
                if torch.sum(should_be_close) == 0:
                    continue

                output_1 = self.model(data[should_be_close, 0])
                output_2 = self.model(data[should_be_close, 1])

                d = torch.norm(output_1 - output_2, p = 2, dim = 1)

                # get parameters for f distribution. not sure these are right..
                d1 = torch.Tensor([output_1.shape[-1]]).to(self.device)
                d2 = torch.Tensor([1]).to(self.device)

                # compute p-value
                p = reg_betainc(d1*d/(d1*d+d2), d1/2, d2/2)
                # reject null hypothesis
                d = d[p < self.ignore]
                # do means
                cumulative_dist += torch.mean(d).item()

                del output_1
                del output_2
                del should_be_close

        avg_dist = cumulative_dist / len(data_loader)

        self.exp_dist = avg_dist

    def _cluster(self, X):
        # runs spectral clustering based off self.exp_dist as Gaussian kernel bandwidth
        # sets self.n_clusters and returns cluster_assignments
        X = X.reshape(X.shape[0], -1)

        n = min(X.shape[0], self.cluster_subsample_n)

        inds = np.random.choice(X.shape[0], n, replace = False)
        D = pairwise_distances(X[inds], n_jobs = -1, metric = 'euclidean')

        sigma = (self.exp_dist*self.gamma)**2
        A = np.exp(-D**2 / sigma) # known bug: sigma should be larger because subsampling

        sums = A.sum(axis = 1)
        D = np.diag(sums)
        L = D - A

        vals, vecs = eigh(L, turbo = True)#, eigvals = [0, int(X.shape[0]**.5)]) # assuming only sqrt possible clusters
        # print(vals)

        n_zeros = np.sum(vals <= self.zero_cutoff)
        self._print_with_verbosity(f"found {n_zeros} candidate clusters", 3)
        self._print_with_verbosity(f"running k-means..", 3)
        k = KMeans(n_zeros, n_init = 100)
        init_clusters = k.fit_predict(vecs[:, :n_zeros])
        # print(np.unique(init_clusters))

        self._print_with_verbosity(f"applying KNN filter..", 3)
        n_neighbors = int(2*np.log2(X.shape[0]))
        clusters = KNeighborsClassifier(n_neighbors).fit(X[inds], init_clusters).predict(X)
        clusters = change_cluster_labels_to_sequential(clusters)

        self.n_clusters = np.unique(clusters).shape[0]

        if self.update_zero_cutoff:
            self._update_zero_cutoff(vals)

        return clusters

    def _update_zero_cutoff(self, eign):
        # slowly decrease zero cutoff for spectral clustering calculation
        # reduces noise in clustering
        # using a separate function because might make this more complex in the future
        # this cutoff is just the first eigenvalue that DID NOT correspond to a cluster
        # new_zero_cutoff = min(self.zero_cutoff, eign[self.n_clusters])
        # new_zero_cutoff = max(new_zero_cutoff, 1e-8) # don't want negative, that's just numerical error
        new_zero_cutoff = max(10*eign[self.n_clusters], self.minimum_zero_cutoff)

        if new_zero_cutoff != self.zero_cutoff:
            self._print_with_verbosity(f"updating eigenvalue zero cutoff from {self.zero_cutoff} to {new_zero_cutoff}", 3)
            self.zero_cutoff = new_zero_cutoff

    def predict(self, X, model = None, return_embedding = False):
        if model is None:
            assert self.best_full_net is not None, "have not trained a prediction network yet!"
            model = self.best_full_net # if not self.final_model_trained else self.final_model

        dataset = TensorDataset(X)
        data_loader = DataLoader(dataset, batch_size = 4096, shuffle = False)
        preds = []
        embeddings = []

        model = model.to(self.device)
        model.eval()
        with torch.no_grad():
            for data in self._progressbar_with_verbosity(data_loader, 3):
                data = data[0].to(self.device)

                if return_embedding:
                    embedding = model.embed_net(data)
                    _, pred = torch.max(model.cluster_net(embedding), 1)
                    embeddings.extend(embedding.cpu().numpy())
                else:
                    _, pred = torch.max(model(data), 1)

                preds.extend(pred.cpu().numpy())

        embeddings = np.array(embeddings)
        preds = np.array(preds)

        if return_embedding:
            return preds, embeddings
        else:
            return preds

    def _build_cluster_subnet(self, X, transformed, clusters):
        # creates clustering subnet and updates best model
        # sets self.cluster_subnet

        self._print_with_verbosity("training cluster subnet to predict spectral labels", 2)

        cluster_counts = torch.zeros(self.n_clusters).float().to(self.device)
        for cluster_assignment in clusters:
            cluster_counts[cluster_assignment] += 1
        cluster_weights = len(clusters)/self.n_clusters/cluster_counts

        dataset = TensorDataset(torch.Tensor(transformed), torch.Tensor(clusters).long())
        data_loader = DataLoader(dataset, shuffle = True, batch_size = self.batch_size)

        cluster_subnet = ClusterNet(transformed.shape[-1], self.n_clusters, p = self.cluster_subnet_dropout_p)
        cluster_subnet_optimizer = optim.Adam(cluster_subnet.parameters())
        cluster_subnet_crit = nn.CrossEntropyLoss(weight = cluster_weights)
        cluster_subnet.train()
        cluster_subnet = cluster_subnet.to(self.device)

        for i in self._progressbar_with_verbosity(range(self.cluster_subnet_training_epochs), 2):
            self._train_one_epoch(cluster_subnet, data_loader, cluster_subnet_optimizer, cluster_subnet_crit)

        # now fine-tune the whole pipeline
        dataset = TensorDataset(X, torch.Tensor(clusters).long())
        data_loader = DataLoader(dataset, shuffle = True, batch_size = self.batch_size)

        full_net = FullNet(copy.deepcopy(self.model), cluster_subnet)
        full_net_optimizer = optim.Adam(full_net.parameters(), lr = 1e-4)
        full_net_crit = cluster_subnet_crit
        full_net.train()
        full_net = full_net.to(self.device)

        if self.fine_tune_end_to_end:
            self._print_with_verbosity("fine-tuning whole end-to-end network", 2)

            for i in self._progressbar_with_verbosity(range(self.fine_tune_epochs), 2):
                self._train_one_epoch(full_net, data_loader, full_net_optimizer, full_net_crit)

        preds = self.predict(X, model = full_net, return_embedding = False)

        # new_transformed = self.transform(X, model = full_net.embed_net)

        # delta_mi = silhouette_score(new_transformed, preds)
        # delta_mi = adjusted_mutual_info_score(preds, clusters)
        # preds_entropy = get_entropy(list(Counter(preds).values()))
        # embedding_score = self._test_label_fit(embedding, preds) # maybe this should use transformed instead?
        # sample_score = self._test_label_fit(X, preds) # compensate for random
        # delta_mi = preds_entropy * embedding_score * sample_score # average ability to pattern-match times information content of labels
        # random_labels_score = self._test_label_fit(X, np.random.randint(0, self.n_clusters, X.shape[0]))
        # self._print_with_verbosity(f"this delta mi: {delta_mi}, from embedding: {embedding_score}, from original data: {sample_score}, entropy: {preds_entropy}", 1)
        # if delta_mi > self.best_delta_mi:
            # self._print_with_verbosity(f"found new best delta mi", 1)

        # we're just going to take the most recent epoch
        # this is reasonable because of the new changes to the F-distribution loss
        # self.best_delta_mi = delta_mi
        self.best_full_net = full_net
        self.best_n_clusters = self.n_clusters
        self.best_embedding_net = copy.deepcopy(self.model)

        return preds

    def _test_label_fit(self, X, y, test_proportion = .2):
        # trains a simple classifier to predict the labels from the dataset
        # if the labels are a good fit, this score should go up
        if np.unique(y).shape[0] == 1:
            # single label dataset has accuracy 1
            return 1

        X = X.reshape(X.shape[0], -1)
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_proportion)
        # c = naive_bayes.ComplementNB()
        # X -= X.min() # get rid of negative values for CNB
        c = self.simple_classifier
        with nostdout():
            # score = cross_val_score(c, X, y, n_jobs = -1).mean() # pretty sure this uses reproducable random state by default
            # since only training internally, only test internally for this
            score = np.exp(-log_loss(y, c.fit(X, y).predict_proba(X)))

        return score

    def fit(self, X, y = None, y_for_verification = None, plot = False):
        # assert not self.semisupervised, "semisupervised not supported yet"

        self.best_delta_mi = -1
        self.best_full_net = None
        self.best_embedding_net = None
        # self.final_model = None
        # self.final_model_trained = False
        self.best_n_clusters = 1
        self.zero_cutoff = self.initial_zero_cutoff
        self.exp_dist = 0

        if self.random_seed is not None:
            np.random.seed(self.random_seed)

        use_y_to_verify_performance = y_for_verification is not None
        self.semisupervised = self.semisupervised and y is not None

        if self.semisupervised and self.semisupervised_weight is None:
            self.semisupervised_weight = np.sum(y != -1) / y.shape[0]

        if self.semisupervised:
            n_classes = np.unique(y[y != -1]).shape[0] # because of the -1 

        if use_y_to_verify_performance:
            verify_n_classes = np.unique(y_for_verification).shape[0]
            self._print_with_verbosity(f"number of classes in verification set: {verify_n_classes}", 3)

        if self.model == "auto":
            self.model = self._select_model(X)

        if self.is_tokens:
            X = pad_sequence(X, padding_value = 0, batch_first = True)

        if type(X) is not torch.Tensor:
            X = torch.Tensor(X)

        self.device = torch.device("cuda") if (torch.cuda.is_available() and self.use_gpu) else torch.device("cpu")
        if self.device.type == "cpu":
            self._print_with_verbosity("WARNING: using CPU, may be very slow", 0, strict = True)

        self._print_with_verbosity(f"using torch device {self.device}", 1)

        self._print_with_verbosity("building dataset", 1)

        dataset = self._build_dataset(
            X, 
            y = y if self.semisupervised else None, 
        )

        data_loader = DataLoader(dataset, shuffle = True, batch_size = self.batch_size)

        self.model = self.model.to(self.device)

        if self.optimizer_override is None:
            self.optimizer = optim.Adam(self.model.parameters(), lr = self.learning_rate)
        else:
            self.optimizer = self.optimizer_override(self.model.parameters(), lr = self.learning_rate)

        if self.semisupervised:
            label_subnet = ClusterNet(self.n_components, n_classes).to(self.device)
            self.semisupervised_model = FullNet(self.model, label_subnet).to(self.device)
            self.optimizer = optim.Adam(self.semisupervised_model.parameters(), lr = self.learning_rate)

        self._print_with_verbosity("training", 1)

        for i in self._progressbar_with_verbosity(range(self.epochs), 0, strict = True):
            self.model.train()
            self._print_with_verbosity(f"this is epoch {i}", 1)
            self._train_siamese_one_epoch(data_loader)
            self.model.eval()
            transformed = self.transform(X, model = self.model)

            self._get_exp_dist(data_loader)
            self._print_with_verbosity(f"found expected distance between related points as {self.exp_dist}", 3)
            cluster_assignments = self._cluster(transformed)
            self._print_with_verbosity(f"found {self.n_clusters} clusters", 1)

            preds = self._build_cluster_subnet(X, transformed, cluster_assignments)

            if use_y_to_verify_performance:
                nmi_score = normalized_mutual_info_score(cluster_assignments, y_for_verification, 'geometric')
                self._print_with_verbosity(f"NMI of cluster labels with y: {nmi_score}", 2)

                nmi_score = normalized_mutual_info_score(preds, y_for_verification, 'geometric')
                self._print_with_verbosity(f"NMI of network predictions with y: {nmi_score}", 1)

                if self.n_clusters == verify_n_classes:
                    acc_score = get_accuracy(cluster_assignments, y_for_verification)
                    self._print_with_verbosity(f"accuracy of cluster labels: {acc_score}", 2)

                if np.unique(preds).shape[0] == verify_n_classes:
                    acc_score = get_accuracy(preds, y_for_verification)
                    self._print_with_verbosity(f"accuracy of network predictions: {acc_score}", 1)
                else:
                    self._print_with_verbosity(f"number of predicted classes did not match number of clusters so not computing accuracy, correct {verify_n_classes} vs {self.n_clusters}", 2)

            if plot:
                if self.n_components == 2:
                    plot_2d(transformed, cluster_assignments, show = False, no_legend = True)

                    if use_y_to_verify_performance:
                        plot_2d(transformed, y_for_verification, show = False, no_legend = True)

                    plt.show()

                elif self.n_components == 3:
                    plot_3d(transformed, cluster_assignments, show = False)

                    if use_y_to_verify_performance:
                        plot_3d(transformed, y_for_verification, show = False)

                    plt.show()


if __name__ == "__main__":
    from torchvision.datasets import MNIST, USPS, FashionMNIST, CIFAR10
    from torchtext.datasets import AG_NEWS

    n = None
    # semisupervised_proportion = .2

    e = DEN(n_components = 2, internal_dim = 128)

    USPS_data_train = USPS("./", train = True, download = True)
    USPS_data_test = USPS("./", train = False, download = True)
    USPS_data = ConcatDataset([USPS_data_test, USPS_data_train])
    X, y = zip(*USPS_data)

    y_numpy = np.array(y[:n])
    X_numpy = np.array([np.asarray(X[i]) for i in range(n if n is not None else len(X))])
    X = torch.Tensor(X_numpy).unsqueeze(1)

    # which = np.random.choice(len(y_numpy), int((1-semisupervised_proportion)*len(y_numpy)), replace = False)
    # y_for_verification = copy.deepcopy(y_numpy)
    # y_numpy[which] = -1

    # news_train, news_test = AG_NEWS('./', ngrams = 1)
    # X, y = zip(*([item[1], item[0]] for item in news_test))
    # X = X[:n]
    # y = y[:n]
    # y_numpy = np.array(y)
    # y_for_verification = copy.deepcopy(y_numpy)

    # X_numpy = np.load("shekhar_data_pca_40.npy")[:n]
    # y_numpy_strs = np.load("shekhar_labels.npy", allow_pickle = True)[:n]
    # str_to_ind = {name:i for i, name in enumerate(np.unique(y_numpy_strs))}
    # y_numpy = np.array([str_to_ind[name] for name in y_numpy_strs])
    # X = torch.Tensor(X_numpy)
    # which = y_numpy < 16 # to just focus on interesting stuff
    # X = X[which]
    # y_numpy = y_numpy[which]
    y_for_verification = copy.deepcopy(y_numpy)

    e.fit(X, None, y_for_verification = y_for_verification, plot = True)
    # e.save("test_thing.pt")
    # e.load("test_thing.pt")
    # # e.find_differentiating_features(X[0], X)
    # e.summerize_differentiating_features(X)
