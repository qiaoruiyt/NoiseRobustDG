# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import logging
import os
import torch
from PIL import Image, ImageFile
from torchvision import transforms
from torch.utils.data import TensorDataset, Subset
from torchvision.transforms.functional import rotate
import torchvision.datasets.folder
from torchvision.datasets import MNIST, ImageFolder
import numpy as np
import random
from copy import deepcopy

logger = logging.getLogger('dataset')
ImageFile.LOAD_TRUNCATED_IMAGES = True

DATASETS = [
    # Debug
    "Debug28",
    "Debug224",
    # Small images
    "ColoredMNIST",
    "CMNISTMod",
    "RotatedMNIST",
    # Big images
    "VLCS",
    "PACS",
    "OfficeHome",
    "TerraIncognita",
    "DomainNet",
    "SVIRO",
    # WILDS datasets
    "WILDSCamelyon",
    "WILDSFMoW",
    # WILDS Add-on
    "WILDSWaterbirds",
    "WILDSCelebA",
    "WILDSCivilComments",
    # Toy dataset
    "Toy"
]

def get_dataset_class(dataset_name):
    """Return the dataset class with the given name."""
    # NOTE: modified
    dataset_name = dataset_name.split('&')[0]
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}. Please correct the code above if the dataset \
         has '&' in its name".format(dataset_name))
    return globals()[dataset_name]


def num_environments(dataset_name):
    return len(get_dataset_class(dataset_name).ENVIRONMENTS)


class MultipleDomainDataset:
    N_STEPS = 5001           # Default, subclasses may override
    CHECKPOINT_FREQ = 100    # Default, subclasses may override
    N_WORKERS = 4            # Default, subclasses may override
    ENVIRONMENTS = None      # Subclasses should override
    INPUT_SHAPE = None       # Subclasses should override

    def __getitem__(self, index):
        return self.datasets[index]

    def __len__(self):
        return len(self.datasets)

    def torch_bernoulli_(self, p, size):
        return (torch.rand(size) < p).float()

    def torch_xor_(self, a, b):
        return (a - b).abs()


class Toy(MultipleDomainDataset):
    N_STEPS = 30001           # Default, subclasses may override
    ENVIRONMENTS = ['0', '1', '2']
    N_WORKERS = 0            # Default, subclasses may override

    def __init__(self, root, test_envs, hparams):
        super().__init__()
        self.test_envs = test_envs
        self.hparams = hparams
        self.num_classes = 2
        self.datasets = []
        # generate training data with two environments with different spurious correlation
        self.flip_prob = hparams['flip_prob']
        spu_err = hparams['spu_err']
        d_causal = hparams['d_causal']
        d_spurious = hparams['d_spurious']
        p_correlations = [float(i) for i in spu_err]
        for i, p_correlation in enumerate(p_correlations):
            (x,y,g), n_groups = self.generate_toy_data_complete(n=500, d_causal=d_causal, d_spurious=d_spurious, 
                p_correlation=p_correlation, seed=i, train=True, label_noise=self.flip_prob)
            self.datasets.append(TensorDataset(torch.tensor(x).float(), torch.tensor(y).long()))
        # generate val and test data without the spurious correlation 
        (val_x, val_y, val_g), n_groups = self.generate_toy_data_complete(n=1000, d_causal=d_causal, d_spurious=d_spurious, train=False)
        val_set = TensorDataset(torch.tensor(val_x).float(), torch.tensor(val_y).long())
        self.datasets.append(val_set)
        self.grouped_val_datasets = self.convert_to_dataset_by_groups(val_x, val_y, val_g, n_groups)

        (test_x, test_y, test_g), n_groups = self.generate_toy_data_complete(n=5000, d_causal=d_causal, d_spurious=d_spurious, train=False)
        test_set = TensorDataset(torch.tensor(test_x).float(), torch.tensor(test_y).long())
        self.datasets.append(test_set)
        self.input_shape = test_x.shape[1]
        self.grouped_test_datasets = self.convert_to_dataset_by_groups(test_x, test_y, test_g, n_groups)

    def convert_to_dataset_by_groups(self, x, y, g, n_groups):
        x = torch.tensor(x).float()
        y = torch.tensor(y).long()
        g = torch.tensor(g).long()
        datasets = []
        for i in range(n_groups):
            indices = torch.nonzero(g == i).squeeze()
            datasets.append(TensorDataset(x[indices], y[indices]))
        return datasets

    def generate_toy_data_complete(self, n=1000, d_causal=5, d_spurious=5, p_correlation=0.99,
                                         mean_causal=1, var_causal=0.25, mean_spurious=1, var_spurious=0.25, d_noise=2000,
                                         noise_type='gaussian', mean_noise=0, var_noise=1, 
                                         train=True, label_noise=None, seed=None):
        # group_idx: (y, a)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        groups = [(1,1), (1,-1), (-1,1), (-1,-1)]
        n_groups = len(groups)
        y_list, a_list, x_causal_list, x_spurious_list, g_list = [], [], [], [], []

        processed_n = 0
        for group_idx, (y_value, a_value) in enumerate(groups):
            if group_idx==len(groups)-1:
                n_group = n - processed_n
            else:
                if train:
                    if y_value==a_value:
                        n_group = round(n/2*p_correlation)
                    else:
                        n_group = round(n/2*(1-p_correlation))
                else:
                    n_group = round(n/n_groups)
            processed_n += n_group
            

            y_list.append(np.ones(n_group)*y_value)
            a_list.append(np.ones(n_group)*a_value)
            g_list.append(np.ones(n_group)*group_idx)
            x_causal_list.append(np.random.multivariate_normal(mean=y_value*np.ones(d_causal)*mean_causal,
                                                            cov=np.eye(d_causal)*var_causal,
                                                            size=n_group))
            x_spurious_list.append(np.random.multivariate_normal(mean=a_value*np.ones(d_spurious)*mean_spurious,
                                                                cov=np.eye(d_spurious)*var_spurious,
                                                                size=n_group))


        if d_noise>0:
            x_noise = np.random.multivariate_normal(mean=mean_noise*np.ones(d_noise),
                                                cov=np.eye(d_noise)*var_noise/d_noise,
                                                size=n)
        else:
            x_noise = None

        if label_noise is not None and train:
            # flip binary y with probability label_noise
            y_list = [np.random.choice([-1,1], size=len(y), p=[label_noise, 1-label_noise])*y for y in y_list]
            
        y = np.concatenate(y_list)
        a = np.concatenate(a_list)
        g = np.concatenate(g_list)


        x_causal = np.vstack(x_causal_list)
        x_spurious = np.vstack(x_spurious_list)
        if x_noise is not None:
            x = np.hstack([x_causal, x_spurious, x_noise])
        else:
            x = np.hstack([x_causal, x_spurious])

        y = [0 if i==-1 else i for i in y]
        return (x, y, g), n_groups


class Debug(MultipleDomainDataset):
    def __init__(self, root, test_envs, hparams):
        super().__init__()
        self.input_shape = self.INPUT_SHAPE
        self.num_classes = 2
        self.datasets = []
        for _ in [0, 1, 2]:
            self.datasets.append(
                TensorDataset(
                    torch.randn(16, *self.INPUT_SHAPE),
                    torch.randint(0, self.num_classes, (16,))
                )
            )

class Debug28(Debug):
    INPUT_SHAPE = (3, 28, 28)
    ENVIRONMENTS = ['0', '1', '2']

class Debug224(Debug):
    INPUT_SHAPE = (3, 224, 224)
    ENVIRONMENTS = ['0', '1', '2']


class MultipleEnvironmentMNIST(MultipleDomainDataset):
    def __init__(self, root, environments, dataset_transform, input_shape,
                 num_classes):
        super().__init__()
        if root is None:
            raise ValueError('Data directory not specified!')
        self.environments = environments
        print(self.environments)

        original_dataset_tr = MNIST(root, train=True, download=True)
        original_dataset_te = MNIST(root, train=False, download=True)

        original_images = torch.cat((original_dataset_tr.data,
                                     original_dataset_te.data))

        original_labels = torch.cat((original_dataset_tr.targets,
                                     original_dataset_te.targets))

        shuffle = torch.randperm(len(original_images))

        original_images = original_images[shuffle]
        original_labels = original_labels[shuffle]

        self.datasets = []

        for i in range(len(environments)):
            images = original_images[i::len(environments)]
            labels = original_labels[i::len(environments)]
            self.datasets.append(dataset_transform(images, labels, i))

        self.input_shape = input_shape
        self.num_classes = num_classes


class ColoredMNIST(MultipleEnvironmentMNIST):
    ENVIRONMENTS = ['+90%', '+80%', '-90%']

    def __init__(self, root, test_envs, hparams):
        self.input_shape = (2, 28, 28,)
        self.num_classes = 2
        self.hparams = hparams
        self.test_envs = test_envs
        spu_err = self.hparams['spu_err'] # the error of only using spurious correlated features
        super(ColoredMNIST, self).__init__(root, spu_err,
                                         self.color_dataset, (2, 28, 28,), 2)


    def color_dataset(self, images, labels, env_id):
        # # Subsample 2x for computational convenience
        # images = images.reshape((-1, 28, 28))[:, ::2, ::2]
        # Assign a binary label based on the digit
        environment = self.environments[env_id]
        labels = (labels < 5).float()
        # Flip label with probability 0.25
        labels = self.torch_xor_(labels,
                                 self.torch_bernoulli_(0.25, len(labels)))

        # Assign a color based on the label; flip the color with probability e
        colors = self.torch_xor_(labels,
                                 self.torch_bernoulli_(environment,
                                                       len(labels)))
        images = torch.stack([images, images], dim=1)
        # Apply the color to the image by zeroing out the other color channel
        images[torch.tensor(range(len(images))), (
            1 - colors).long(), :, :] *= 0

        x = images.float().div_(255.0)
        y = labels.view(-1).long()

        return TensorDataset(x, y)


class CMNISTMod(ColoredMNIST):
    # ColoredMNIST customizable label flipping and spurious correlation
    ENVIRONMENTS = ['+90%', '+80%', '-90%']
    def __init__(self, root, test_envs, hparams):
        self.flip_prob = hparams['flip_prob']
        self.spu_err = hparams['spu_err']
        if len(self.spu_err) != len(self.ENVIRONMENTS):
            raise Exception("number of spu_err misspecified, should be either an interger or a list \
                with length equal to len(ENVIRONMENTS)")
        self.ENVIRONMENTS = ['+' + str(i*100) + '%' for i in self.spu_err]
        if isinstance(self.flip_prob, list):
            if len(self.flip_prob) != len(self.ENVIRONMENTS):
                raise Exception("number of flip probs misspecified, should be either an interger or a list \
                    with length equal to len(ENVIRONMENTS)")
        self.data_hard = [] # int, to study noise, hard, simple examples: normal 0, noise 1, hard 2
        self.data_noise = []
        self.hard_datasets = [] 
        self.noisy_datasets = []
        self.hard_dataset_ids = []
        self.noisy_dataset_ids = []
        super(CMNISTMod, self).__init__(root, test_envs, hparams)
    
    def color_dataset(self, images, labels, env_id):
        # # Subsample 2x for computational convenience
        # images = images.reshape((-1, 28, 28))[:, ::2, ::2]
        # Assign a binary label based on the digit
        environment = self.environments[env_id]
        labels = (labels < 5).float() # binarize labels

        data_hard = torch.zeros_like(labels).bool() # track noise and hard examples
        data_noise = torch.zeros_like(labels).bool()
        env_flip_prob = self.flip_prob
        if self.flip_prob is not None:
            if isinstance(self.flip_prob, list):
                env_flip_prob = self.flip_prob[env_id]
            if env_flip_prob > 0 and env_id not in self.test_envs:
                is_flip = self.torch_bernoulli_(env_flip_prob, len(labels))
                data_noise[is_flip == 1] = 1
                labels = self.torch_xor_(labels,
                                 is_flip)

        # Assign a color based on the label; flip the color with probability e
        # logger.info("color {}".format(environment))
        color_assignment = self.torch_bernoulli_(environment,
                                                       len(labels))
        
        data_hard[color_assignment == 1] = 1
        if self.hparams['study_noise']:
            logger.info("environment {}".format(environment))
            logger.info("% noise:{}".format(sum(data_noise)/len(data_noise)))
            logger.info("% hard:{}".format(sum(data_hard)/len(data_hard)))
        colors = self.torch_xor_(labels,
                                 color_assignment)
        images = torch.stack([images, images], dim=1)
        # Apply the color to the image by zeroing out the other color channel
        images[torch.tensor(range(len(images))), (
            1 - colors).long(), :, :] *= 0

        x = images.float().div_(255.0)
        y = labels.view(-1).long()
        ids = torch.arange(y.size(0))
        
        if not self.hparams['include_ids']:
            data_hard_idx = torch.nonzero(data_hard).squeeze()
            data_noise_idx = torch.nonzero(data_noise).squeeze()
            if data_hard_idx.sum() > 0:
                self.hard_datasets.append(TensorDataset(x[data_hard_idx], y[data_hard_idx]))
            if env_flip_prob is not None and env_flip_prob > 0:
                if data_noise_idx.sum() > 0:
                    self.noisy_datasets.append(TensorDataset(x[data_noise_idx], y[data_noise_idx]))

            return TensorDataset(x, y)
        else:
            data_hard_idx = torch.nonzero(data_hard).squeeze()
            data_noise_idx = torch.nonzero(data_noise).squeeze()
            if data_hard_idx.sum() > 0:
                self.hard_datasets.append(TensorDataset(x[data_hard_idx], y[data_hard_idx], data_hard_idx))
            if env_flip_prob is not None and env_flip_prob > 0:
                if data_noise_idx.sum() > 0:
                    self.noisy_datasets.append(TensorDataset(x[data_noise_idx], y[data_noise_idx], data_noise_idx))

            return TensorDataset(x, y, ids)


class RotatedMNIST(MultipleEnvironmentMNIST):
    ENVIRONMENTS = ['0', '15', '30', '45', '60', '75']

    def __init__(self, root, test_envs, hparams):
        super(RotatedMNIST, self).__init__(root, [0, 15, 30, 45, 60, 75],
                                           self.rotate_dataset, (1, 28, 28,), 10)

    def rotate_dataset(self, images, labels, angle):
        rotation = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Lambda(lambda x: rotate(x, angle, fill=(0,),
                interpolation=torchvision.transforms.InterpolationMode.BILINEAR)),
            transforms.ToTensor()])

        x = torch.zeros(len(images), 1, 28, 28)
        for i in range(len(images)):
            x[i] = rotation(images[i])

        y = labels.view(-1)

        return TensorDataset(x, y)


class MultipleEnvironmentImageFolder(MultipleDomainDataset):
    def __init__(self, root, test_envs, augment, hparams):
        super().__init__()
        environments = [f.name for f in os.scandir(root) if f.is_dir()]
        environments = sorted(environments)

        transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        augment_transform = transforms.Compose([
            # transforms.Resize((224,224)),
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.datasets = []
        for i, environment in enumerate(environments):

            if augment and (i not in test_envs):
                env_transform = augment_transform
            else:
                env_transform = transform

            path = os.path.join(root, environment)
            env_dataset = ImageFolder(path,
                transform=env_transform)

            self.datasets.append(env_dataset)        

        self.input_shape = (3, 224, 224,)
        self.num_classes = len(self.datasets[-1].classes)
        
        # Modified for noisy DG
        self.flip_prob = hparams['flip_prob']
        if isinstance(self.flip_prob, list):
            if len(self.flip_prob) != len(self.ENVIRONMENTS):
                raise Exception("ENVIRONMENTs unspecified or the number of flip probs is misspecified, should be either an interger or a list \
                    with length equal to len(ENVIRONMENTS)")
        self.data_noise = []
        self.noisy_datasets = []
        self.hard_datasets = [] 
        self.noisy_dataset_ids = []

        for env_id, environment in enumerate(environments):
            path = os.path.join(root, environment)
            
            env_noise_dataset = ImageFolder(path,
                transform=transform)
            env_noise_dataset.targets = []
            env_noise_dataset.samples = []

            if self.flip_prob is not None and env_id not in test_envs:
                labels = [i for i in self.datasets[env_id].targets]
                data_noise = torch.zeros(len(labels)).bool()
                env_flip_prob = self.flip_prob

                if isinstance(self.flip_prob, list):
                    env_flip_prob = self.flip_prob[env_id]
                if env_flip_prob > 0:
                    is_flip = self.torch_bernoulli_(env_flip_prob, len(labels))
                    data_noise[is_flip == 1] = 1
                # sample indices for flipped samples
                indices = data_noise.nonzero().squeeze().tolist()
                for i in indices:
                    # label = np.random.randint(self.num_classes)
                    label = np.random.randint(self.num_classes-1)
                    if label == self.datasets[env_id].targets[i]:
                        label = self.num_classes-1
                    self.datasets[env_id].targets[i] = label
                    isample = self.datasets[env_id].samples[i] 
                    newsample = (isample[0], label)
                    self.datasets[env_id].samples[i] = newsample

                    # update noise dataset
                    env_noise_dataset.targets.append(label)
                    env_noise_dataset.samples.append(newsample)


                if hparams['study_noise']: # evaluate the noisy data specifically to track the training accuracies
                    print("environment {}".format(environments[env_id]))
                    print("% noise:{}".format(sum(data_noise)/len(data_noise)))

                    # This noisy dataset should somewhat rely on imagefolder for transformation
                    data_noise_idx = torch.nonzero(data_noise).squeeze()
                    if env_flip_prob is not None and env_flip_prob > 0:
                        self.noisy_datasets.append(env_noise_dataset)
        # Mod ends



class VLCS(MultipleEnvironmentImageFolder):
    # N_WORKERS = 8            # Default, subclasses may override
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["C", "L", "S", "V"]
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "VLCS/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)

class PACS(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["A", "C", "P", "S"]
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "PACS/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)

class DomainNet(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 1000
    ENVIRONMENTS = ["clip", "info", "paint", "quick", "real", "sketch"]
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "domain_net/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)

class OfficeHome(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["A", "C", "P", "R"]
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "office_home/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)

class TerraIncognita(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["L100", "L38", "L43", "L46"]
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "terra_incognita/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)

class SVIRO(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["aclass", "escape", "hilux", "i3", "lexus", "tesla", "tiguan", "tucson", "x5", "zoe"]
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "sviro/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)


class WILDSEnvironment:
    def __init__(
            self,
            wilds_dataset,
            metadata_name,
            metadata_value,
            transform=None):
        self.name = metadata_name + "_" + str(metadata_value)

        metadata_index = wilds_dataset.metadata_fields.index(metadata_name)
        metadata_array = wilds_dataset.metadata_array
        subset_indices = torch.where(
            metadata_array[:, metadata_index] == metadata_value)[0]

        self.dataset = wilds_dataset
        self.indices = subset_indices
        self.transform = transform

    def __getitem__(self, i):
        x = self.dataset.get_input(self.indices[i])
        if type(x).__name__ != "Image" and self.dataset.dataset_name != 'civilcomments':
            x = Image.fromarray(x)

        y = self.dataset.y_array[self.indices[i]]
        if self.transform is not None:
            try:
                x = self.transform(x)
            except:
                print(x) 
                raise Exception("transform failed")
        return x, y

    def __len__(self):
        return len(self.indices)


class WILDSDataset(MultipleDomainDataset):
    INPUT_SHAPE = (3, 224, 224)
    def __init__(self, dataset, metadata_name, test_envs, augment, hparams):
        super().__init__()

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # NOTE: updated augment transform to match the DRO and DFR papers 
        augment_transform = transforms.Compose([
            # transforms.Resize((224, 224)),
            transforms.RandomResizedCrop(224,
                scale=(0.7, 1.0),
                ratio=(0.75, 1.3333333333333333),
                interpolation=2),
            transforms.RandomHorizontalFlip(),
            # transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            # transforms.RandomGrayscale(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.dataset_name = dataset.dataset_name
        self.datasets = []
        self.test_group_indices = []
        self.grouped_test_datasets = []
        self.grouped_val_datasets = []

        # NOTE: CelebA and waterbirds have their environments grouped strictly according to spurious correlation
        # e.g., let (x,y) be (feature, label), then group 0 is (0,0), group 1 is (0,1), group 2 is (1,0), group 3 is (1,1)
        if dataset.dataset_name == 'waterbirds' or dataset.dataset_name == 'celebA' or dataset.dataset_name == 'civilcomments':

            # get split indices
            train_indices = dataset._split_array == 0
            val_indices = dataset._split_array == 1
            test_indices = dataset._split_array == 2

            # Abusing the values here. Change the background to be the environment index
            dataset._metadata_fields = ['group', 'y']
            if dataset.dataset_name == 'civilcomments':
                is_toxic = dataset.y_array

                # define identifies to extract the subgroups
                is_male = dataset._metadata_array[:, 0].bool()
                is_white = dataset._metadata_array[:, 7].bool()
                is_black = dataset._metadata_array[:, 6].bool()
                # is_identity = is_male | is_white
                is_identity = is_black

                dataset._metadata_array = is_identity*2 + is_toxic.int()

                transform = initialize_bert_transform(hparams)
                augment_transform = initialize_bert_transform(hparams)
            else:
                dataset._metadata_array = dataset.metadata_array[:, 0]*2 + dataset.metadata_array[:, 1]
            dataset._metadata_array = dataset.metadata_array.view(-1, 1)
            print("number of train", sum(train_indices), "; number of val", sum(val_indices), ";number of test", sum(test_indices))

            BIG_CONSTANT = int(1e6) # a not-so-clean trick to update the array indices in place

            # extract groups from the testset for worst-case tracking
            dataset.metadata_array[test_indices] = dataset.metadata_array[test_indices]+BIG_CONSTANT
            for i, metadata_value in enumerate(
                    self.metadata_values(dataset, metadata_name)):
                if int(metadata_value) >= BIG_CONSTANT:
                    env_transform = transform
                    env_dataset = WILDSEnvironment(
                        dataset, metadata_name, metadata_value, env_transform)
                    self.grouped_test_datasets.append(env_dataset)
            dataset.metadata_array[test_indices] = dataset.metadata_array[test_indices]-BIG_CONSTANT
            assert sum([len(d) for d in self.grouped_test_datasets]) == sum(test_indices)

            # extract groups from the val set for worst-case model selection
            dataset.metadata_array[val_indices] = dataset.metadata_array[val_indices]+BIG_CONSTANT
            for i, metadata_value in enumerate(
                    self.metadata_values(dataset, metadata_name)):
                if int(metadata_value) >= BIG_CONSTANT:
                    env_transform = transform
                    env_dataset = WILDSEnvironment(
                        dataset, metadata_name, metadata_value, env_transform)
                    self.grouped_val_datasets.append(env_dataset)
            dataset.metadata_array[val_indices] = dataset.metadata_array[val_indices]-BIG_CONSTANT
            assert sum([len(d) for d in self.grouped_val_datasets]) == sum(val_indices)

            if hparams['wilds_single']: # ignore group information
                dataset.metadata_array[train_indices] = 0

            elif hparams['wilds_spu_study']: # convert the dataset to be like CMNIST with two environments
                # by default we set the ratio of spurious correlation to be 2:1 between envs, like CMNIST
                dataset.metadata_array[train_indices] = dataset.metadata_array[train_indices]+BIG_CONSTANT
                g1, g4 = dataset.metadata_array == BIG_CONSTANT, dataset.metadata_array == BIG_CONSTANT+3
                g2, g3 = dataset.metadata_array == BIG_CONSTANT+1, dataset.metadata_array == BIG_CONSTANT+2
                if dataset.dataset_name == 'waterbirds' or dataset.dataset_name == 'civilcomments':
                    s1, s4 = int(sum(g1)/2), int(sum(g4)/2) # s1 (waterbird on water), s4 (landbird on land) are major
                    s2, s3 = int(sum(g2)/3), int(sum(g3)/3) # s2 (landbird on water), s3 (waterbird on land) are minor
                elif dataset.dataset_name == 'celebA': # In fact, celebA is also a class imbalanced problem as there are fewer blond images
                    s1, s2, s3 = int(sum(g1)/2), int(sum(g2)/2), int(sum(g3)/2) # s1 (non-blond women, blond women, non-blond man) are major
                    s4 = int(sum(g4)/3) # s4 (blond man) is the minority group
                gs = [g1, g2, g3, g4]
                ss = [s1, s2, s3, s4]

                n0 = sum(dataset.metadata_array == 0)
                n1 = sum(dataset.metadata_array == 1)
                for g, s in zip(gs, ss):
                    indices = torch.nonzero(g).squeeze()
                    dataset._metadata_array[indices[:s]] = 0
                    dataset._metadata_array[indices[s:]] = 1

                    n0_new = sum(dataset._metadata_array == 0)
                    n1_new = sum(dataset._metadata_array == 1)
                    n0 = n0_new
                    n1 = n1_new

            if dataset.metadata_array[train_indices].max() + 1 != min(test_envs):
                raise Exception("The number of train environments does not match the indices for test envs",
                    "set the test_envs to indices starting from the number of train environments")

            if dataset.dataset_name != 'civilcomments':
                # Assign the val and test split to different new groups
                dataset._metadata_array[val_indices] = dataset._metadata_array[train_indices].max() + 1
                dataset._metadata_array[test_indices] = dataset._metadata_array[train_indices].max() + 2

        for i, metadata_value in enumerate(
                self.metadata_values(dataset, metadata_name)):
            if augment and (i not in test_envs):
                env_transform = augment_transform
            else:
                env_transform = transform

            env_dataset = WILDSEnvironment(
                dataset, metadata_name, metadata_value, env_transform)

            self.datasets.append(env_dataset)

        self.input_shape = self.INPUT_SHAPE
        self.num_classes = dataset.n_classes

        # Modified for noisy DG
        self.flip_prob = hparams['flip_prob']
        if self.flip_prob is None or self.flip_prob == 0:
            return
        if isinstance(self.flip_prob, list):
            if len(self.flip_prob) != len(self.datasets):
                raise Exception("ENVIRONMENTs unspecified or the number of flip probs is misspecified, \
                    should be either an interger or a list with length equal to len(ENVIRONMENTS)")
        self.data_noise = []
        self.noisy_datasets = []
        self.hard_datasets = [] 
        self.noisy_dataset_ids = []

        for env_id, metadata_value in enumerate(
                self.metadata_values(dataset, metadata_name)):
                
            env_noise_dataset = WILDSEnvironment(
                dataset, metadata_name, metadata_value, transform)
            
            if self.flip_prob is not None and env_id not in test_envs:
                # only flip the training data
                labels = [i for i in self.datasets[env_id].dataset.y_array]
                data_noise = torch.zeros(len(labels)).bool()
                env_flip_prob = self.flip_prob

                if isinstance(self.flip_prob, list):
                    env_flip_prob = self.flip_prob[env_id]
                if env_flip_prob > 0:
                    is_flip = self.torch_bernoulli_(env_flip_prob, len(labels))
                    data_noise[is_flip == 1] = 1

                # sample indices for flipped samples
                indices = data_noise.nonzero().squeeze().tolist()
                for i in indices:
                    new_label = np.random.randint(self.num_classes-1)
                    if new_label == labels[i]:
                        new_label = self.num_classes-1
                    self.datasets[env_id].dataset.y_array[i] = new_label

                    # update noise dataset
                    env_noise_dataset.dataset.y_array[i] = new_label

                env_noise_dataset.indices = indices

                if hparams['study_noise']:
                    print("environment {}".format(metadata_value))
                    print("% noise:{}".format(sum(data_noise)/len(data_noise)))

                    data_noise_idx = torch.nonzero(data_noise).squeeze()
                    if env_flip_prob is not None and env_flip_prob > 0:
                        self.noisy_datasets.append(env_noise_dataset)
        # Mod ends

    def metadata_values(self, wilds_dataset, metadata_name):
        metadata_index = wilds_dataset.metadata_fields.index(metadata_name)
        metadata_vals = wilds_dataset.metadata_array[:, metadata_index]
        return sorted(list(set(metadata_vals.view(-1).tolist())))

class WILDSCamelyon(WILDSDataset):
    ENVIRONMENTS = [ "hospital_0", "hospital_1", "hospital_2", "hospital_3",
            "hospital_4"]
    def __init__(self, root, test_envs, hparams):
        from wilds.datasets.camelyon17_dataset import Camelyon17Dataset
        dataset = Camelyon17Dataset(root_dir=root)
        super().__init__(
            dataset, "hospital", test_envs, hparams['data_augmentation'], hparams)

class WILDSFMoW(WILDSDataset):
    ENVIRONMENTS = [ "region_0", "region_1", "region_2", "region_3",
            "region_4", "region_5"]
    def __init__(self, root, test_envs, hparams):
        from wilds.datasets.fmow_dataset import FMoWDataset
        dataset = FMoWDataset(root_dir=root)
        super().__init__(
            dataset, "region", test_envs, hparams['data_augmentation'], hparams)

class WILDSWaterbirds(WILDSDataset):
    # Abusing the values of the environments here. 
    # The environments need to be specified depending on the use case.
    # If we follow Waterbirds, the first 4 groups are for training, and group 4/5 are for val/test. 
    # If we follow Waterbirds+ as proposed, the first 2 groups are for training, and group 3/4 are for val/test, and group 4/5 are ignored.
    ENVIRONMENTS = ["group_0", "group_1", "group_2", "group_3", "group_4", "group_5"]
    def __init__(self, root, test_envs, hparams):
        from wilds.datasets.waterbirds_dataset import WaterbirdsDataset
        dataset = WaterbirdsDataset(root_dir=root)
        super().__init__(
            dataset, "group", test_envs, hparams['data_augmentation'], hparams)


class WILDSCelebA(WILDSDataset):
    # Abusing the values of the environments here. 
    # The environments need to be specified depending on the use case.
    # If we follow CelebA, the first 4 groups are for training, and group 4/5 are for val/test. 
    # If we follow CelebA+ as proposed, the first 2 groups are for training, and group 3/4 are for val/test, and group 4/5 are ignored.
    ENVIRONMENTS = ["group_0", "group_1", "group_2", "group_3", "group_4", "group_5"]
    def __init__(self, root, test_envs, hparams):
        from wilds.datasets.celebA_dataset import CelebADataset
        dataset = CelebADataset(root_dir=root)
        super().__init__(
            dataset, "group", test_envs, hparams['data_augmentation'], hparams)


class WILDSCivilComments(WILDSDataset):
    # Abusing the values of the environments here. 
    # The environments need to be specified depending on the use case.
    # If we follow CivilComments, the first 4 groups are for training, and group 4/5 are for val/test. 
    # If we follow CivilComments+ as proposed, the first 2 groups are for training, and group 3/4 are for val/test, and group 4/5 are ignored.
    INPUT_SHAPE = None
    N_STEPS = 5001           # Default, subclasses may override
    CHECKPOINT_FREQ = 300    # Default, subclasses may override
    N_WORKERS = 4            # Default, subclasses may override
    ENVIRONMENTS = ["group_0", "group_1", "group_2", "group_3", "group_4", "group_5"]
    def __init__(self, root, test_envs, hparams):
        from wilds.datasets.civilcomments_dataset import CivilCommentsDataset
        dataset = CivilCommentsDataset(root_dir=root)
        super().__init__(
            dataset, "group", test_envs, hparams['data_augmentation'], hparams)
    
try:
    from transformers import BertTokenizerFast, DistilBertTokenizerFast
except:
    pass

def initialize_bert_transform(hparams):
    def get_bert_tokenizer(model):
        if model == "bert-base-uncased":
            return BertTokenizerFast.from_pretrained(model)
        elif model == "distilbert-base-uncased":
            return DistilBertTokenizerFast.from_pretrained(model)
        else:
            raise ValueError(f"Model: {model} not recognized.")

    assert "bert" in hparams['lm']
    assert hparams['max_seq_length'] is not None

    tokenizer = get_bert_tokenizer(hparams['lm'])

    def transform(text):
        text = str(text)
        tokens = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=hparams['max_seq_length'],
            return_tensors="pt",
        )
        if hparams['lm'] == "bert-base-uncased":
            x = torch.stack(
                (
                    tokens["input_ids"],
                    tokens["attention_mask"],
                    tokens["token_type_ids"],
                ),
                dim=2,
            )
        elif hparams['lm'] == "distilbert-base-uncased":
            x = torch.stack((tokens["input_ids"], tokens["attention_mask"]), dim=2)
        x = torch.squeeze(x, dim=0)  # First shape dim is always 1
        return x

    return transform