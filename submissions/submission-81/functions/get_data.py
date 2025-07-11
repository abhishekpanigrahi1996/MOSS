from functions.utils import *
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


def data_n_loaders(dataset_name, batch_size, return_data=True, \
                   data_path=None, standardise_data=False, return_transform=False):
    """
    Creates mini-batches of data and returns dataloaders 
    (optionally also returns training, test datasets)
    dataset_name: (str) name of the dataset to load
    batch_size: (int) size of mini-batches
    return_data: (bool) whether to return the training and test datasets
    data_path: (str) path to the dataset
    standardise_data: (bool) whether to standardise the data
    return_transform: (bool) whether to return the mean and scaling factor used for standardisation
    """
    if data_path is None:
        path = 'data'
    else:
        path = data_path
    torch.manual_seed(0)

    if dataset_name=='2dtoy':
        dataname = 'hetero2dfeatures'
    else:
        dataname = dataset_name
    training_data = HeteroToyDataset(root=path, train=True, dataname=dataname)
    test_data = HeteroToyDataset(root=path, train=False, dataname=dataname)   
    
    # standardise data to mean 0 and variance 1 over the training data
    if standardise_data:
        train_mean = torch.mean(training_data.data, dim=0, keepdims=True)
        training_data.data = training_data.data-train_mean
        test_data.data = test_data.data - train_mean #same transformation on test
        data_dim = len(training_data.data[0,:])
        expected_sq_norms = torch.mean(torch.sum(torch.square(training_data.data), \
            dim=-1), dim=0)
        scaling = torch.sqrt(data_dim*1/expected_sq_norms)
        training_data.data = scaling*training_data.data
        test_data.data = scaling*test_data.data #same scaling applied to test data
    
    #create dataloaders
    g_tr = torch.Generator()
    g_tr.manual_seed(0)
    g_te = torch.Generator()
    g_te.manual_seed(0)
    train_dataloader = DataLoader(training_data, batch_size=batch_size, \
                                  worker_init_fn=seed_worker, generator=g_tr)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, \
                                 worker_init_fn=seed_worker, generator=g_te)
    results = [train_dataloader, test_dataloader]

    if return_data:
        results.extend([training_data, test_data])
    if return_transform:
        results.extend([train_mean, scaling])
    return tuple(results)


class HeteroToyDataset(Dataset):
    """
    Dataset of clusters in a N dimensional space
    """

    def __init__(self, root, dim=2, train=True, type=None, dataname=None):
        self.root = root
        self.train = train #train data or test
        self.dim = dim

        #load the dataset
        if dataname is None:
            dataname = f'hetero{dim}dfeatures'
        if train:
            filename = f"traindata.pt"
        else:
            filename = f"testdata.pt"
        datapath = root+f'/{dataname}/'+filename
        file = torch.load(datapath)

        #save data, labels and true features in class
        self.data = file['data']
        self.labels = file['labels']
        self.truefeatures = file['truefeatures']
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx, :], self.labels[idx]


def get_semicircle(N, center, radius, flip_y=False, noise=True, noise_level=0.1):
    """
    Generates a semicircle of points in 2D space
    N: (int) number of points
    center: (list) center of the semicircle
    radius: (float) radius of the semicircle
    flip_y: (bool) whether to flip the y-coordinates
    noise: (bool) whether to add noise to the points- redundant parameter, not used
    noise_level: (float) level of noise to add
    """
    # generate x coordinates
    x1 = 2*radius*(torch.rand((N,))-0.5)+center[0]
    
    noise_y = noise_level*radius*torch.rand((N))

    # generate y coordinates on appropriate side
    if flip_y:
        y1 = -1*torch.sqrt(radius**2 - (x1-center[0])**2)+center[1] + noise_y
    else:            
        y1 = torch.sqrt(radius**2 - (x1-center[0])**2)+center[1] + noise_y
    
    data = torch.stack((x1, y1),dim=-1)
    return data
