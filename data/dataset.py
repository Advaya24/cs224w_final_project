from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.data import DataLoader

# Download and process data at './dataset/ogbg_molhiv/'
dataset = PygNodePropPredDataset(name = "ogbn-mag", root = 'dataset/')

 
# Get split indices
split_idx = dataset.get_idx_split()

# dataset_dict = dict(dataset[0])
# print(dataset_dict)
# train_set = {prop: d['paper'][split_idx["train"]['paper']] for prop, d in dataset[0]}

for prop, d in dataset[0]:
    print(prop, d)
print(split_idx)