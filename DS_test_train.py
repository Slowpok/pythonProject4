import Datasets
import DS_init
import pandas as pd
import torch
import torch.utils.data as data
from torch.utils.data.sampler import WeightedRandomSampler
import numpy as np
from nltk.corpus import stopwords
import DS_dupl_search
import DS_training
import DS_possum_tokenizer
from collections import Counter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# df = pd.read_excel(DS_init.finename, skiprows=1)
df = pd.read_csv(filepath_or_buffer=DS_init.finename,
                         names=['request', 'class_row'], encoding='Windows-1251', delimiter=';', header=1)

stopwords_mass = stopwords.words("russian")
x_mass, y_mass, num_unique_ngramm = DS_possum_tokenizer.possum_tokenizer(df)

duplication_search_model = DS_dupl_search.Search_of_duplicates_test(num_unique_ngramm, "siam-twins")
duplication_search_model.to(device)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(duplication_search_model.parameters(), lr=DS_init.learning_rate)

for x in range(len(df)):
    if x == 1:
        break

    req_x = x_mass[x]
    req_y = y_mass[x]

    list_y = [1 if y == req_y else 0 for y in y_mass]

    x_mass = torch.Tensor(x_mass).to(device)

    y_mass_bin = torch.as_tensor(list_y).to(device)

    counter = Counter(list_y)

    MyDataset = Datasets.MyDataset(x_mass, y_mass_bin)

    weights = [1 / counter.get(y.item()) for x, y in MyDataset]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights))

    train_loader_bin = data.DataLoader(MyDataset, batch_size=DS_init.batch_size, pin_memory=False, sampler=sampler)
    val_loader_bin = data.DataLoader(MyDataset, batch_size=DS_init.batch_size, pin_memory=False, sampler=sampler)

    duplication_search_model = DS_training.training(duplication_search_model, loss_fn, optimizer,
                                                    np.array(req_x), train_loader_bin, val_loader_bin, n_epoch=5)




