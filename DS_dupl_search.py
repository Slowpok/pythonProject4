import DS_init
import torch.nn as nn
import torch
import torch.nn.functional as F

class Search_of_duplicates(nn.Module):
    def __init__(self, unique_words, name):
        super(Search_of_duplicates, self).__init__()
        self.inplanes = 128
        self.name = name
        self.layer0 = nn.Embedding(unique_words, embedding_dim=64, max_norm=DS_init.num_of_ngramm_in_string) # -> batch*50*500*64
        # -> batch*50*32*500
        # self.layer1 = nn.BatchNorm1d(DS_init.num_of_ngramm_in_string)
        self.layer2 = nn.Conv2d(DS_init.num_of_ngramm_in_string, 128, kernel_size=5, stride=1, padding=2) # -> batch*50*64*128
        self.layer3 = nn.Conv2d(128, 64, kernel_size=5, stride=1, padding=2)  # -> batch*50*64*64
        self.layer4 = nn.ReLU()
        self.layer5 = nn.MaxPool2d(kernel_size=64, stride=2) # -> batch*50*64*1 -> batch*50*64
        self.flat = nn.Flatten()
        self.drop_out = nn.Dropout(0.25)
        self.fc1 = nn.Linear(50*64, 50*64)
        self.fc2 = nn.Linear(50*64, 50*64)

    def forward(self, request, X_batch):
        print("beg emb ", request.size())
        outx = self.layer0(request.long())
        print("layer0 ", outx.size())
        outx = outx.view(outx.shape[0], outx.shape[2], outx.shape[1], outx.shape[3])
        print("reshape ", outx.size())
        # outx = outx.view(outx.shape[0], outx.shape[1], outx.shape[3], outx.shape[2])
        # print("reshape ", outx.size())
        # outx = self.layer1(outx)
        # print("layer1 ", outx.size())
        outx = self.layer2(outx)
        print("layer2 ", outx.size())
        outx = self.layer3(outx)
        print("layer3 ", outx.size())
        outx = self.layer4(outx)
        print("layer4 ", outx.size())
        outx = outx.view(outx.shape[0], outx.shape[2], outx.shape[1], outx.shape[3])
        outx = self.layer5(outx)
        print("layer5 ", outx.size())
        outx = self.flat(outx)
        print("flat ", outx.size())
        outx = self.drop_out(outx)
        print("layer drop", outx.size())
        outx = self.fc1(outx)
        print("layer6 fc1", outx.size())
        outx = self.fc2(outx)
        print("layer6 fc2", outx.size())

        # massiv textov
        print("beg emb ", X_batch.size())
        outy = self.layer0(X_batch.long())
        print("layer0 ", outy.size())
        outy = outy.view(outy.shape[0], outy.shape[2], outy.shape[1], outy.shape[3])
        print("reshape ", outy.size())
        # outy = outy.view(outy.shape[0], outy.shape[1], outy.shape[4], outy.shape[3])
        # print("reshape ", outy.size())
        # outy = self.layer1(outy)
        # print("layer1 ", outy.size())
        outy = self.layer2(outy)
        print("layer2 ", outy.size())
        outy = self.layer3(outy)
        print("layer3 ", outy.size())
        outy = self.layer4(outy)
        print("layer4 ", outy.size())
        outy = self.layer5(outy)
        print("layer5 ", outy.size())
        outy = self.flat(outy)
        print("flat ", outy.size())
        outy = self.drop_out(outy)
        print("layer drop", outy.size())
        outy = self.fc1(outy)
        print("layer6 fc1", outy.size())
        outy = self.fc2(outy)
        print("layer6 fc2", outy.size())

        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        output = cos(outx, outy)

        return output

class Search_of_duplicates_test(nn.Module):
    def __init__(self, unique_words, name):
        super(Search_of_duplicates_test, self).__init__()
        self.inplanes = 128
        self.name = name
        emb_dim = 64

        self.layer0 = nn.Embedding(unique_words, embedding_dim=emb_dim, max_norm=DS_init.num_of_ngramm_in_string) # -> batch*50*500*64
        # -> batch*50*32000
        self.layer1 = nn.BatchNorm1d(DS_init.num_of_words)
        self.layer2 = nn.Conv1d(DS_init.num_of_ngramm_in_string * emb_dim, 1500, kernel_size=5, stride=1, padding=2) # -> batch*50*1500
        self.layer3 = nn.ReLU()
        self.layer4 = nn.Conv1d(1500, 128, kernel_size=5, stride=1, padding=2)  # -> batch*50*128
        self.layer5 = nn.ReLU()
        self.layer6 = nn.MaxPool1d(kernel_size=50, stride=2)
        self.flat = nn.Flatten()
        self.drop_out = nn.Dropout(0.25)
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 128)

    def forward(self, request, X_batch):
        print("beg emb ", request.size())
        outx = self.layer0(request.long())
        print("layer0 ", outx.size())
        outx = outx.view(outx.shape[0], outx.shape[1], outx.shape[2] * outx.shape[3])
        print("reshape ", outx.size())
        outx = self.layer1(outx)
        print("layer1 ", outx.size())
        outx = outx.view(outx.shape[0], outx.shape[2], outx.shape[1])
        print("reshape ", outx.size())
        outx = self.layer2(outx)
        print("layer2 ", outx.size())
        outx = self.layer3(outx)
        print("layer3 ", outx.size())
        outx = self.layer4(outx)
        print("layer4 ", outx.size())
        outx = self.layer5(outx)
        print("layer5 ", outx.size())
        outx = self.layer6(outx)
        print("layer6 ", outx.size())
        outx = self.flat(outx)
        print("flat ", outx.size())
        outx = self.drop_out(outx)
        print("layer drop", outx.size())
        outx = self.fc1(outx)
        print("layer7 fc1", outx.size())
        outx = self.fc2(outx)
        print("layer8 fc2", outx.size())

        # massiv textov
        print("beg emb ", X_batch.size())
        outy = self.layer0(X_batch.long())
        print("layer0 ", outy.size())
        outy = outy.view(outy.shape[0], outy.shape[1], outy.shape[2] * outy.shape[3])
        print("reshape ", outy.size())
        outy = self.layer1(outy)
        print("layer1 ", outy.size())
        outy = outy.view(outy.shape[0], outy.shape[2], outy.shape[1])
        print("reshape ", outy.size())
        outy = self.layer2(outy)
        print("layer2 ", outy.size())
        outy = self.layer3(outy)
        print("layer3 ", outy.size())
        outy = self.layer4(outy)
        print("layer4 ", outy.size())
        outy = self.layer5(outy)
        print("layer5 ", outy.size())
        outy = self.layer6(outy)
        print("layer6 ", outy.size())
        outy = self.flat(outy)
        print("flat ", outy.size())
        outy = self.drop_out(outy)
        print("layer drop", outy.size())
        outy = self.fc1(outy)
        print("layer7 fc1", outy.size())
        outy = self.fc2(outy)
        print("layer8 fc2", outy.size())

        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        return cos(outx, outy)

