import datetime
import functools
import random

from tqdm import tqdm

import torch
from torch.utils import data
from src import fusion_model, va_dataset

losses = {
    "nll": torch.nn.NLLLoss(),
    "cross-entropy": torch.nn.CrossEntropyLoss()
}

optimisers = {
    "sgd": torch.optim.SGD,
    "adam": torch.optim.Adam
}

def train(mdl, train_ldr, val_ldr, criterion, optimiser, n_epochs=100, v_every=200, eval_every=5):
    v_every = min(len(train_ldr),v_every)
    target = 20
    for e in range(n_epochs):
        print(f"Epoch {e+1}")
        rloss = 0.0
        racc = 0.0
        for batch_idx, batch_data in tqdm(enumerate(train_ldr),total=len(train_ldr)):
            batch_input, batch_lbls = batch_data
            optimiser.zero_grad()

            out = mdl(batch_input)
            _, pred = torch.max(out,dim=1)

            acc = torch.eq(pred, batch_lbls).float().mean()
            loss = criterion(out, batch_lbls)
            loss.backward()

            optimiser.step()

            rloss += loss.item()
            racc += acc.item() * 100

            if batch_idx % v_every == v_every - 1:
                print(f"Epoch {e+1}: Batch {batch_idx+1}: Average Loss at {rloss/v_every:.3f}, Average Accuracy at {(racc/v_every):.3f}")
                rloss = 0.0
                racc = 0.0
        if e % eval_every == eval_every - 1:
            print("Evaluating...")
            target = eval(mdl,val_ldr,criterion,v_every, target=target)
def eval(mdl, dataldr, criterion, v_every=200, target=20):
    mdl.eval()
    rloss = 0.0
    racc = 0.0
    v_every = min(len(val_ldr),v_every)
    for batch_idx, batch_data in tqdm(enumerate(dataldr),total=len(val_ldr)):
        batch_input, batch_lbls = batch_data

        out = mdl(batch_input)
        _, pred = torch.max(out, dim=1)

        acc = torch.eq(pred, batch_lbls).float().mean()
        loss = criterion(out, batch_lbls)


        rloss += loss.item()
        racc += acc.item() * 100

        if (batch_idx % v_every == v_every - 1):
            rloss /= v_every
            racc /= v_every
            print(f"Batch {batch_idx + 1}: Average Loss at {rloss:.3f}, Average Accuracy at {(racc):.3f}")
            if rloss / target < 0.8 and racc > 50:
                target = rloss
                save_mdl(mdl,
                         f"results/embedding_{datetime.datetime.now().strftime('%Y-%m-%d-%H:%M')}_{racc:.3f}.pth")
            rloss = 0.0
            racc = 0.0
    mdl.train()
    return target

def build_mdl(nclasses, embedding=True, weights="weights.pt"):
    mdl = fusion_model.FusionModel(nclasses,embedding,train=True)
    if weights:
        load = torch.load(weights)
        if "state_dict" in load.keys():
            sd = load["state_dict"]
        else:
            sd = load
        mdl.load_state_dict(sd)
    return mdl
def build_dataset_and_loader(dir, subset="", split=0.7, batch_size=128, nworkers=4):
    dataset = va_dataset.EmbeddingDataset(dir,activities=subset)
    idcs = list(range(len(dataset)))
    random.shuffle(idcs)
    train_loader = None
    val_loader = None
    if split:
        train = int(split*len(idcs))
        train_idcs, val_idcs = idcs[:train], idcs[train:]
        val_sampler = data.sampler.SubsetRandomSampler(val_idcs)
    else:
        train_idcs = idcs
    train_sampler = data.sampler.SubsetRandomSampler(train_idcs)
    train_loader = data.DataLoader(dataset,batch_size=batch_size,num_workers=nworkers,pin_memory=True, sampler=train_sampler)
    val_loader = data.DataLoader(dataset,batch_size=batch_size,num_workers=nworkers,pin_memory=True, sampler=val_sampler)
    return train_loader, val_loader

def build_loss(parameters, criterion="nll", optim="sgd", lr=0.001, momentum=0.0, eps=0, weight_decay=0):
    loss = losses[criterion]
    optimiser = functools.partial(optimisers[optim])
    if momentum:
        optimiser = functools.partial(optimiser, momentum=momentum)
    if eps:
        optimiser = functools.partial(optimiser, eps=eps)
    if weight_decay:
        optimiser = functools.partial(optimiser, weight_decay=weight_decay)

    optimiser = optimiser(parameters, lr=lr)
    return loss, optimiser

def save_mdl(mdl,path):
    print(f"Saving to {path}")
    torch.save(mdl.state_dict(), path)

if __name__ == "__main__":
    # mdl = build_mdl(51,weights="fusion_init.pth")
    mdl = build_mdl(51,weights="results/embedding_2019-09-19-21:31_92.576.pth")
    params = [{"params": mdl.classifier.parameters(), "lr": 0.001}] #For embedding (freezes feature extractors)
    train_ldr, val_ldr = build_dataset_and_loader("/media/datasets/VA/UCF-101/",subset="audio_classes.txt",nworkers=4)
    loss, optim = build_loss(params,criterion="cross-entropy", optim="adam",eps=1e-3)
    train(mdl,train_ldr,val_ldr,loss,optim, n_epochs=1000)
    print("done")
