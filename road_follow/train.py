from pickletools import optimize
from statistics import mode
import torch
from torch.nn import *
from tqdm import tqdm
import PIL
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import numpy as np
import glob
import torchvision
from torchvision.transforms import *
from torch.utils.tensorboard import SummaryWriter

os.chdir(os.path.join(os.getcwd(), r"jetracer/train"))

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using: {device}")

class JetRacerDataset(Dataset):
    def __init__(self, imPaths, random_hflip, imT):
        self.imPaths = imPaths
        self.imT = imT
        self.random_hflip = random_hflip

    def __len__(self):
        return len(self.imPaths)

    def __getitem__(self, idx):
        #get image path and label
        imgPath = self.imPaths[idx]
        x, y = self._parse(imgPath)

        #open image
        img = Image.open(imgPath)
        w, h = img.size

        #org
        # x = ((2.0*x)/w)-1
        # x = -x

        x = (float(x)/10000)-2.0
        y = (float(y)/10000)-2.0
        
        
        #transformations
        if self.random_hflip and float(np.random.random(1)) > 0.5:
            img = img.transpose(PIL.Image.FLIP_LEFT_RIGHT)
            x = -x

        if self.imT: #imT must not include FlipHorizontal because x has to be transformed as well in this case
            img = self.imT(img)

        return img, torch.tensor([x, y])

    def _parse(self, path):
        basename = os.path.basename(path)
        x, y, *_ = basename.split('_')
        return int(x), int(y)


class JetRacerModel(Module):
    def __init__(self):
        super(JetRacerModel, self).__init__()

        self.backbone = torchvision.models.resnet18(pretrained=True)
        self.backbone.fc = torch.nn.Linear(512, 2)

    def forward(self, images):
        prediction = self.backbone(images)
        return prediction


def train_loop(model: Module, loss_fn, trainLoader: DataLoader, optimizer: torch.optim.Adam):
    trainLoss = 0.0
    model.train()

    for data in tqdm(trainLoader):
        images, labels = data
        images, labels = images.to(device), labels.to(device)

        preds = model(images)
        loss = loss_fn(preds, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        trainLoss += loss.item()

    return trainLoss / len(trainLoader)

def eval_loop(model: Module, loss_fn, validLoader: DataLoader):
    validLoss = 0.0
    model.eval()

    with torch.no_grad():
        for data in tqdm(validLoader):
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            preds = model(images)
            loss = loss_fn(preds, labels)

            validLoss += loss.item()

    return validLoss / len(validLoader)


if __name__ == "__main__":
    tbWriter = SummaryWriter()

    #transformations
    augs = Compose([
        ColorJitter(0.2, 0.2, 0.2, 0.2),
        Resize((224, 224)),
        ToTensor(),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    #datasets
    imFiles = glob.glob("data/**/*.jpg")
    imFilesTr, imFilesVal = train_test_split(imFiles, test_size=0.20, random_state=69)
    trainset = JetRacerDataset(imFilesTr, False, augs)
    validset = JetRacerDataset(imFilesVal, False, augs)

    trainLoader = DataLoader(trainset, batch_size=64, shuffle=True)
    validLoader = DataLoader(validset, batch_size=64, shuffle=False)
            
    model = JetRacerModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = MSELoss()

    bestValidLoss = np.Inf
    for e in range(0, 100):
        avgTrainLoss = train_loop(model, loss_fn, trainLoader, optimizer)
        avgValidLoss = eval_loop(model, loss_fn, validLoader)

        if avgValidLoss < bestValidLoss:
            torch.save(model, "model/" + str(e) + "-jetRacer.pth")
            bestValidLoss = avgValidLoss

        tbWriter.add_scalar('Loss/train', avgTrainLoss, e)
        tbWriter.add_scalar('Loss/valid', avgValidLoss, e)
        print(f"Epoch: {e}, avgTrainLoss: {avgTrainLoss}")
        print(f"Epoch: {e}, avgValidLoss: {avgValidLoss}")


