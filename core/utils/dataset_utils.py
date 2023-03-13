import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor, ToPILImage, \
    Normalize, Compose, Resize, CenterCrop
from imageio import imread, imsave
from glob import glob
from os.path import join
from torch.utils.data import Subset, SubsetRandomSampler

class ImagePathDataset(Dataset):
    """
    Modeling Matlab ImageDatastore, using a set of image paths to create the dataset.
    TODO: Support image data without labels.
    """
    def __init__(self, imgfp_vect, scores, img_dim=(227, 227), transform=None):
        self.imgfps = imgfp_vect
        if scores is None:
            self.scores = torch.tensor([0.0] * len(imgfp_vect))
        else:
            self.scores = torch.tensor(scores)
        # self.img_dim = img_dim
        if transform is None:
            self.transform = Compose([ToTensor(),
                                      Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225]),
                                      Resize(img_dim)])
        else:
            print(f"The {img_dim} setting is overwritten by the size in custom transform")
            self.transform = transform

    def __len__(self):
        return len(self.imgfps)

    def __getitem__(self, idx):
        img_path = self.imgfps[idx]
        img = imread(img_path)
        if len(img.shape) == 2:
            img = img[:, :, None]
        if img.shape[2] == 4:
            img = img[:, :, :3]
        elif img.shape[2] == 1:
            img = img.repeat(3, axis=2)
        imgtsr = self.transform(img)
        score = self.scores[idx]
        return imgtsr, score


# ImageNet Validation Dataset
def create_imagenet_valid_dataset(imgpix=256, normalize=True, rootdir=r"E:\Datasets\imagenet-valid"):
    # Labels for the imagenet validation set.
    #   ILSVRC2012_validation_ground_truth.txt
    RGB_mean = torch.tensor([0.485, 0.456, 0.406]) #.view(1,-1,1,1).cuda()
    RGB_std  = torch.tensor([0.229, 0.224, 0.225]) #.view(1,-1,1,1).cuda()
    preprocess = Compose([ToTensor(),
                          Resize(imgpix, ),
                          CenterCrop((imgpix, imgpix), ),
                          Normalize(RGB_mean, RGB_std) if normalize else lambda x: x
                          ])
    dataset = ImageFolder(rootdir, transform=preprocess)
    return dataset


def Invariance_dataset():
    img_src = r"N:\Stimuli\Invariance\Project_Manifold\ready"
    imglist = sorted(glob(join(img_src, "*.jpg")))
    return ImagePathDataset(imglist, None)