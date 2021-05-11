import torchvision.io
from torch.utils.data.dataset import Dataset
import os
import concurrent.futures

DIR_NAME = "all_vids"  # dummy names
ANNO_NAME = "all_anno"
root = os.path.join(os.gcwd(), DIR_NAME)
annotations = os.path.join(os.gcwd(), ANNO_NAME)


class NoodraVid(Dataset):
    """
    Custom dataset that will be used to train vision models for Noorda
    """
    def __init__(self, dir_vids: str, annotations: str, *transformations):
        """
        :param dir_vids: directory of all videos
        :param annotations: path to .txt file where each row is the expected value of the corresponding video
        :param *transformations: some undefined amount of transformation that we would like to submit our dataset to
        """
        self.path = dir_vids
        self.expected_vals = annotations
        self.dataset = self.gather()

    def __getitem__(self, index: int):
        """
        allows torch.data.Dataloader to create training sets
        :param index: location of subscription
        :return: vid tensor, label for sample
        """
        return self.dataset[0]

    def __len__(self):
        """
        Allows for size of dataset to be returned
        :return: size
        """
        pass

    def gather(self):
        """
        Use multiprocessing to gather all image tensors and process them into some dimension
        :return: list of video tensors to be zip()'ed with the annotations
        """
        def sub_ingest(path: str):
            """
            TODO: add pooling to decrease dimensionality
            :param path:
            :return: tensor representing the video
            """
            return torchvision.io.read_video(path)[0]

        path_names = list(os.listdir(self.path))
        with concurrent.futures.ProcessPoolExecutor() as executor:
            tensor_total = executor.map(sub_ingest, path_names)  # list of (tensor, path)
        return tensor_total

