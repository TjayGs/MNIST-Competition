from src.MNISTDataset import MNISTDataset
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
import src.yamlConfigConstants as constant

train_file_path = './resources/datasets/digit-recognizer/digit-recognizer/train.csv'
test_file_path = './resources/datasets/digit-recognizer/digit-recognizer/test.csv'

transform = transforms.Compose([])


def get_train_and_validation_dataset():
    """
    method will create the train and validation dataset. It will first of all create the train dataset
    and after that it will create the validation dataset out of the trainings dataset by extracting 20 percent
    out of it
    :return: train_dataset, validation_dataset
    """
    train_dataset = MNISTDataset(train_file_path, with_label=True)
    train_dataset_size = int(0.8 * train_dataset.__len__())  # TODO make size configureable
    validation_dataset_size = train_dataset.__len__() - train_dataset_size

    return random_split(train_dataset, [train_dataset_size, validation_dataset_size])


def dataloader_creation_phase(yaml_config):
    """
    Method will create all necessary dataset (train, validation, test)
    Could apply transformations on train dataset
    :param yaml_config: the overall yaml_config data
    :return: a dict with train, validation and test as key and the dataset behind it
    """
    train_dataset, validation_dataset = get_train_and_validation_dataset()
    train_dataset.dataset.transform = transform
    train_dataloader = DataLoader(train_dataset, batch_size=yaml_config[constant.BATCH_SIZE], shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=yaml_config[constant.BATCH_SIZE_TEST],
                                       shuffle=True)

    test_dataset = MNISTDataset(test_file_path, with_label=False)
    test_dataloader = DataLoader(test_dataset, batch_size=yaml_config[constant.BATCH_SIZE_TEST])

    return {'test': test_dataloader, 'validation': validation_dataloader, 'train': train_dataloader}
