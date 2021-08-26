from src.MNISTDataset import MNISTDataset
from torch.utils.data import DataLoader, random_split
import src.yamlConfigConstants as constant

train_file_path = './resources/datasets/digit-recognizer/digit-recognizer/train.csv'
test_file_path = './resources/datasets/digit-recognizer/digit-recognizer/test.csv'


def dataLoaderCreationPhase(yaml_config):
    train_dataset = MNISTDataset(train_file_path, with_label=True)
    train_dataset_size = int(0.8 * train_dataset.__len__())  # TODO make size configureable
    validation_dataset_size = train_dataset.__len__() - train_dataset_size

    train_dataset, validation_dataset = random_split(train_dataset, [train_dataset_size, validation_dataset_size])
    train_dataloader = DataLoader(train_dataset, batch_size=yaml_config[constant.BATCH_SIZE], shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=yaml_config[constant.BATCH_SIZE_TEST],
                                       shuffle=True)

    test_dataset = MNISTDataset(test_file_path, with_label=False)
    test_dataloader = DataLoader(test_dataset, batch_size=yaml_config[constant.BATCH_SIZE_TEST])

    return {'test': test_dataloader, 'validation': validation_dataloader, 'train': train_dataloader}
