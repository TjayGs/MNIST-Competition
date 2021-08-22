from __future__ import print_function

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from yaml import load

import src.yamlConfigConstants as const
import src.submissionHelper as submission_helper
from src.MnistNets import MnistNetV1
from src.MNISTDataset import MNISTDataset
from src.validationHelper import AnalyzingHelper

train_file_path = './resources/datasets/digit-recognizer/digit-recognizer/train.csv'
test_file_path = './resources/datasets/digit-recognizer/digit-recognizer/test.csv'


def main():
    print("Here we go !")
    # Configuration Phase

    # Load Config
    print("Load Config")
    file_stream = open('resources/config/config.yaml', 'r')
    yaml_config = load(file_stream)
    if yaml_config[const.DEBUG]:
        print('Yaml Config File:')
        print(yaml_config)

    # Create Datasets and Dataloader
    print('Start creating datasets and dataloaders')
    train_dataset = MNISTDataset(train_file_path, with_label=True)
    train_dataset_size = int(0.8 * train_dataset.__len__())  # TODO make size configureable
    validation_dataset_size = train_dataset.__len__() - train_dataset_size

    train_dataset, validation_dataset = random_split(train_dataset, [train_dataset_size, validation_dataset_size])
    train_dataloader = DataLoader(train_dataset, batch_size=yaml_config[const.BATCH_SIZE], shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=yaml_config[const.BATCH_SIZE_TEST], shuffle=True)

    test_dataset = MNISTDataset(test_file_path, with_label=False)
    test_dataloader = DataLoader(test_dataset, batch_size=yaml_config[const.BATCH_SIZE_TEST])

    # Creation Phase
    print('Create model and optimizer')
    model = MnistNetV1(yaml_config[const.DEBUG])
    optimizer = optim.Adam(model.parameters(), yaml_config[const.LEARNING_RATE])
    analyzer = AnalyzingHelper(yaml_config[const.SAVE_MODEL],
                               './resources/output/model.pt')

    # Training Phase
    print('Beginning Trainings/Validation Phase')
    for x in range(yaml_config[const.EPOCHS]):
        print('Beginning Training in Epoch {}'.format(x + 1))
        model.train()
        epoch_loss = 0
        for local_batch, local_labels in train_dataloader:
            output = model(local_batch)
            loss = F.cross_entropy(output, local_labels)
            epoch_loss += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Verficiation Phase
        print('Epoch loss: {}'.format(epoch_loss))
        model.eval()
        prediction_score = 0
        for local_valid_batch, local_valid_label in validation_dataloader:
            with torch.no_grad():
                output = model(local_valid_batch)
                prediction_score += (torch.max(output, 1)[1].data.squeeze() == local_valid_label).sum().item()
        analyzer.handle_precision_score((prediction_score / len(validation_dataset)),
                                        model=model)

    analyzer.print_current_scores()

    # Test Phase
    saved_model = torch.load('./resources/output/model.pt')
    saved_model.eval()
    prediction_list = []
    for local_valid_batch, local_valid_label in test_dataloader:
        with torch.no_grad():
            output = saved_model(local_valid_batch)
            prediction_list.append(torch.max(output, 1)[1].numpy()[0])

    submission_helper.createSampleSubmissionFile(prediction_list)

    # Result Phase
    print("We are done !")


if __name__ == '__main__':
    main()
