from __future__ import print_function

import torch
import torch.nn.functional as F
import torch.optim as optim
import configPhase
import dataLoaderCreationPhase

import src.submissionHelper as submission_helper
import src.yamlConfigConstants as const
from src.MnistNets import MnistNetV1
from src.validationHelper import AnalyzingHelper
from src.Stopwatch import Stopwatch


def main():
    print("Here we go !")
    stopwatch = Stopwatch()
    stopwatch.start('Overall')
    # Configuration Phase

    # Config Phase
    print("Config Phase")
    stopwatch.start('ConfigPhase')
    yaml_config = configPhase.configPhase()
    stopwatch.stop('ConfigPhase')

    # DataLoaderCreationPhase
    print('Start creating datasets and dataloaders')
    stopwatch.start('DataLoaderCreationPhase')
    dataLoader = dataLoaderCreationPhase.dataLoaderCreationPhase(yaml_config=yaml_config)
    stopwatch.stop('DataLoaderCreationPhase')

    # ModelCreationPhase
    print('Create model and optimizer')
    stopwatch.start('ModelCreationPhase')
    model = MnistNetV1(yaml_config[const.DEBUG])
    optimizer = optim.Adam(model.parameters(), yaml_config[const.LEARNING_RATE])
    analyzer = AnalyzingHelper(yaml_config[const.SAVE_MODEL],
                               './resources/output/model.pt')
    stopwatch.stop('ModelCreationPhase')
    # Training Phase
    print('Beginning Trainings/Validation Phase')
    stopwatch.start('TrainingsValidationPhase')
    for x in range(yaml_config[const.EPOCHS]):
        stopwatch.start('TrainingEpoch{}'.format(x))
        print('Beginning Training in Epoch {}'.format(x + 1))
        model.train()
        epoch_loss = 0
        for local_batch, local_labels in dataLoader['train']:
            output = model(local_batch)
            loss = F.cross_entropy(output, local_labels)
            epoch_loss += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        stopwatch.stop('TrainingEpoch{}'.format(x))

        # Verfication Phase
        print('Epoch loss: {}'.format(epoch_loss))
        model.eval()
        prediction_score = 0
        stopwatch.start('ValidationEpoch{}'.format(x))
        for local_valid_batch, local_valid_label in dataLoader['validation']:
            with torch.no_grad():
                output = model(local_valid_batch)
                prediction_score += (torch.max(output, 1)[1].data.squeeze() == local_valid_label).sum().item()
        analyzer.handle_precision_score((prediction_score / len(dataLoader['validation'])),
                                        model=model)
        stopwatch.stop('ValidationEpoch{}'.format(x))

    analyzer.print_current_scores()

    # Test Phase
    stopwatch.start('TestPhase')

    saved_model = torch.load('./resources/output/model.pt')
    saved_model.eval()
    prediction_list = []
    for local_valid_batch, local_valid_label in dataLoader['test']:
        with torch.no_grad():
            output = saved_model(local_valid_batch)
            prediction_list.append(torch.max(output, 1)[1].numpy()[0])

    submission_helper.createSampleSubmissionFile(prediction_list)

    stopwatch.stop('TestPhase')
    # Result Phase
    stopwatch.stop('Overall')
    print("We are done !")


if __name__ == '__main__':
    main()
