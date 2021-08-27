import torch
import matplotlib.pyplot as plt
import tabulate


class AnalyzingHelper():
    precision_score = 0
    loss_list = []
    save_model = False
    model_path = None

    def __init__(self,
                 save_model=False,
                 model_path=''):
        self.save_model = save_model
        self.model_path = model_path

    def handle_loss(self,
                    loss: float):
        """
        method will take the given loss and save it within this object.
        It will also print the current loss
        :param loss:
        :return:
        """
        print('Current Loss Score: {}'.format(loss))
        self.loss_list.append(loss)

    def show_loss_flow(self):
        """
        Method will print the loss for all epochs and will create a diagram with the loss flow
        """
        tabulate_data = []
        for x in range(len(self.loss_list)):
            tabulate_data.append([x, self.loss_list[x]])
        print(tabulate.tabulate(tabulate_data, ['Epoch', 'Loss Value']))

        fig = plt.figure()
        plt.plot(self.loss_list)
        plt.savefig('./resources/output/loss.png')
        plt.close(fig)

    def handle_precision_score(self,
                               precision_score: float,
                               model=None):
        print('We\'ve got following precision score: {}'.format(precision_score))
        if precision_score > self.precision_score:
            print('We have an higher precision score found with a precision of {} > {}'.format(precision_score,
                                                                                               self.precision_score))
            self.precision_score = precision_score
            if self.save_model:
                print('model will be saved under {}'.format(self.model_path))
                torch.save(model, self.model_path)

    def print_current_scores(self):
        print('The highest achieved precision score was: {}'.format(self.precision_score))
