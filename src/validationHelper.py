import torch


class AnalyzingHelper():
    precision_score = 0
    save_model = False
    model_path = None

    def __init__(self,
                 save_model=False,
                 model_path=''):
        self.save_model = save_model
        self.model_path = model_path

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
