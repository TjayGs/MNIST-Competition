import yamlConfigConstants as constants


class EarlyStopHelper:
    """
    Class will help you to use the function of early stops.
    """
    best_loss_value: float = float("inf")
    deterioration_number: int = 0

    allowed_deterioration = 3
    is_early_stop_allowed = False

    def __init__(self, yaml_config):
        self.is_early_stop_allowed = yaml_config[constants.ALLOW_EARLY_STOP]
        self.allowed_deterioration = yaml_config[constants.EARLY_STOP_ALLOWED_DETERIORATION]

    def is_stop(self, loss_value: float):
        """
        Method will return boolean value if you can stop your training or not. It will compare the given loss value
        with the best loss value given before
        """
        # If early stop is not allowed or allowed_deterioration is 0 then you should never stop
        if self.is_early_stop_allowed and self.allowed_deterioration > 0:
            # If best loss value is worse than given loss (current loss value is better or equal)
            # value then reset deterioration_number and return false
            if self.best_loss_value >= loss_value:
                print('Given loss number {} is higher then a best loss number before {}'.format(loss_value,
                                                                                                self.best_loss_value))
                self.deterioration_number = 0
                self.best_loss_value = loss_value
                return False
            else:
                # If value is not better, than increase deterioration number and if allowed is equal to this number,
                # then return True
                self.deterioration_number += 1
                if self.deterioration_number == self.allowed_deterioration:
                    print('Stopping training, because of EarlyStopHelper')
                    return True
                return False
        else:
            return False
