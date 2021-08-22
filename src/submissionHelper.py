def createSampleSubmissionFile(prediction_list):
    """
    this file will create a file called 'result_submission.txt' foundable in the output directory.
    This file will contain the ImageId and the Label and each row (except the first one) will represent
    one prediction. We will begin the index calculation by 1 in this file
    :param prediction_list: a list of predicted labels (It must be ordered, because the order will specify the row)
    """
    file_stream = open('./resources/output/result_submission.txt', 'w')  # TODO make file nameable via yaml
    file_stream.write('ImageId,Label\n')
    for index, label in enumerate(prediction_list):
        file_stream.write('{},{}\n'.format(index + 1, label))
    file_stream.close()
