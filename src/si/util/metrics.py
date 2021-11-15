def accuracy_score(y_true, y_pred):
    '''Class performance metric that computes the accuracy and y_pred'''
    correct = 0
    for true, pred in zip(y_true, y_pred):
        if true == pred:
            correct += 1                                  # count the correct predictions
    accuracy = correct/len(y_true)
    return accuracy

def mse(y_true, y_pred, squared = True):
    '''Mean squared error regression loss funcion.
    Parameters
    
    :param numpy.array y_true: array-like of shape(n_samples,)
        Ground truth (correct) target values
    :param numpy.array y_pred: array-like of shape(n_samples,)
        Estimated target values
    _param bool squared: If True resturns MSE, if false returns RMSE'''
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    errors = np.average((y_true-y_pred)**2, axis = 0)
    if not squared:
        errors = np.sqrt(errors)
    return np.average(errors)
    
