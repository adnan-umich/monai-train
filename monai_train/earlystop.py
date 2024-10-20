class EarlyStopping:
    def __init__(self, min_epochs=10, patience=5, threshold=0.01):
        """
        Args:
            min_epochs (int): Minimum number of epochs to run before checking for early stopping.
            patience (int): Number of epochs to wait after last improvement.
            threshold (float): Minimum change in the monitored score to qualify as an improvement.
        """
        self.min_epochs = min_epochs
        self.patience = patience
        self.threshold = threshold
        self.best_score = None
        self.counter = 0

    def should_stop(self, epoch, current_score):
        if epoch < self.min_epochs:
            return False
        
        if self.best_score is None or (current_score - self.best_score) > self.threshold:
            self.best_score = current_score
            self.counter = 0
        else:
            self.counter += 1

        return self.counter >= self.patience