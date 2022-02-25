
class ClusterLabels:

    def __init__(self, labels):
        self.labels_ = labels

class TrivialClustering:

    def fit(self, data):
        return ClusterLabels([0 for _ in data])

