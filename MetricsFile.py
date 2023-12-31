class Metrics():
    def __init__(self):
        self.true_pos = 0
        self.true_neg = 0
        self.false_pos = 0
        self.false_neg = 0

        self.loss = 0
        self.batches = 0

    def batch(self, labels, preds, loss=0):
        self.true_pos += ((labels == preds) & (labels == 1)).sum().item()
        self.true_neg += ((labels == preds) & (labels == 0)).sum().item()
        self.false_pos += ((labels != preds) & (labels == 1)).sum().item()
        self.false_neg += ((labels != preds) & (labels == 0)).sum().item()

        self.loss += loss
        self.batches += 1

    def summary(self):
        return {
            "Model Accuracy":     self.accuracy(self.true_pos, self.true_neg, self.false_pos, self.false_neg),
            "Model Loss":         self.loss,
            "Model Sensitivity":  self.sensitivity(self.true_pos, self.false_neg),
            "Model Specificity":  self.specificity(self.true_neg, self.false_pos),
            "Model F1":           self.f1(self.true_pos, self.true_neg, self.false_pos, self.false_neg),
            "Model Precision":    self.precision(self.true_pos, self.false_pos)
        }

    def print_summary(self):
        s = self.summary()
        for key in s: print(key + ": ", s[key])

    def print_one_liner(self, phase):
        s = self.summary()
        print('{} Acc: {:.4f}, {} Loss: {:.4f}, {} Sens: {:.4f}, {} Spec: {:.4f}'.format(phase, s["Model Accuracy"], phase, s["Model Loss"], phase, s["Model Sensitivity"], phase, s["Model Specificity"]))
        return s

    def accuracy(self, true_pos, true_neg, false_pos, false_neg):
        if true_pos + true_neg + false_pos + false_neg == 0:
            return 0
        return (true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg)

    def sensitivity(self, true_pos, false_neg):
        if true_pos + false_neg == 0:
            return 0
        return (true_pos) / (true_pos + false_neg)

    def specificity(self, true_neg, false_pos):
        if true_neg + false_pos == 0:
            return 0
        return (true_neg) / (true_neg + false_pos)

    def precision(self, true_pos, false_pos):
        if true_pos + false_pos == 0:
            return 0
        return (true_pos) / (true_pos + false_pos)

    def f1(self, true_pos, true_neg, false_pos, false_neg):
        if 2 * true_pos + false_pos + false_neg == 0:
            return 0
        return (2 * true_pos) / (2 * true_pos + false_pos + false_neg)










