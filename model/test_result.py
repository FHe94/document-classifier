
class TestResult:

    def __init__(self, accuracy, per_class_accuracies):
        self.model_name = ""
        self.labels = []
        self.peak_memory_usage = 0
        self.accuracy = accuracy
        self.per_class_accuracies = per_class_accuracies

    def __str__(self):
        return """
Results for model "{}":
    Accuracy:{:>20}
    Peak memory usage: {:>10}
    Per-class accuracies:\t {}
        """.format(self.model_name, self.accuracy, self.peak_memory_usage, self.__per_class_accuracies_to_string())

    def __per_class_accuracies_to_string(self):
        if len(self.labels) == len(self.per_class_accuracies):
            out_str = [""]
            for i in range(len(self.labels)):
                out_str.append("{:<15}{:>10}".format(str(self.labels[i])+":", self.per_class_accuracies[i]))
            return "\n\t".join(out_str)
        else:
            return str(self.per_class_accuracies)