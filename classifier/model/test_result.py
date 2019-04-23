import utils.utils as utils
from utils.memory_profiler import MemoryUsageInfo

class TestResult:

    def __init__(self, accuracy, per_class_accuracies):
        self.model_name = ""
        self.labels = []
        self.memory_usage_info = None
        self.accuracy = accuracy
        self.per_class_accuracies = per_class_accuracies

    def __str__(self):
        return """
Results for model "{}":
    Accuracy:{:>20}
    {}
    Per-class accuracies:\t {}
        """.format(self.model_name, self.accuracy, self.__get_memory_usage_string(), self.__per_class_accuracies_to_string())

    def __get_memory_usage_string(self):
        if self.memory_usage_info is not None:
            return "Peak memory usage: {:>10}".format(self.memory_usage_info.format_data_point(self.memory_usage_info.peak_memory_usage))
        else:
            return ""

    def __per_class_accuracies_to_string(self):
        if len(self.labels) == len(self.per_class_accuracies):
            out_str = [""]
            for i in range(len(self.labels)):
                out_str.append("{:<15}{:>10}".format(str(self.labels[i])+":", self.per_class_accuracies[i]))
            return "\n\t".join(out_str)
        else:
            return str(self.per_class_accuracies)


class TestResultLoader:

    def save_test_results(self, path, test_results):
        results_json = []
        for result in test_results:
            result_dict = result.__dict__
            result_dict["memory_usage_info"] = result.memory_usage_info.__dict__
            results_json.append(result_dict)
        utils.save_json_file(path, results_json)

    def load_test_results(self, path):
        return utils.try_parse_json_config(self._load_test_results, path)

    def _load_test_results(self, path):
        test_results = []
        for result_json in utils.read_json_file(path):
            test_result = TestResult(result_json["accuracy"], result_json["per_class_accuracies"])
            test_result.labels = result_json.get("labels", [])
            test_result.model_name = result_json.get("model_name", "")
            test_result.memory_usage_info = self.__parse_memory_usage_info(result_json["memory_usage_info"])
            test_results.append(test_result)
        return test_results


    def __parse_memory_usage_info(self, memory_usage_json):
        return MemoryUsageInfo(memory_usage_json["memory_usage"], memory_usage_json["peak_memory_usage"], memory_usage_json["interval"])