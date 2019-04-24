import utils.utils as utils
from utils.memory_profiler import MemoryUsageInfo
import matplotlib.pyplot as plt
import math

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
            result_dict["per_class_accuracies"] = result_dict["per_class_accuracies"].tolist()
            if result.memory_usage_info is not None:
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
            test_result.memory_usage_info = self.__parse_memory_usage_info(result_json.get("memory_usage_info"))
            test_results.append(test_result)
        return test_results


    def __parse_memory_usage_info(self, memory_usage_json):
        if memory_usage_json is not None:
            return MemoryUsageInfo(memory_usage_json["memory_usage"], memory_usage_json["peak_memory_usage"], memory_usage_json["interval"])
        else:
            return None

class TestResultPlotter:

    def plot_memory_usages(self, test_result_groups):
        plt.figure()
        num_columns = min(len(test_result_groups.keys()), 3)
        num_rows = math.ceil(len(test_result_groups.keys()) / num_columns)
        index = 1
        position = self.__get_figure_layout(num_rows, num_columns)
        for group_name, test_results in test_result_groups.items():
            self.__create_subplot(test_results, group_name, position+index)
            index +=1
        plt.show()

    def __get_figure_layout(self, num_rows, num_columns):
        return num_rows * 100 + num_columns * 10

    def __create_subplot(self, test_results, plot_title, position):
        plt.subplot(position)
        plt.title(plot_title)
        plt.ylabel("Memory usage in MB")
        plt.xlabel("Time in seconds")
        plot_function = self.__plot_test_result_single if len(test_results) == 1 else self.__plot_test_result_multiple
        for test_result in test_results:
            plot_function(test_result)
        plt.legend()

    def __plot_test_result_single(self, test_result):
        memory_usage = test_result.memory_usage_info
        if memory_usage is not None:
            plt.plot(self.__get_x_axis(memory_usage), memory_usage.memory_usage)

    def __plot_test_result_multiple(self, test_result):
        memory_usage = test_result.memory_usage_info
        if memory_usage is not None:
            plt.plot(self.__get_x_axis(memory_usage), memory_usage.memory_usage, label=test_result.model_name)

    def __get_x_axis(self, memory_usage):
        return [ i * memory_usage.interval for i in range(len(memory_usage.memory_usage)) ]

