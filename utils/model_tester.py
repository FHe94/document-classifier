import random
from .memory_profiler import MemoryProfiler
from classifier.loading.model_loader import ModelLoader
from classifier.model.test_result import TestResultLoader

class ModelTester:

    def test_models(self, model_config_paths, samples, labels):
        samples_for_memory_usage_test = self.get_samples_for_memory_usage_test(samples, 50)
        test_results = []
        for model_config_path in model_config_paths:
            test_results.append(self.test_model(model_config_path, samples, labels, samples_for_memory_usage_test))
        return test_results

    def test_model(self, model_config_path, samples, labels, samples_for_memory_usage_test):
        model = ModelLoader().load_model(model_config_path)
        print("Testing model {}".format(model.name))
        test_result = self.__test_accuracy(model, samples, labels)
        print("Measuring memory usage")
        test_result.memory_usage_info = self.__test_memory_usage_for_predict(model_config_path, samples_for_memory_usage_test)
        print(test_result.memory_usage_info.format_data_point(test_result.memory_usage_info.peak_memory_usage))
        return test_result

    def __test_accuracy(self, model, samples, labels):
        test_result = model.test_model(samples, labels)
        print("Total accuracy: {}".format(test_result.accuracy))
        print("Per-class accuracies: ")
        print(test_result.per_class_accuracies)
        return test_result

    def __test_memory_usage_for_predict(self, model_config_path, samples):
        args = [ "python3", "predict.py", model_config_path] + samples
        return MemoryProfiler().profile(args)

    def get_samples_for_memory_usage_test(self, samples, num_samples):
        return random.sample(samples, min(num_samples, len(samples)))