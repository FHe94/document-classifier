import argparse
from classifier.model.test_result import TestResultLoader, TestResultPlotter

def main():
    args = parse_args()
    loader = TestResultLoader()
    plot_groups = dict()
    for test_result_filepath in args.test_result_filepaths:
        test_results = loader.load_test_results(test_result_filepath)
        model_names = []
        for test_result in test_results:
            model_names.append(test_result.model_name)
        group_name = get_group_name(model_names, 50)
        plot_groups[group_name] = test_results
    TestResultPlotter().plot_memory_usages(plot_groups)

def get_group_name(model_names, max_length):
    group_name = ", ".join(model_names)
    if len(group_name) > max_length:
        group_name = "{}...".format(group_name[0:max_length - 3])
    return group_name

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("test_result_filepaths", nargs="+")
    return parser.parse_args()

if __name__ == "__main__":
    main()