import json

class ClassificationArgs:

    def __init__(self, to_classify_url, expected_output):
        self.to_classify_url = to_classify_url
        self.expected_output = expected_output

class ClassificationArgsParser:

    def parse_args(self, args_string):
        raw_obj = json.loads(args_string, encoding="utf-8")
        return ClassificationArgs(raw_obj["to_classify_url_str"], raw_obj["expected_output"])