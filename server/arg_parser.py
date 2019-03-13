import json

class ClassificationServerCommand:

    def __init__(self, command, commandArgs):
        self.command = command
        self.args = commandArgs

    def to_json_string(self):
        return json.dumps(self.__dict__, ensure_ascii=False)

class ClassificationArgsParser:

    def parse_args(self, args_string):
        raw_obj = json.loads(args_string, encoding="utf-8")
        return ClassificationServerCommand(raw_obj["command"], raw_obj.get("args", dict()))