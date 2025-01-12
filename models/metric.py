class Metric:
    def __init__(self, name, description, scale):
        self.name = name
        self.description = description
        self.scale = scale

    def set_instruction(self, instruction):
        self.instruction = instruction
    def to_str(self):
        return f"[Metric]: Name: {self.name}, Description: {self.description}, Scale: {self.scale} [Metric]\n"
    def to_inst(self):
        return  f"[Metric]: Name: {self.name}, Description: {self.description}, Scale: {self.scale} [Metric]\n{self.instruction}"

class Metrics_set:
    def __init__(self, dataset, source, set_of_metrics):
        self.dataset = dataset
        self.source = source
        self.set_of_metrics = set_of_metrics
        self.overall_metric = set_of_metrics[-1]
        