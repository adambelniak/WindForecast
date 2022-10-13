from dataclasses import dataclass

@dataclass
class AnalysisSettings:
    input_file: str = 'analysis.json'

    target_parameter: str = 'Temperature'