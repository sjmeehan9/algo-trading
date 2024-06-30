import yaml
import json

# Load a YAML config file
def config_loader(filepath) -> dict:
    try:
        with open(filepath, 'r') as f:
            config = yaml.safe_load(f)
            return config
    except FileNotFoundError:
        raise Exception("Config file not found")
    except yaml.parser.ParserError as exc:
        raise Exception(f"Error parsing YAML: {exc}")


# Load a JSON pipeline file
def pipeline_loader(filepath) -> dict:
    try:
        with open(filepath, 'r') as f:
            pipeline = json.load(f)
            return pipeline
    except FileNotFoundError:
        raise Exception("Pipeline file not found")
    except json.decoder.JSONDecodeError as exc:
        raise Exception(f"Error parsing JSON: {exc}")