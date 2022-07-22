import yaml
from common.truss_server import TrussServer
from model_wrapper import ModelWrapper

CONFIG_FILE = 'config.yaml'


if __name__ == '__main__':
    with open(CONFIG_FILE, encoding='utf-8') as config_file:
        config = yaml.safe_load(config_file)
        model = ModelWrapper(config)
        model.load()
        TrussServer(workers=1).start([model])
