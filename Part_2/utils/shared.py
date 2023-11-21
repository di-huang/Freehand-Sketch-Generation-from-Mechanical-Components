from utils.config import Config
from utils.file_writer import FileWriter

args = Config()
stroke_config = Config()
logger = FileWriter(lazy_init=True)

def update_args(args_):
    args.update(args_)

def update_config(config_):
    stroke_config.update(config_)