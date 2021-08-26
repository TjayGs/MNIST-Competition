from yaml import load, FullLoader
import src.yamlConfigConstants as constant


def configPhase():
    file_stream = open('resources/config/config.yaml', 'r')
    yaml_config = load(file_stream, Loader=FullLoader)  # TODO learn more about loader
    if yaml_config[constant.DEBUG]:
        print('Yaml Config File:')
        print(yaml_config)
    return yaml_config
