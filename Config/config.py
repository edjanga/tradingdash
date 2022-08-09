import json
import pdb
import os


class BaseConfig(object):

    """
        Base class for handling Config files
    """

    def __init__(self,filename):
        with open(filename, 'r') as f:
            config = json.load(f)
        del config['Meta']
        self.config = config


class PlatformConfig(BaseConfig):

    """
        Class handling Config file related to platforms
    """

    def __init__(self,filename='config_platform.json'):
        super().__init__(filename)
        # Filter in relevant info
        platforms = self.config['Platforms'][0]
        self.sources = [source for source in list(platforms.keys()) if source not in ['telegram']]
        platforms = {key: value[0] for key, value in platforms.items()}
        platforms = {key1: {key2: value2 for key2, value2 in value1.items() if key2 not in ['Meta', 'Communication']} \
                 for key1, value1 in platforms.items()}

        self.platforms = platforms


class TiingoConfig(PlatformConfig):

    def __init__(self,filename='config_platform.json'):
        super().__init__(filename)
        self.api = self.platforms['tiingo']['API']
        self.endpoint = self.platforms['tiingo']['Endpoint']
        self.headers = self.platforms['tiingo']['headers'][0]

if __name__ == '__main__':
    tiigo_config_obj = TiingoConfig()
