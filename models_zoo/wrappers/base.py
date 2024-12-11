from abc import abstractmethod


class BaseModel:
    def __init__(self, config):
        self.config = config
    
    @abstractmethod
    def generate(self):
        raise NotImplementedError

    @abstractmethod
    def construct_prompt(self):
        raise NotImplementedError

    def preprocess_input(self, text):
        pass

    def postprocess_output(self, response):

        if 'assistant\n' in response:
            response = response.split('assistant\n')[1].strip()
        #elif response.startswith('[Sharing Image]'):
        #    response = response.split('[Sharing Image] ')[1].strip()
        return response
