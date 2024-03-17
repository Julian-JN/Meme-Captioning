import wandb


class Logger:

    def __init__(self, experiment_name, logger_name='logger', project='inm706', model=None):
        logger_name = f'{logger_name}-{experiment_name}'
        logger = wandb.init(project=project, name=logger_name)
        logger.watch(model, log="all")
        self.logger = logger
        return

    def get_logger(self):
        return self.logger


