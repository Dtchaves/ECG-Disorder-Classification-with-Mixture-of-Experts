from config.config import MainConfig

import logging

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    
    config = MainConfig()
    
    if config.train_bool:
        logging.info("\n\n -----TRAINING-----\n\n")
        config.train.run()
        
    if config.evaluate_bool:
        logging.info("\n\n -----EVALUATING-----\n\n")
        config.evaluate.run()