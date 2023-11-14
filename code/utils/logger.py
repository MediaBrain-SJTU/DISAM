import os
import logging

def Get_Logger(file_name, file_save=True, display=True):
    formatter = logging.Formatter('%(asctime)s %(message)s', datefmt="%Y-%m-%d %H:%M")
    
    logger = logging.Logger(file_name, logging.INFO)
    logger.setLevel(logging.INFO)
    # file handle
    if file_save:
        if os.path.isfile(file_name):
            fh = logging.FileHandler(file_name, mode='a', encoding='utf-8')
        else:
            fh = logging.FileHandler(file_name, mode='w', encoding='utf-8')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    # controler handle
    if display:
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    
    return logger