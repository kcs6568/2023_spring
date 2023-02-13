import os
import logging

import torch.distributed as dist


class TextLogger:
    def __init__(self, log_dir, print_time=True) -> None:
        self.log_file = os.path.join(log_dir, 'learning_log.log')
        self.print_time = print_time
        self.logger = self._get_root_logger()
    
    def _get_root_logger(self):
        handlers = []
    
        logger = logging.getLogger()

        # 로그의 출력 기준 설정
        logger.setLevel(logging.INFO)

        # log 출력 형식
        if self.print_time:
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        else:
            formatter = logging.Formatter('%(levelname)s - %(message)s')

        # log 출력
        stream_handler = logging.StreamHandler()
        handlers.append(stream_handler)
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
        else:
            rank = 0

        if rank == 0:
            file_handler = logging.FileHandler(self.log_file)
            handlers.append(file_handler)
            
        for handler in handlers:
            handler.setFormatter(formatter)
            handler.setLevel(logging.INFO)
            logger.addHandler(handler)
            
        if rank == 0:
            logger.setLevel(logging.INFO)
        else:
            logger.setLevel(logging.ERROR)
        
        
        return logger
    
    
    def log_text(self, text, level='info'):
        if level == 'info':
            self.logger.info(text)
        elif level == 'error':
            self.logger.error(text)
            
            
            
from torch.utils.tensorboard import SummaryWriter
class TensorBoardLogger(SummaryWriter):
    def __init__(self, interval=1, iters=0, distributed=True, 
                 log_dir=None, comment='', purge_step=None, max_queue=10, 
                 flush_secs=120, filename_suffix='', logging_rank=0):
        super().__init__(log_dir, comment, purge_step, max_queue, flush_secs, filename_suffix)
    
    
    def update_scalars(self, data, time, proc='train'):
        for k, loss in data.items():
            self.add_scalar(f"{proc}/{k}", loss, time)
        
    
    def close_tb(self):
        self.close()