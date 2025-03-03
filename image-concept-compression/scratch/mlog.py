import os
from datetime import datetime

class SimpleLogger:
    def __init__(self, name="app", log_dir="logs", file_logging=True):
        self.name = name
        self.log_dir = log_dir
        self.file_logging = file_logging
        self.log_file = None
        
        if self.file_logging:
            self._setup_log_file()

    def _setup_log_file(self):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        
        base_filename = f"{self.name}.log"
        log_path = os.path.join(self.log_dir, base_filename)
        
        # Rename log file if it already exists
        counter = 1
        while os.path.exists(log_path):
            log_path = os.path.join(self.log_dir, f"{self.name}_{counter}.log")
            counter += 1
        
        self.log_file = open(log_path, 'w')

    def _log(self, level, *messages):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        message = " ".join(str(msg) for msg in messages)
        log_entry = f"[{timestamp}] {level}: {message}"
        
        print(log_entry)
        
        if self.file_logging and self.log_file:
            self.log_file.write(log_entry + "\n")
            self.log_file.flush()

    def debug(self, *messages):
        self._log("DEBUG", *messages)

    def info(self, *messages):
        self._log("INFO", *messages)

    def warning(self, *messages):
        self._log("WARNING", *messages)

    def error(self, *messages):
        self._log("ERROR", *messages)

    def critical(self, *messages):
        self._log("CRITICAL", *messages)

    def set_file_logging(self, enabled):
        if enabled and not self.file_logging:
            self.file_logging = True
            self._setup_log_file()
        elif not enabled and self.file_logging:
            self.file_logging = False
            if self.log_file:
                self.log_file.close()
                self.log_file = None

    def __del__(self):
        if self.log_file:
            self.log_file.close()