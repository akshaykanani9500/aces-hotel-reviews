import os

# Tests always use the YAML sample stimuli, never a synced Google Sheet snapshot on your computer.
os.environ["HOTEL_LAB_SOURCE"] = "yaml"
