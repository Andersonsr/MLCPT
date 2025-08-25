import os
import sys
from PIL import Image
path = os.path.normpath(os.path.join(os.path.join(os.path.abspath(__file__)), '..', '..'))
sys.path.append(path)
from peft import LoraConfig, get_peft_model
from peft.peft_model import PeftModel



