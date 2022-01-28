from ggg.experiments.GGG import _THIS_FILE_PATH as EXP_FILE_PATH
import subprocess as sp
import os
from tqdm import tqdm
from ggg_data.icml2022_benchmarks import BENCHMARKS
config_base="paperconfigs"
assert os.path.exists(config_base)
variations=["ggg","ne"]#,"rs"]#,"point"]
CONFIG_PATHS=[]
EPOCHS=100

for var in variations:
    var_base=os.path.join(config_base,var)
    c20=[ x for x in os.listdir(var_base) if "comm20.json" in  x and x[-4:]=="json"][0]
    CONFIG_PATHS.append(
        os.path.join(var_base,c20)
    )
if __name__=="__main__":
    for CP in CONFIG_PATHS:
        for b in BENCHMARKS.keys():
            sp.call(["python",EXP_FILE_PATH,"with",CP,f"benchmark={b}",f"epochs={EPOCHS}"])
