import json
import os
from copy import deepcopy

def get_N(ds:str):
    if "molgan" in ds.lower() or "chordal" in ds.lower():
        return 9
    elif "community" in ds.lower() and "20" in ds:
        return 20
    elif "community" in ds.lower() and "100" in ds:
        return 100
    else:
        raise NotImplementedError("Only comparing these 4 datasets for ICML")

for d in ["rs","ne","point"]:
    os.makedirs(d,exist_ok=True)

def get_json(d):
    return [os.path.join(d,x) for x in os.listdir(d) if "json" in x]
ggg_bases=get_json("ggg")
for base_path in ggg_bases:
    rs_path=os.path.join("rs",f"rs_{os.path.basename(base_path)}")
    ne_path=os.path.join("ne",f"ne_{os.path.basename(base_path)}")
    point_path=os.path.join("point",f"ne_{os.path.basename(base_path)}")
    with open(base_path,"r") as f:
        c=json.load(f)
    ds=c["hyper"]["dataset_hpars"]["dataset"]
    context_dim=c["hyper"]["root_hpars"]["context_dim"]
    node_embed_dim=c["hyper"]["root_hpars"]["node_embedding_dim"]

    crs=deepcopy(c)
    crs["hyper"]["root_hpars"]["name"]="random"
    cne["hyper"]["root_hpars"]["node_embedding_dim"]=context_dim+node_embed_dim
    cne["hyper"]["root_hpars"]["context_dim"]=0
    crs["hyper"]["exp_name"]="GGG_ICML_RS"
    with open(rs_path,"w") as f:
        json.dump(crs,f,indent=4, sort_keys=True)

    cne=deepcopy(c)
    cne["hyper"]["root_hpars"]["node_embedding_dim"]=0
    cne["hyper"]["root_hpars"]["name"]="noneq"

    cne["hyper"]["edge_readout_hpars"]["name"]="noneq"
    cne["hyper"]["trunk_hpars"]["feat_dim"]=context_dim
    cne["hyper"]["exp_name"]="GGG_ICML_NE"
    cne["hyper"]["trunk_hpars"]["name"]="mlp"
    with open(ne_path,"w") as f:
        json.dump(cne,f,indent=4, sort_keys=True)

    cpoint=deepcopy(c)
    cpoint["hyper"]["trunk_hpars"]["name"]="pointmlp"
    with open(point_path,"w") as f:
        json.dump(cpoint,f,indent=4, sort_keys=True)
