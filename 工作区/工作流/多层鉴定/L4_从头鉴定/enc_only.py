import sys, os, json, re, numpy as np, torch, time
os.chdir("/stor3/AIMS4Meta/\u6e90\u4ee3\u7801/MSFlow-main")
sys.path.insert(0, "ms_scripts/DiffMS/src")
from mist.models.spectra_encoder import SpectraEncoderGrowing

dev = "cuda:0" if torch.cuda.is_available() else "cpu"
print("Device:", dev)
ckpt = torch.load("checkpoints/Encoder/encoder_msg_cddd.pt", map_location=dev)
enc = SpectraEncoderGrowing(
    inten_transform="float", inten_prob=0.1, remove_prob=0.5,
    peak_attn_layers=2, num_heads=8, pairwise_featurization=True,
    embed_instrument=False, cls_type="ms1", set_pooling="cls",
    spec_features="peakformula", mol_features="fingerprint",
    form_embedder="pos-cos", output_size=512, hidden_size=512,
    spectra_dropout=0.0, top_layers=1, refine_layers=4, magma_modulo=2048)
enc.load_state_dict(ckpt["model_state_dict"])
enc.to(dev); enc.eval()
print("Encoder loaded")

ELEM = ["C","H","As","B","Br","Cl","Co","F","Fe","I","K","N","Na","O","P","S","Se","Si"]
E2I = {e:i for i,e in enumerate(ELEM)}
def f2c(f):
    import re; c=np.zeros(18,dtype=np.int32)
    for e,n in re.findall(r"([A-Z][a-z]?)(\d*)", f):
        n=int(n) if n else 1
        if e in E2I: c[E2I[e]]=min(n,255)
    return c
M = {"C":12.0,"H":1.007825,"O":15.994915,"N":14.003074,"P":30.973762,"S":31.972071,"F":18.998403}
def mm(f):
    import re; m=0.0
    for e,c in re.findall(r"([A-Z][a-z]*)(\d*)", f): m+=M.get(e,0)*(int(c) if c else 1)
    return m
ads = {"[M+H]+":1.007276,"[M+Na]+":22.989218}

sv = "/stor3/AIMS4Meta/\u6570\u636e\u96c6/msflow_datasets/spectraverse/spectraverse_pos/subformulae"
test_ids = [l.strip() for l in open("/tmp/eval_testset.txt") if l.strip()]
all_cddd = []
for sid in test_ids:
    j = json.load(open(f"{sv}/{sid}.json"))
    tbl = j["output_tbl"]; formula = j["cand_form"]; ion = j["cand_ion"]
    pmz = mm(formula) + ads.get(ion, 0)
    mzs = np.array(tbl["mz"]); ints = np.array(tbl["ms2_inten"])
    mask = np.abs(mzs - pmz) > 0.5
    frags = np.column_stack([mzs[mask], ints[mask]])
    n = min(50, len(frags))
    if n == 0: continue
    frags = frags[:n]; mp = n + 1
    fv = np.zeros((mp, 18), dtype=np.int32)
    fv[0] = f2c(formula)
    frag_f = tbl.get("formula", [formula]*n)
    for i in range(n): fv[i+1] = f2c(frag_f[i] if i < len(frag_f) else formula)
    fv_t = torch.from_numpy(fv).float().unsqueeze(0).to(dev)
    ty = torch.zeros(1, mp, dtype=torch.long, device=dev); ty[0,0] = 3
    it = torch.zeros(1, mp, device=dev)
    if n > 0: it[0,1:] = torch.from_numpy(frags[:,1]).float().to(dev)
    batch = {"form_vec": torch.clamp(fv_t,0,255), "types": ty, "intens": it,
        "ion_vec": torch.zeros(1,mp,dtype=torch.long,device=dev),
        "num_peaks": torch.tensor([mp],device=dev),
        "instruments": torch.zeros(1,dtype=torch.long,device=dev), "names": [sid]}
    with torch.no_grad(): cddd,_ = enc(batch)
    all_cddd.append(cddd.cpu())
    print(sid, "done")
cddd_all = torch.cat(all_cddd)
np.save("/tmp/cddd_encoded.npy", cddd_all.numpy())
print("CDDD saved:", cddd_all.shape)
