
import sys, os, numpy as np, torch, pandas as pd, json, csv, re, time
from rdkit import Chem, RDLogger; RDLogger.DisableLog("rdApp.*")
from rdkit.Chem import rdMolDescriptors, DataStructs

MSFLOW = "/stor3/AIMS4Meta/源代码/MSFlow-main"; os.chdir(MSFLOW)
sys.path.insert(0, ".")
from modules.cond_lit_model import CondFlowMolBERTLitModule
from flow_matching.path import MixtureDiscreteProbPath
from flow_matching.path.scheduler import PolynomialConvexScheduler
from utils.sample import cond_generate_mols
from utils.metrics import decode_tokens_to_smiles
from utils.functions import canonicalize
from configs import ID2TOK, TOK2ID, PAD
torch.serialization.add_safe_globals([MixtureDiscreteProbPath, PolynomialConvexScheduler, torch.nn.modules.loss.CrossEntropyLoss])

# Step 1: Encode
print("=== Step 1: Encode spectra to CDDD ===")
sys.path.insert(0, "ms_scripts/DiffMS/src")
from mist.models.spectra_encoder import SpectraEncoderGrowing
sys.path.remove("ms_scripts/DiffMS/src")

dev = "cuda:0" if torch.cuda.is_available() else "cpu"
ckpt = torch.load("checkpoints/Encoder/encoder_msg_cddd.pt", map_location=dev)
enc = SpectraEncoderGrowing(inten_transform="float",inten_prob=0.1,remove_prob=0.5,
    peak_attn_layers=2,num_heads=8,pairwise_featurization=True,embed_instrument=False,
    cls_type="ms1",set_pooling="cls",spec_features="peakformula",mol_features="fingerprint",
    form_embedder="pos-cos",output_size=512,hidden_size=512,spectra_dropout=0.0,
    top_layers=1,refine_layers=4,magma_modulo=2048)
enc.load_state_dict(ckpt["model_state_dict"]); enc.to(dev); enc.eval()

ELEM=["C","H","As","B","Br","Cl","Co","F","Fe","I","K","N","Na","O","P","S","Se","Si"]
E2I={e:i for i,e in enumerate(ELEM)}
def f2c(f):
    c=np.zeros(18,dtype=np.int32)
    for e,n in re.findall(r"([A-Z][a-z]?)(\d*)",f):
        n=int(n) if n else 1
        if e in E2I: c[E2I[e]]=min(n,255)
    return c
M={"C":12.0,"H":1.007825,"O":15.994915,"N":14.003074,"P":30.973762,"S":31.972071,"F":18.998403}
def mm(f):
    m=0.0
    for e,c in re.findall(r"([A-Z][a-z]*)(\d*)",f): m+=M.get(e,0)*(int(c) if c else 1)
    return m
ads={"[M+H]+":1.007276,"[M+Na]+":22.989218}

sv="/stor3/AIMS4Meta/数据集/msflow_datasets/spectraverse/spectraverse_pos/subformulae"
test_ids=[l.strip() for l in open("/tmp/eval_testset.txt") if l.strip()]
all_cddd=[]
for sid in test_ids:
    j=json.load(open(f"{sv}/{sid}.json"))
    tbl=j["output_tbl"]; formula=j["cand_form"]; ion=j["cand_ion"]
    pmz=mm(formula)+ads.get(ion,0)
    mzs=np.array(tbl["mz"]); ints=np.array(tbl["ms2_inten"])
    mask=np.abs(mzs-pmz)>0.5
    frags=np.column_stack([mzs[mask],ints[mask]])
    n=min(50,len(frags))
    if n==0: continue
    frags=frags[:n]; mp=n+1
    fv=np.zeros((mp,18),dtype=np.int32); fv[0]=f2c(formula)
    ff=tbl.get("formula",[formula]*n)
    for i in range(n): fv[i+1]=f2c(ff[i] if i<len(ff) else formula)
    fv_t=torch.from_numpy(fv).float().unsqueeze(0).to(dev)
    ty=torch.zeros(1,mp,dtype=torch.long,device=dev); ty[0,0]=3
    it=torch.zeros(1,mp,device=dev)
    if n>0: it[0,1:]=torch.from_numpy(frags[:,1]).float().to(dev)
    batch={"form_vec":torch.clamp(fv_t,0,255),"types":ty,"intens":it,
        "ion_vec":torch.zeros(1,mp,dtype=torch.long,device=dev),
        "num_peaks":torch.tensor([mp],device=dev),
        "instruments":torch.zeros(1,dtype=torch.long,device=dev),"names":[sid]}
    with torch.no_grad(): cddd,_=enc(batch)
    all_cddd.append(cddd.cpu())
    print(f"  {sid} done")
cddd_all=torch.cat(all_cddd)
np.save("/tmp/msflow_eval_cddd.npy",cddd_all.numpy())
print(f"CDDD: {cddd_all.shape}")

# Step 2: Decode
print("\n=== Step 2: Decode CDDD to molecules ===")
cfm=CondFlowMolBERTLitModule.load_from_checkpoint("checkpoints/Decoder/MSFlow_cddds.ckpt",map_location=dev,strict=False)
dec=cfm.model; dec.eval(); dec.to(dev)
results=[]; times=[]
for i in range(len(test_ids)):
    t0=time.time()
    cond=cddd_all[i].unsqueeze(0).expand(5,-1)
    samps=cond_generate_mols(dec,cond=cond,source_distribution="uniform",
        num_samples=5,steps=128,guidance_scale=1.5,temperature=1,device=dev)
    _,smi=decode_tokens_to_smiles(samps,ID2TOK=ID2TOK,TOK2ID=TOK2ID,PAD=PAD)
    smi=[canonicalize(s) for s in smi if s]
    smi=[s for s in smi if s is not None]
    times.append(time.time()-t0)
    for rank,s in enumerate(smi,1):
        if s: results.append({"query_name":test_ids[i],"rank":rank,"smiles":s})
    print(f"  {test_ids[i]}: {len(smi)} cands ({times[-1]:.1f}s)")
pd.DataFrame(results).to_csv("/tmp/msflow_eval_results.csv",index=False)
print(f"\nDone: {len(results)} candidates, avg {np.mean(times):.1f}s/compound")
