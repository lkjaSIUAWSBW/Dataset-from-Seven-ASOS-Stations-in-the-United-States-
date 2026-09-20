"""
\u5171\u540c\u76ee\u6807\u65f6\u95f4\u70b9\u4e0b\u7684 Raw \u4e0e Processed \u98ce\u901f\u9884\u6d4b\u5bf9\u7167\u5b9e\u9a8c。

Raw \u6a21\u578b\u8f93\u5165：\u8fc7\u53bb 6 \u4e2a\u98ce\u901f\u503c。
Processed \u6a21\u578b\u8f93\u5165：\u8fc7\u53bb 6 \u4e2a\u98ce\u901f\u503c + MissingTimestamp + WindSpeedNegative + \u98ce\u901f\u7f3a\u5931\u6807\u8bb0。

\u4e24\u7ec4\u4f7f\u7528\u5b8c\u5168\u76f8\u540c\u7684\u76ee\u6807\u65f6\u95f4\u70b9、\u8bad\u7ec3\u6d4b\u8bd5\u65f6\u95f4\u5212\u5206、LSTM \u7ed3\u6784\u548c\u968f\u673a\u79cd\u5b50。
\u7f3a\u5931\u98ce\u901f\u53ea\u5728\u6a21\u578b\u8f93\u5165\u9636\u6bb5\u7528\u8bad\u7ec3\u96c6\u5747\u503c\u4e34\u65f6\u586b\u8865，\u5e76\u4fdd\u7559\u7f3a\u5931\u6807\u8bb0；\u4e0d\u4f1a\u4fee\u6539\u4efb\u4f55 CSV。
"""
from __future__ import annotations
import csv, math, random, time
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError as exc:
    raise ImportError("\u672c\u811a\u672c\u9700\u8981 PyTorch，\u8bf7\u5148\u5b89\u88c5：pip install torch") from exc

RAW_DIR = Path(r"D:\\u684c\u9762\\u6709\u7528\u7684\u8bba\u6587\u601d\u8def\scientific data\\u653e\u5728github\u4e0a\u7684\Raw dataset")
PROCESSED_DIR = Path(r"D:\\u684c\u9762\\u6709\u7528\u7684\u8bba\u6587\u601d\u8def\scientific data\\u653e\u5728github\u4e0a\u7684\Processed dataset")
OUTPUT_FILE = Path(__file__).resolve().parent / "LSTM\u5355\u6b65Raw\u4e0eProcessed\u8d28\u91cf\u7279\u5f81\u5bf9\u7167\u7ed3\u679c.csv"
SEED, INPUT_STEPS, TRAIN_RATIO = 20260919, 6, 0.8
EPOCHS, BATCH_SIZE, HIDDEN_SIZE, LEARNING_RATE = 10, 256, 32, 1e-3
MAX_TRAIN_SAMPLES, MAX_TEST_SAMPLES = 50000, 20000

def set_seed():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

def clean(v): return "" if v is None else str(v).replace("\ufeff", "").strip()

def num(v):
    try:
        x=float(clean(v)); return x if math.isfinite(x) else None
    except (TypeError, ValueError): return None

def dt(v): return datetime.strptime(clean(v), "%Y-%m-%d %H:%M")

def read_raw(path):
    out={}
    with path.open("r",encoding="utf-8-sig",newline="") as f:
        for r in csv.DictReader(f):
            try: t=dt(r.get("valid(UTC)"))
            except (TypeError, ValueError): continue
            v=num(r.get("sknt")); out[t]=None if v is None else v*0.514444
    return out

def read_processed(path):
    out={}
    with path.open("r",encoding="utf-8-sig",newline="") as f:
        for r in csv.DictReader(f):
            try: t=dt(r.get("valid(UTC)"))
            except (TypeError, ValueError): continue
            v=num(r.get("wind_speed_ms")); missing=v is None
            out[t]={"speed":v,
                    "missing_time":1 if clean(r.get("MissingTimestamp"))!="0" else 0,
                    "negative":1 if clean(r.get("WindSpeedNegative"))!="0" else 0,
                    "missing_speed":1 if missing else 0}
    return out

def timeline(a,b):
    start=max(min(a),min(b)); end=min(max(a),max(b))
    start=start.replace(minute=(start.minute//10)*10,second=0,microsecond=0)
    return [start+timedelta(minutes=10*i) for i in range(int((end-start).total_seconds()//600)+1)]

def build_samples(raw, proc):
    """\u6784\u9020\u5171\u540c\u76ee\u6807\u65f6\u95f4\u70b9。\u76ee\u6807\u503c\u5fc5\u987b\u5728\u4e24\u5957\u6570\u636e\u4e2d\u5747\u5b58\u5728\u4e14\u975e\u8d1f。"""
    times=timeline(raw,proc); raw_x=[]; pro_x=[]; y=[]; missing_raw=0; missing_pro=0; targets=0
    for i in range(INPUT_STEPS,len(times)):
        input_times=times[i-INPUT_STEPS:i]; target=times[i]
        rv=raw.get(target); pv=proc.get(target)
        if rv is None or rv < 0 or pv is None or pv["speed"] is None or pv["speed"] < 0 or pv["negative"]:
            continue
        raw_seq=[]; pro_seq=[]; valid=True
        for t in input_times:
            r=raw.get(t); p=proc.get(t)
            if r is None: missing_raw+=1
            if p is None or p["speed"] is None: missing_pro+=1
            raw_seq.append(np.nan if r is None else r)
            if p is None:
                pro_seq.append([np.nan,1,0,1])
            else:
                pro_seq.append([np.nan if p["speed"] is None else p["speed"],p["missing_time"],p["negative"],p["missing_speed"]])
        raw_x.append(raw_seq); pro_x.append(pro_seq); y.append(rv); targets+=1
    return raw_x, pro_x, y, targets, missing_raw, missing_pro

def fill_and_arrays(raw_x, pro_x, y, train_end):
    raw_arr=np.asarray(raw_x,dtype=np.float32).reshape(-1,INPUT_STEPS,1)
    pro_arr=np.asarray(pro_x,dtype=np.float32).reshape(-1,INPUT_STEPS,4)
    y_arr=np.asarray(y,dtype=np.float32).reshape(-1,1)
    raw_mean=float(np.nanmean(raw_arr[:train_end])) if np.isnan(raw_arr[:train_end]).all()==False else 0.0
    pro_mean=float(np.nanmean(pro_arr[:train_end,:,0])) if np.isnan(pro_arr[:train_end,:,0]).all()==False else 0.0
    raw_arr[:,:,0]=np.where(np.isnan(raw_arr[:,:,0]),raw_mean,raw_arr[:,:,0])
    pro_arr[:,:,0]=np.where(np.isnan(pro_arr[:,:,0]),pro_mean,pro_arr[:,:,0])
    return raw_arr,pro_arr,y_arr

def select(idx,limit):
    if limit is None or len(idx)<=limit: return idx
    return idx[np.linspace(0,len(idx)-1,limit,dtype=int)]

class Net(nn.Module):
    def __init__(self,n_features):
        super().__init__(); self.lstm=nn.LSTM(n_features,HIDDEN_SIZE,batch_first=True); self.fc=nn.Linear(HIDDEN_SIZE,1)
    def forward(self,x):
        z,_=self.lstm(x); return self.fc(z[:,-1,:])

def fit_predict(xtr,ytr,xt,n_features):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); model=Net(n_features).to(device)
    opt=torch.optim.Adam(model.parameters(),lr=LEARNING_RATE); loss_fn=nn.MSELoss()
    loader=DataLoader(TensorDataset(torch.from_numpy(xtr),torch.from_numpy(ytr)),batch_size=BATCH_SIZE,shuffle=True)
    model.train()
    for _ in range(EPOCHS):
        for xb,yb in loader:
            xb,yb=xb.to(device),yb.to(device); opt.zero_grad(); loss_fn(model(xb),yb).backward(); opt.step()
    pred=[]; model.eval()
    with torch.no_grad():
        for xb in DataLoader(torch.from_numpy(xt),batch_size=BATCH_SIZE,shuffle=False): pred.append(model(xb.to(device)).cpu().numpy())
    return np.concatenate(pred).reshape(-1)

def metrics(y,p):
    e=p-y; mae=float(np.mean(np.abs(e))); rmse=float(np.sqrt(np.mean(e**2))); den=float(np.sum((y-y.mean())**2)); r2=float(1-np.sum(e**2)/den) if den>0 else float("nan"); return mae,rmse,r2

def run_condition(x,y,condition,station,n_features,quality_input):
    started=time.perf_counter(); split=int(len(x)*TRAIN_RATIO); a=select(np.arange(split),MAX_TRAIN_SAMPLES); b=select(np.arange(split,len(x)),MAX_TEST_SAMPLES)
    xtr,ytr,xt,yt=x[a],y[a],x[b],y[b]
    if len(xtr)<10 or len(xt)<10:
        return {"\u6570\u636e\u6761\u4ef6":condition,"\u7ad9\u70b9":station,"\u5171\u540c\u76ee\u6807\u65f6\u95f4\u70b9\u6570":len(x),"\u8bad\u7ec3\u6837\u672c\u6570":len(xtr),"\u6d4b\u8bd5\u6837\u672c\u6570":len(xt),"\u6a21\u578b\u8f93\u5165\u7279\u5f81\u6570":n_features,"\u8f93\u5165\u7f3a\u5931\u6bd4\u4f8b":f"{quality_input:.6f}","\u5e73\u5747\u7edd\u5bf9\u8bef\u5dee（m/s）":"","\u5747\u65b9\u6839\u8bef\u5dee（m/s）":"","\u51b3\u5b9a\u7cfb\u6570":"","\u8bad\u7ec3\u4e0e\u9884\u6d4b\u8017\u65f6（\u79d2）":f"{time.perf_counter()-started:.2f}","\u9a8c\u8bc1\u72b6\u6001":"\u6837\u672c\u4e0d\u8db3，\u672a\u8bad\u7ec3"},None
    set_seed(); p=fit_predict(xtr,ytr,xt,n_features); truth=yt.reshape(-1); mae,rmse,r2=metrics(truth,p)
    return {"\u6570\u636e\u6761\u4ef6":condition,"\u7ad9\u70b9":station,"\u5171\u540c\u76ee\u6807\u65f6\u95f4\u70b9\u6570":len(x),"\u8bad\u7ec3\u6837\u672c\u6570":len(xtr),"\u6d4b\u8bd5\u6837\u672c\u6570":len(xt),"\u6a21\u578b\u8f93\u5165\u7279\u5f81\u6570":n_features,"\u8f93\u5165\u7f3a\u5931\u6bd4\u4f8b":f"{quality_input:.6f}","\u5e73\u5747\u7edd\u5bf9\u8bef\u5dee（m/s）":f"{mae:.6f}","\u5747\u65b9\u6839\u8bef\u5dee（m/s）":f"{rmse:.6f}","\u51b3\u5b9a\u7cfb\u6570":f"{r2:.6f}","\u8bad\u7ec3\u4e0e\u9884\u6d4b\u8017\u65f6（\u79d2）":f"{time.perf_counter()-started:.2f}","\u9a8c\u8bc1\u72b6\u6001":"\u5b8c\u6210"},(truth,p)

def main():
    set_seed(); rf={p.name:p for p in RAW_DIR.glob("*.csv")}; pf={p.name:p for p in PROCESSED_DIR.glob("*.csv")}
    if not rf or set(rf)!=set(pf): raise FileNotFoundError("Raw \u4e0e Processed \u7684 CSV \u6587\u4ef6\u540d\u4e0d\u4e00\u81f4。")
    rows=[]; agg={"Raw \u4ec5\u4f7f\u7528\u98ce\u901f":[],"Processed \u4f7f\u7528\u98ce\u901f\u548c\u8d28\u91cf\u7279\u5f81":[]}
    for name in sorted(rf):
        station=Path(name).stem; raw=read_raw(rf[name]); proc=read_processed(pf[name]); rx,px,y,n,_,_=build_samples(raw,proc)
        split=int(n*TRAIN_RATIO); ra,pa,ya=fill_and_arrays(rx,px,y,split)
        missing_raw=float(np.isnan(np.asarray(rx,dtype=float)).mean()) if rx else 0.0; missing_pro=float(np.isnan(np.asarray(px,dtype=float)[:,:,0]).mean()) if px else 0.0
        r,rv=run_condition(ra,ya,"Raw \u4ec5\u4f7f\u7528\u98ce\u901f",station,1,missing_raw); rows.append(r)
        if rv: agg["Raw \u4ec5\u4f7f\u7528\u98ce\u901f"].append(rv)
        p,pv=run_condition(pa,ya,"Processed \u4f7f\u7528\u98ce\u901f\u548c\u8d28\u91cf\u7279\u5f81",station,4,missing_pro); rows.append(p)
        if pv: agg["Processed \u4f7f\u7528\u98ce\u901f\u548c\u8d28\u91cf\u7279\u5f81"].append(pv)
    for condition,batches in agg.items():
        if not batches: continue
        y=np.concatenate([z[0] for z in batches]); p=np.concatenate([z[1] for z in batches]); mae,rmse,r2=metrics(y,p)
        rows.append({"\u6570\u636e\u6761\u4ef6":condition,"\u7ad9\u70b9":"\u5168\u90e8\u7ad9\u70b9\u6c47\u603b","\u5171\u540c\u76ee\u6807\u65f6\u95f4\u70b9\u6570":"","\u8bad\u7ec3\u6837\u672c\u6570":"","\u6d4b\u8bd5\u6837\u672c\u6570":len(y),"\u6a21\u578b\u8f93\u5165\u7279\u5f81\u6570":"","\u8f93\u5165\u7f3a\u5931\u6bd4\u4f8b":"","\u5e73\u5747\u7edd\u5bf9\u8bef\u5dee（m/s）":f"{mae:.6f}","\u5747\u65b9\u6839\u8bef\u5dee（m/s）":f"{rmse:.6f}","\u51b3\u5b9a\u7cfb\u6570":f"{r2:.6f}","\u8bad\u7ec3\u4e0e\u9884\u6d4b\u8017\u65f6（\u79d2）":"","\u9a8c\u8bc1\u72b6\u6001":"\u5b8c\u6210"})
    headers=["\u6570\u636e\u6761\u4ef6","\u7ad9\u70b9","\u5171\u540c\u76ee\u6807\u65f6\u95f4\u70b9\u6570","\u8bad\u7ec3\u6837\u672c\u6570","\u6d4b\u8bd5\u6837\u672c\u6570","\u6a21\u578b\u8f93\u5165\u7279\u5f81\u6570","\u8f93\u5165\u7f3a\u5931\u6bd4\u4f8b","\u5e73\u5747\u7edd\u5bf9\u8bef\u5dee（m/s）","\u5747\u65b9\u6839\u8bef\u5dee（m/s）","\u51b3\u5b9a\u7cfb\u6570","\u8bad\u7ec3\u4e0e\u9884\u6d4b\u8017\u65f6（\u79d2）","\u9a8c\u8bc1\u72b6\u6001"]
    with OUTPUT_FILE.open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=headers); w.writeheader(); w.writerows(rows)
    print(f"\u5b9e\u9a8c\u5b8c\u6210，\u7ed3\u679c\u5df2\u4fdd\u5b58\u5230：{OUTPUT_FILE}")

if __name__=="__main__": main()
