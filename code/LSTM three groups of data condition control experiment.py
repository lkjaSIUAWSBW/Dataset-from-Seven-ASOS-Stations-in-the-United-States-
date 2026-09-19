"""
共同目标时间点下的 Raw 与 Processed 风速预测对照实验。

Raw 模型输入：过去 6 个风速值。
Processed 模型输入：过去 6 个风速值 + MissingTimestamp + WindSpeedNegative + 风速缺失标记。

两组使用完全相同的目标时间点、训练测试时间划分、LSTM 结构和随机种子。
缺失风速只在模型输入阶段用训练集均值临时填补，并保留缺失标记；不会修改任何 CSV。
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
    raise ImportError("本脚本需要 PyTorch，请先安装：pip install torch") from exc

RAW_DIR = Path(r"D:\桌面\有用的论文思路\scientific data\放在github上的\Raw dataset")
PROCESSED_DIR = Path(r"D:\桌面\有用的论文思路\scientific data\放在github上的\Processed dataset")
OUTPUT_FILE = Path(__file__).resolve().parent / "LSTM单步Raw与Processed质量特征对照结果.csv"
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
    """构造共同目标时间点。目标值必须在两套数据中均存在且非负。"""
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
        return {"数据条件":condition,"站点":station,"共同目标时间点数":len(x),"训练样本数":len(xtr),"测试样本数":len(xt),"模型输入特征数":n_features,"输入缺失比例":f"{quality_input:.6f}","平均绝对误差（m/s）":"","均方根误差（m/s）":"","决定系数":"","训练与预测耗时（秒）":f"{time.perf_counter()-started:.2f}","验证状态":"样本不足，未训练"},None
    set_seed(); p=fit_predict(xtr,ytr,xt,n_features); truth=yt.reshape(-1); mae,rmse,r2=metrics(truth,p)
    return {"数据条件":condition,"站点":station,"共同目标时间点数":len(x),"训练样本数":len(xtr),"测试样本数":len(xt),"模型输入特征数":n_features,"输入缺失比例":f"{quality_input:.6f}","平均绝对误差（m/s）":f"{mae:.6f}","均方根误差（m/s）":f"{rmse:.6f}","决定系数":f"{r2:.6f}","训练与预测耗时（秒）":f"{time.perf_counter()-started:.2f}","验证状态":"完成"},(truth,p)

def main():
    set_seed(); rf={p.name:p for p in RAW_DIR.glob("*.csv")}; pf={p.name:p for p in PROCESSED_DIR.glob("*.csv")}
    if not rf or set(rf)!=set(pf): raise FileNotFoundError("Raw 与 Processed 的 CSV 文件名不一致。")
    rows=[]; agg={"Raw 仅使用风速":[],"Processed 使用风速和质量特征":[]}
    for name in sorted(rf):
        station=Path(name).stem; raw=read_raw(rf[name]); proc=read_processed(pf[name]); rx,px,y,n,_,_=build_samples(raw,proc)
        split=int(n*TRAIN_RATIO); ra,pa,ya=fill_and_arrays(rx,px,y,split)
        missing_raw=float(np.isnan(np.asarray(rx,dtype=float)).mean()) if rx else 0.0; missing_pro=float(np.isnan(np.asarray(px,dtype=float)[:,:,0]).mean()) if px else 0.0
        r,rv=run_condition(ra,ya,"Raw 仅使用风速",station,1,missing_raw); rows.append(r)
        if rv: agg["Raw 仅使用风速"].append(rv)
        p,pv=run_condition(pa,ya,"Processed 使用风速和质量特征",station,4,missing_pro); rows.append(p)
        if pv: agg["Processed 使用风速和质量特征"].append(pv)
    for condition,batches in agg.items():
        if not batches: continue
        y=np.concatenate([z[0] for z in batches]); p=np.concatenate([z[1] for z in batches]); mae,rmse,r2=metrics(y,p)
        rows.append({"数据条件":condition,"站点":"全部站点汇总","共同目标时间点数":"","训练样本数":"","测试样本数":len(y),"模型输入特征数":"","输入缺失比例":"","平均绝对误差（m/s）":f"{mae:.6f}","均方根误差（m/s）":f"{rmse:.6f}","决定系数":f"{r2:.6f}","训练与预测耗时（秒）":"","验证状态":"完成"})
    headers=["数据条件","站点","共同目标时间点数","训练样本数","测试样本数","模型输入特征数","输入缺失比例","平均绝对误差（m/s）","均方根误差（m/s）","决定系数","训练与预测耗时（秒）","验证状态"]
    with OUTPUT_FILE.open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=headers); w.writeheader(); w.writerows(rows)
    print(f"实验完成，结果已保存到：{OUTPUT_FILE}")

if __name__=="__main__": main()
