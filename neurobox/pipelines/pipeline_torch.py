import pandas as pd
from tqdm import tqdm
import numpy as np
import torch

def pipeline_torch(sampleIterator, model):
    outputs = []
    metas = []
    model = model.eval()
    with torch.no_grad():
        for i, (data, meta) in enumerate(tqdm(sampleIterator)):
            y = model.inference(data)
            y = y.data.cpu().numpy()
            metas.append(meta)
            outputs.append(y)
    outputs = np.concatenate(outputs,axis=0)
    metas = pd.concat(metas).reset_index(drop=True)
    if len(model.classNames) == 1:
        metas[model.classNames[0]] = outputs
    else:
        for k, c in enumerate(model.classNames):
            metas[c] = outputs[:, k]
    return metas