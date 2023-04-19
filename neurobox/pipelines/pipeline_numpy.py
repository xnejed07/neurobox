import numpy as np
from tqdm import tqdm
import pandas as pd



def pipeline_numpy(sampleIterator, model):
    outputs = []
    metas = []
    for i, (data, meta) in enumerate(tqdm(sampleIterator)):
        y = model.inference(data)
        # if isinstance(meta,pd.core.series.Series):
        #     meta = pd.DataFrame(meta).T.reset_index(drop=True)
        metas.append(meta)
        outputs.append(y)
    outputs = np.concatenate(outputs,axis=0)
    metas = pd.DataFrame(metas).reset_index(drop=True)

    if len(model.classNames) == 1:
        metas[model.classNames[0]] = outputs
    else:
        for k, c in enumerate(model.classNames):
            metas[c] = outputs[:, k]
    return metas