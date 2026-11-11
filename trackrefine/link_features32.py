import numpy as np
from .link_model import link_features_v2, _fresh_boundary

EXTRA_NAMES=['cross_cosine_mean','cross_cosine_std','cross_cosine_p10','cross_cosine_p90',
             'backward_prediction_error','velocity_cosine','gap_scaled_velocity_disagreement',
             'tail_motion_fit_residual','head_motion_fit_residual','endpoint_mask_fill_log_change',
             'tail_fresh_age','head_mask_fill_std']

def fit_motion(observations,height):
    t=np.array([d.frame for d in observations],np.float32)
    c=np.array([(d.box[:2]+d.box[2:])/2 for d in observations],np.float32)
    dt=t-t.mean(); dc=c-c.mean(0)
    v=(dt[:,None]*dc).sum(0)/max(float((dt**2).sum()),1e-6)
    residual=float(np.sqrt(np.mean(np.sum((dc-dt[:,None]*v)**2,axis=1))))/height
    return v,min(residual,3)/3

def features32(first,second):
    base=link_features_v2(first,second)
    a,b=first.last,second.observations[0]
    gap=b.frame-a.frame
    height=max(a.box[3]-a.box[1],b.box[3]-b.box[1],25)
    tail=_fresh_boundary(first.observations,True)
    head=_fresh_boundary(second.observations,False)
    cross=np.stack([d.feature for d in tail])@np.stack([d.feature for d in head]).T
    va,ra=fit_motion(first.observations[-8:],height)
    vb,rb=fit_motion(second.observations[:8],height)
    ca=(a.box[:2]+a.box[2:])/2; cb=(b.box[:2]+b.box[2:])/2
    backward=min(float(np.linalg.norm(cb-vb*gap-ca)/height),12)/12
    cosine=float(va@vb/((np.linalg.norm(va)+1e-6)*(np.linalg.norm(vb)+1e-6)))
    disagreement=min(float(np.linalg.norm(va-vb)*gap/height),12)/12
    fill_a=a.area/max(float(np.prod(a.box[2:]-a.box[:2])),1)
    fill_b=b.area/max(float(np.prod(b.box[2:]-b.box[:2])),1)
    fill_change=min(abs(float(np.log(max(fill_a,1e-6)/max(fill_b,1e-6)))),3)/3
    fresh_a=[d.frame for d in first.observations if d.feature_is_fresh]
    age=min(a.frame-max(fresh_a),90)/90 if fresh_a else 1.
    fill_std=float(np.std([d.area/max(float(np.prod(d.box[2:]-d.box[:2])),1) for d in second.observations[:8]]))
    extra=np.array([cross.mean(),cross.std(),np.quantile(cross,.1),np.quantile(cross,.9),backward,cosine,disagreement,ra,rb,fill_change,age,fill_std],np.float32)
    result=np.concatenate([base,extra])
    assert result.shape==(32,) and np.isfinite(result).all()
    return result
