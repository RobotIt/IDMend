"""Background-only LK flow / ORB, with RANSAC and failed-fit rejection."""
import json
from pathlib import Path
import cv2
import numpy as np

def background(gray,boxes,scale):
    mask=np.full(gray.shape,255,np.uint8)
    for box in boxes:
        x1,y1,x2,y2=np.round(np.array(box)*scale).astype(int)
        mask[max(0,y1-5):min(gray.shape[0],y2+5),max(0,x1-5):min(gray.shape[1],x2+5)]=0
    mask[:5]=0;mask[-5:]=0;mask[:,:5]=0;mask[:,-5:]=0
    return mask

def compute(paths,raw_cache,target,method='background_lk'):
    if target.exists(): return np.load(target)['homographies']
    raw={}
    for line in raw_cache.read_text().splitlines():
        r=json.loads(line);raw[r['frame']]=[d['box'] for d in r['detections']]
    transforms=np.tile(np.eye(3,dtype=np.float64),(max(int(p.stem) for p in paths)+1,1,1))
    cumulative=np.eye(3);previous=None;logs=[]
    orb=cv2.ORB_create(nfeatures=1500,fastThreshold=12);matcher=cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)
    for i,path in enumerate(paths):
        full=cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)
        scale=min(1.,720/full.shape[1]);gray=cv2.resize(full,None,fx=scale,fy=scale) if scale<1 else full
        frame=int(path.stem);mask=background(gray,raw.get(frame,[]),scale)
        step=np.eye(3);used='identity';inlier_ratio=0.;median_error=None
        if previous is not None:
            src=dst=None
            if method=='background_lk':
                pts=cv2.goodFeaturesToTrack(previous,600,.01,8,mask=prev_mask)
                if pts is not None and len(pts)>=12:
                    nxt,status,_=cv2.calcOpticalFlowPyrLK(previous,gray,pts,None)
                    back,status2,_=cv2.calcOpticalFlowPyrLK(gray,previous,nxt,None)
                    good=status.ravel().astype(bool)&status2.ravel().astype(bool)&(np.linalg.norm(back-pts,axis=2).ravel()<1.)
                    locations=np.round(nxt[:,0]).astype(int)
                    inside=(locations[:,0]>=0)&(locations[:,0]<gray.shape[1])&(locations[:,1]>=0)&(locations[:,1]<gray.shape[0])
                    good&=inside
                    good[inside]&=mask[locations[inside,1],locations[inside,0]]>0
                    src=pts[good,0];dst=nxt[good,0];used='background_lk'
            if src is None or len(src)<12:
                kp1,des1=orb.detectAndCompute(previous,prev_mask);kp2,des2=orb.detectAndCompute(gray,mask)
                if des1 is not None and des2 is not None:
                    matches=sorted(matcher.match(des1,des2),key=lambda x:x.distance)[:400]
                    src=np.float32([kp1[m.queryIdx].pt for m in matches]);dst=np.float32([kp2[m.trainIdx].pt for m in matches]);used='background_orb'
            if src is not None and len(src)>=12:
                affine,inliers=cv2.estimateAffinePartial2D(src,dst,method=cv2.RANSAC,ransacReprojThreshold=2.,maxIters=2000,confidence=.995)
                if affine is not None and inliers is not None:
                    keep=inliers.ravel().astype(bool);inlier_ratio=float(keep.mean())
                    errors=np.linalg.norm(src@affine[:,:2].T+affine[:,2]-dst,axis=1)
                    median_error=float(np.median(errors[keep])) if keep.any() else 999.
                    s=float(np.linalg.norm(affine[:,0]));rotation=abs(np.arctan2(affine[1,0],affine[0,0]))
                    translation=float(np.linalg.norm(affine[:,2]))
                    if keep.sum()>=12 and inlier_ratio>=.5 and median_error<=2. and .9<=s<=1.1 and rotation<np.deg2rad(10) and translation<.2*np.hypot(*gray.shape):
                        step[:2]=affine;step[:2,2]/=scale
                    else:used='rejected_identity'
            else:used='insufficient_identity'
            cumulative=step@cumulative
        transforms[frame]=cumulative
        logs.append({'frame':frame,'method':used,'inlier_ratio':inlier_ratio,'median_error':median_error})
        previous,prev_mask=gray,mask
        if (i+1)%200==0 or i+1==len(paths):print(f'CMC {method}/{path.parent.name} {i+1}/{len(paths)}',flush=True)
    target.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(target,homographies=transforms.astype(np.float32))
    target.with_suffix('.json').write_text(json.dumps({'method':method,'fits':logs},indent=2))
    return transforms
