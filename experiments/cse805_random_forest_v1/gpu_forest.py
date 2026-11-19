"""Evaluate the saved sklearn trees in their original order on full-size pixel blocks."""
import numpy as np
import torch
import triton
import triton.language as tl


@triton.jit
def _trees(X, roots, feature, threshold, children, probability, output, n,
           TREES:tl.constexpr, DEPTH:tl.constexpr, BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    m=tl.load(X+4*i,mask=i<n,other=0)
    h=tl.load(X+4*i+1,mask=i<n,other=0)
    margin=tl.load(X+4*i+2,mask=i<n,other=0)
    cls=tl.load(X+4*i+3,mask=i<n,other=0).to(tl.int32)
    result=tl.full((BLOCK,),0,tl.float64)
    for t in range(TREES):
        at=tl.load(roots+t)+tl.full((BLOCK,),0,tl.int32)
        for level in range(DEPTH):
            f=tl.load(feature+at)
            value=tl.where(f==0,m,tl.where(f==1,h,tl.where(f==2,margin,(cls==f-3).to(tl.float32))))
            go_right=(value.to(tl.float64)>tl.load(threshold+at)).to(tl.int32)
            at=tl.load(children+2*at+go_right)
        result+=tl.load(probability+at)
    tl.store(output+i,result/TREES,mask=i<n)


class ForestScorer:
    def __init__(self, model):
        forest=model["forest"]
        assert forest.n_jobs==1 and np.array_equal(forest.classes_,[0,1])
        self.depth=max(t.tree_.max_depth for t in forest.estimators_)
        self.trees=len(forest.estimators_)
        roots,features,thresholds,children,probs=[],[],[],[],[]
        base=0
        for estimator in forest.estimators_:
            t=estimator.tree_
            nodes=np.arange(t.node_count)
            # Self-loops keep an early leaf fixed until the deepest tree is done.
            link=np.column_stack([np.where(edges<0,nodes,edges)+base for edges in
                                  (t.children_left,t.children_right)]).astype(np.int32)
            roots.append(base)
            features.append(t.feature.astype(np.int32))
            thresholds.append(t.threshold)
            children.append(link)
            probs.append(t.value[:,0,1])
            base+=t.node_count
        arrays=(np.array(roots,dtype=np.int32),np.concatenate(features),np.concatenate(thresholds),
                np.concatenate(children),np.concatenate(probs))
        self.arrays=[torch.as_tensor(np.ascontiguousarray(a),device="cuda") for a in arrays]

    def __call__(self, X):
        assert X.ndim==2 and X.shape[1]==4 and X.dtype==np.float32
        x=torch.as_tensor(np.ascontiguousarray(X),device="cuda")
        out=torch.empty(len(X),dtype=torch.float64,device="cuda")
        _trees[(triton.cdiv(len(X),128),)](x,*self.arrays,out,len(X),self.trees,self.depth,128)
        return out.cpu().numpy()
