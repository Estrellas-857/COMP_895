"""Verify and visualize saved Stage 5D measurements; no model or dataset reads."""
from collections import Counter
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from benchmark_final_reliability import ROOT, read_csv, statistics, wrap_up
from run_final_reliability import load_final_config


def main():
    rows = read_csv(ROOT/'runtime_per_image.csv')
    summary = read_csv(ROOT/'runtime_summary.csv')
    memory = read_csv(ROOT/'memory_summary.csv')
    subset = read_csv(ROOT/'test_subset.csv')
    assert len(rows)==120 and len(subset)==20
    assert Counter(r['condition'] for r in subset)==dict(fog=5,rain=5,snow=5,night=5)
    assert len({(r['image'],r['repeat'],r['pipeline']) for r in rows})==120
    for r in rows:
        assert (r['width'],r['height'],r['batch_size'])==('1920','1080','1')
    for s,m in zip(summary,memory):
        measured = [r for r in rows if r['pipeline']==s['pipeline']]
        assert len(measured)==60
        assert Counter(r['pair_position'] for r in measured)=={'1':30,'2':30}
        stats=statistics([float(r['latency_ms']) for r in measured])
        for k,v in stats.items():
            assert np.isclose(v,float(s[k]),rtol=0,atol=1e-12)
        assert max(int(r['peak_allocated_bytes']) for r in measured)==int(m['peak_allocated_bytes'])
        assert len({r['resident_allocated_bytes_before_run'] for r in measured})==1
    colors=['#718096','#157f8d']; labels=['SegFormer baseline','Final reliability']
    for filename,values,ylabel,title,notes in (
        ('fig1_latency_comparison.png',[float(s['mean_latency_ms']) for s in summary],
         'Mean latency per image (ms)','Frozen system · measured computational latency',
         'RTX 3060 · batch 1 · 20 RGB images · 3 repeats/image · CUDA-synchronized wall clock\nDots: individual measurements (60/pipeline). No decoding, provenance I/O or file writing.'),
        ('fig2_memory_comparison.png',[float(s['peak_allocated_mib']) for s in memory],
         'Peak PyTorch allocated memory (MiB)','Frozen system · peak GPU allocation',
         'RTX 3060 · batch 1 · 1920 × 1080 outputs\nMaximum over measured runs; includes resident model. Not total GPU usage or reserved memory.'),
    ):
        fig,ax=plt.subplots(figsize=(9,5.5))
        bars=ax.bar([0,1],values,width=.5,color=colors,alpha=.85)
        ax.bar_label(bars,labels=[f'{v:.2f}' for v in values],padding=8,fontsize=12,
                     bbox=dict(facecolor='white',edgecolor='none',alpha=.9,pad=2))
        if filename.startswith('fig1'):
            for i,s in enumerate(summary):
                observations=[float(r['latency_ms']) for r in rows if r['pipeline']==s['pipeline']]
                # Deterministic horizontal offsets are display-only.
                ax.scatter(i+np.linspace(-.18,.18,len(observations)),observations,s=10,c='#202936',alpha=.4)
        ax.set(xticks=[0,1],xticklabels=labels,ylabel=ylabel,title=title,ylim=(0,max(values)*1.32))
        ax.set_axisbelow(True); ax.grid(axis='y',alpha=.2)
        fig.text(.5,.025,notes,ha='center',fontsize=9,linespacing=1.6)
        fig.tight_layout(rect=(0,.115,1,1))
        fig.savefig(ROOT/filename,dpi=220)
        plt.close(fig)
    runtime,memory_text=(ROOT/'runtime_summary.txt').read_text().split('\nGPU memory is ',1)
    config,_=load_final_config()
    wrap_up(runtime,'GPU memory is '+memory_text,config)
    result=dict(status='PASS',observations=120,images=20,statistics_recomputed_from_saved_samples=True,
        run_order_balanced=True,resident_allocated_memory_constant=True,no_dataset_reads=True,
        no_inference_rerun=True)
    (ROOT/'measurement_verification.json').write_text(json.dumps(result,indent=2)+'\n')
    with (ROOT/'terminal_output.log').open('a') as f:
        f.write('\nSaved-measurement verification PASS: all 120 samples, statistics, balanced run order, and memory peaks verified. Figures and final summary generated from saved artifacts only.\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
