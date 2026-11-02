"""CLI: python -m anc.run {defender,attack,evaluate,plot,demo}."""
import argparse
import hashlib
import json
import platform
from pathlib import Path
import subprocess
import time
import traceback
import torch
from .core import Keyed, Eve, BASE, sample, generator, stream_seed, metrics, errors, freeze, ab_step, eve_step

ROOT = Path(__file__).resolve().parents[1]

def write(path, obj):
    path.write_text(json.dumps(obj, indent=2)+'\n')

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def setup(cfg):
    torch.set_num_threads(cfg['threads'])
    torch.use_deterministic_algorithms(True)

def metadata(cfg, seed, stage):
    files = [*ROOT.glob('anc/*.py'), *ROOT.glob('configs/*.json'), *ROOT.glob('tests/*.py')]
    return {'config':cfg, 'seed':seed, 'stage':stage, 'python':platform.python_version(),
            'torch':str(torch.__version__), 'platform':platform.platform(), 'device':'cpu',
            'cuda_available':torch.cuda.is_available(), 'source_hashes':{str(p.relative_to(ROOT)):digest(p) for p in files},
            'stream_seeds':{label:stream_seed(seed,label) for label in ['defender_train','attacker_train','validation','final_test','random_guess']},
            'parameter_counts':{'alice':sum(p.numel() for p in Keyed().parameters()),'bob':sum(p.numel() for p in Keyed().parameters()),'baseline_eve':sum(p.numel() for p in Eve(cfg['baseline_channels']).parameters()),'wider_eve':sum(p.numel() for p in Eve(cfg['wider_channels']).parameters())},
            'reference_revision':'30efea2b4d170c862703ff16f0fd5ad4ab76ddc7',
            'paper':'https://arxiv.org/abs/1610.06918v1',
            'dependencies':subprocess.check_output([__import__('sys').executable,'-m','pip','freeze'],text=True).splitlines()}

def create_run(path,cfg,seed,stage):
    path.mkdir(parents=True,exist_ok=False)
    write(path/'metadata.json', metadata(cfg,seed,stage))
    write(path/'status.json', {'status':'running'})
    # Keep exact executable sources even if the working directory changes later.
    snapshot = path/'source'; snapshot.mkdir()
    for p in ROOT.glob('anc/*.py'): (snapshot/p.name).write_bytes(p.read_bytes())

def log(path, row):
    with (path/'history.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')

def save(path, a,b,e,step,optimizer=None):
    torch.save({'alice':a.state_dict(),'bob':b.state_dict(),'eve':e.state_dict(),
                'step':step,'optimizer':optimizer.state_dict() if optimizer else None},path)

def load_defender(path):
    state = torch.load(path/'best.pt',weights_only=True)
    a,b = Keyed(),Keyed()
    a.load_state_dict(state['alice']); b.load_state_dict(state['bob'])
    freeze(a); freeze(b)
    return a,b

def defender(path,cfg,seed):
    create_run(path,cfg,seed,'defender')
    start = time.perf_counter()
    try:
        torch.manual_seed(seed)
        a,b,e = Keyed(),Keyed(),Eve()
        ab = torch.optim.Adam([*a.parameters(),*b.parameters()],lr=cfg['learning_rate'])
        eo = torch.optim.Adam(e.parameters(),lr=cfg['learning_rate'])
        g = generator(seed,'defender_train')
        vp,vk = sample(generator(seed,'validation'),cfg['validation_size'])
        best = float('inf'); step = 0
        for step in range(cfg['defender_updates']+1):
            if step:
                ab_step(a,b,e,ab,*sample(g,cfg['batch_size']))
                for _ in range(2): eve_step(a,e,eo,*sample(g,cfg['batch_size']))
            stop = time.perf_counter()-start >= cfg['defender_seconds']
            if step % cfg['validate_every']==0 or step==cfg['defender_updates'] or stop:
                with torch.no_grad():
                    c = a(vp,vk); bm=metrics(b(c,vk),vp); em=metrics(e(c),vp)
                log(path,{'step':step,'seconds':time.perf_counter()-start,'bob':bm,'training_eve':em})
                # Predeclared selection: Bob validation L1 alone. Never fresh Eve or test.
                if bm['normalized_l1'] < best:
                    best=bm['normalized_l1']; save(path/'best.pt',a,b,e,step,ab)
            if stop: break
        save(path/'last.pt',a,b,e,step,ab)
        torch.save({'eve_optimizer':eo.state_dict(),'train_rng':g.get_state()},path/'training_state.pt')
        write(path/'status.json',{'status':'complete' if step==cfg['defender_updates'] else 'time_limit',
                                'updates':step,'eve_updates':2*step,'seconds':time.perf_counter()-start})
    except BaseException:
        write(path/'status.json',{'status':'failed','traceback':traceback.format_exc(),'seconds':time.perf_counter()-start})
        raise

def attack(path,parent,cfg,seed,channels):
    create_run(path,cfg,seed,'attack')
    write(path/'parent.json',{'path':str(parent.resolve()),'checkpoint_sha256':digest(parent/'best.pt'),'channels':channels,
                             'data_stream_seed':stream_seed(json.loads((parent/'metadata.json').read_text())['seed'],'attacker_train')})
    start=time.perf_counter()
    try:
        parent_seed=json.loads((parent/'metadata.json').read_text())['seed']
        a,b=load_defender(parent)
        original={k:v.clone() for k,v in a.state_dict().items()}
        torch.manual_seed(seed); e=Eve(channels)
        opt=torch.optim.Adam(e.parameters(),lr=cfg['learning_rate'])
        # Data independent of initialization and architecture; exact same batches for all Eves.
        g=generator(parent_seed,'attacker_train')
        vp,vk=sample(generator(parent_seed,'validation'),cfg['validation_size'])
        with torch.no_grad(): vc=a(vp,vk)
        best=float('inf'); step=0
        for step in range(cfg['attacker_updates']+1):
            if step: eve_step(a,e,opt,*sample(g,cfg['batch_size']))
            stop=time.perf_counter()-start>=cfg['attacker_seconds']
            if step%cfg['validate_every']==0 or step==cfg['attacker_updates'] or stop:
                with torch.no_grad(): em=metrics(e(vc),vp)
                log(path,{'step':step,'seconds':time.perf_counter()-start,'eve':em})
                if em['normalized_l1']<best:
                    best=em['normalized_l1']
                    torch.save({'eve':e.state_dict(),'step':step,'optimizer':opt.state_dict()},path/'best.pt')
            if stop: break
        assert all(torch.equal(v,original[k]) for k,v in a.state_dict().items())
        torch.save({'eve':e.state_dict(),'step':step,'optimizer':opt.state_dict(),'train_rng':g.get_state()},path/'last.pt')
        write(path/'status.json',{'status':'complete' if step==cfg['attacker_updates'] else 'time_limit',
                                'updates':step,'seconds':time.perf_counter()-start,
                                'parameters':sum(p.numel() for p in e.parameters())})
    except BaseException:
        write(path/'status.json',{'status':'failed','traceback':traceback.format_exc(),'seconds':time.perf_counter()-start})
        raise

def attackers(parent):
    for path in sorted(parent.glob('attack_*')):
        meta=json.loads((path/'parent.json').read_text())
        assert meta['checkpoint_sha256']==digest(parent/'best.pt'), 'Defender changed'
        status=json.loads((path/'status.json').read_text())
        if status['status']!='complete': raise ValueError(f'Incomplete attacker: {path}')
        e=Eve(meta['channels']); e.load_state_dict(torch.load(path/'best.pt',weights_only=True)['eve']); e.eval()
        yield path,e,status

def evaluate(parent,cfg):
    output=parent/'test_results.json'
    if output.exists(): raise FileExistsError('Final test already evaluated; preserved without rerunning')
    seed=json.loads((parent/'metadata.json').read_text())['seed']
    p,k=sample(generator(seed,'final_test'),cfg['test_size'])
    a,b=load_defender(parent)
    loaded=list(attackers(parent))
    expected=2*len(cfg['attacker_seeds'])
    if len(loaded)!=expected: raise ValueError(f'Expected {expected} fresh attackers')
    if len({s['updates'] for _,_,s in loaded})!=1: raise ValueError('Unequal attacker budgets')
    with torch.no_grad():
        c=a(p,k); result={'bob':metrics(b(c,k),p)}
        for path,e,_ in loaded: result[path.name]=metrics(e(c),p)
        guess,_=sample(generator(seed,'random_guess'),len(p))
        result['random_guess']=metrics(guess,p)
    write(output,{'metrics':result,'examples':len(p),'test_seed':stream_seed(seed,'final_test'),
                  'checkpoint_hashes':{str(path.name):digest(path/'best.pt') for path,_,_ in loaded},
                  'defender_hash':digest(parent/'best.pt')})
    print(json.dumps(result,indent=2))

def plot(parent):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    paths=[parent,*sorted(parent.glob('attack_*'))]
    fig,axes=plt.subplots(1,3,figsize=(15,4))
    for path in paths:
        rows=[json.loads(x) for x in (path/'history.jsonl').read_text().splitlines()]
        keys=['bob','training_eve'] if path==parent else ['eve']
        for key in keys:
            for ax,metric in zip(axes,['normalized_l1','bit_error_rate','message_accuracy']):
                ax.plot([r['step'] for r in rows],[r[key][metric] for r in rows],label=f'{path.name}/{key}')
                ax.set(xlabel='Updates (defender: AB cycles)',ylabel=metric,title='Validation only')
    axes[0].legend(fontsize=6); fig.tight_layout(); fig.savefig(parent/'validation.png',dpi=150); plt.close(fig)

def demo(parent):
    a,b=load_defender(parent); p,k=sample(generator(999,'demonstration'),3)
    def bits(x): return ''.join('1' if v>=0 else '0' for v in x)
    with torch.no_grad():
        c=a(p,k); predictions={'Bob':b(c,k)}
        predictions.update({path.name:e(c) for path,e,_ in attackers(parent)})
    print('ILLUSTRATIVE EXAMPLES ONLY; not aggregate evidence. Predictions thresholded at zero.')
    for i in range(len(p)):
        print('Plaintext',bits(p[i]))
        for name,pred in predictions.items(): print(name,bits(pred[i]))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['defender','attack','evaluate','plot','demo'])
    parser.add_argument('--config',required=True,type=Path)
    parser.add_argument('--root',required=True,type=Path)
    args=parser.parse_args(); cfg=json.loads(args.config.read_text()); setup(cfg)
    for seed in cfg['defender_seeds']:
        parent=args.root/f'defender_{seed}'
        if args.command!='defender':
            saved_cfg=json.loads((parent/'metadata.json').read_text())['config']
            if cfg!=saved_cfg: raise ValueError('Config differs from saved defender configuration')
        if args.command=='defender': defender(parent,cfg,seed)
        elif args.command=='attack':
            for init in cfg['attacker_seeds']:
                for name in ['baseline','wider']:
                    attack(parent/f'attack_{name}_{init}',parent,cfg,init,cfg[name+'_channels'])
        elif args.command=='evaluate': evaluate(parent,cfg)
        elif args.command=='plot': plot(parent)
        else: demo(parent)

if __name__=='__main__': main()
