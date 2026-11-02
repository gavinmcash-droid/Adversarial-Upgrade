import inspect
import pytest
import torch
from anc.core import *

torch.set_num_threads(1)

@pytest.fixture
def models():
    torch.manual_seed(42)
    return Keyed(),Keyed(),Eve()

def snapshot(m): return {k:v.clone() for k,v in m.state_dict().items()}
def unchanged(m,s): return all(torch.equal(v,s[k]) for k,v in m.state_dict().items())
def changed(m,s): return not unchanged(m,s)

def test_shapes_mixing_padding_and_counts(models):
    a,b,e=models
    p,k=sample(generator(1,'shapes'),7)
    assert a.fc.weight.shape==(32,32) and e.fc.weight.shape==(32,16)
    # Every output FC feature can depend on every input, before local convolutions.
    x=torch.randn(1,32,requires_grad=True)
    a.fc(x)[0,0].backward()
    assert (x.grad!=0).all()
    for model,inputs,channels,count in [(a,(p,k),BASE,1111),(b,(p,k),BASE,1111),(e,(p,),BASE,599),(Eve(WIDE),(p,),WIDE,717)]:
        shapes=[]
        hooks=[c.register_forward_hook(lambda m,i,o: shapes.append(tuple(o.shape))) for c in model.convs]
        y=model(*inputs)
        for h in hooks: h.remove()
        assert y.shape==(7,16)
        assert shapes==[(7,channels[0],32),(7,channels[1],16),(7,channels[2],16),(7,1,16)]
        assert y.dtype==torch.float32 and (y.abs()<=1).all()
        assert sum(v.numel() for v in model.parameters())==count
    # Explicit SAME pad should be one left/two right for k=4,s=1.
    z=torch.sigmoid(a.fc(torch.cat((p,k),-1))).unsqueeze(1)
    expected=a.convs[0](torch.nn.functional.pad(z,(1,2)))
    assert expected.shape[-1]==32
    assert [c.stride for c in e.convs]==[(1,),(2,),(1,),(1,)]
    assert [c.kernel_size for c in e.convs]==[(4,),(2,),(1,),(1,)]

def test_representation_and_streams():
    seeds=[stream_seed(101,s) for s in ['defender_train','attacker_train','validation','final_test','random_guess']]
    assert len(set(seeds))==5
    batches=[sample(generator(101,s),512) for s in ['defender_train','attacker_train','validation','final_test']]
    for p,k in batches:
        assert set(p.unique().tolist())=={-1.,1.} and set(k.unique().tolist())=={-1.,1.}
        assert not torch.equal(p,k)
        assert k.unique(dim=0).shape[0]>450
    for i in range(4):
        for j in range(i): assert not torch.equal(batches[i][0],batches[j][0])
    p,k=sample(generator(101,'attacker_train'),512)
    assert torch.equal(p,batches[1][0]) and torch.equal(k,batches[1][1])

def test_eve_no_key(models):
    a,b,e=models; p,k=sample(generator(2,'eve'),4)
    assert list(inspect.signature(e.forward).parameters)==['x']
    with pytest.raises(TypeError): e(p,k)
    c=a(p,k).detach(); assert torch.equal(e(c),e(c.clone()))

def test_optimizer_isolation_and_gradients(models):
    a,b,e=models; p,k=sample(generator(3,'optimization'),128)
    sa,sb,se=map(snapshot,models)
    opt=torch.optim.Adam([*a.parameters(),*b.parameters()],lr=.0008)
    ab_step(a,b,e,opt,p,k)
    assert changed(a,sa) and changed(b,sb) and unchanged(e,se)
    for m in [a,b]:
        assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in m.parameters())
        assert any(v.grad.abs().sum()>0 for v in m.parameters())
    sa,sb,se=map(snapshot,models)
    eve_step(a,e,torch.optim.Adam(e.parameters(),lr=.0008),p,k)
    assert unchanged(a,sa) and unchanged(b,sb) and changed(e,se)
    assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in e.parameters())

def test_frozen_fresh_attackers(models):
    a,b,_=models; freeze(a); freeze(b)
    sa,sb=snapshot(a),snapshot(b)
    for channels in [BASE,WIDE]:
        e=Eve(channels); old=snapshot(e); opt=torch.optim.Adam(e.parameters(),lr=.0008)
        g=generator(44,'attacker_train')
        for _ in range(3): eve_step(a,e,opt,*sample(g,64))
        assert changed(e,old) and unchanged(a,sa) and unchanged(b,sb)
        assert all(v.grad is None for m in [a,b] for v in m.parameters())

def test_save_load_predictions(models,tmp_path):
    p,k=sample(generator(5,'roundtrip'),9)
    for m,args in [(models[0],(p,k)),(models[1],(p,k)),(models[2],(p,)),(Eve(WIDE),(p,))]:
        path=tmp_path/'state.pt'; torch.save(m.state_dict(),path)
        fresh=Keyed() if isinstance(m,Keyed) else Eve(tuple(c.out_channels for c in m.convs))
        fresh.load_state_dict(torch.load(path,weights_only=True))
        torch.testing.assert_close(fresh(*args),m(*args),rtol=0,atol=1e-7)

def test_hand_metrics_and_objective():
    p=torch.tensor([[1.,-1.],[1.,1.]])
    pred=torch.tensor([[.5,-.5],[-1.,0.]])
    # Use 16-bit messages by repeating this pair eight times.
    p=p.repeat(1,8); pred=pred.repeat(1,8)
    m=metrics(pred,p)
    assert m=={'normalized_l1':.5,'bit_error_rate':.25,'message_accuracy':.5}
    assert metrics(p,p)=={'normalized_l1':0.,'bit_error_rate':0.,'message_accuracy':1.}
    assert ab_objective(p,torch.zeros_like(p),p)==0
    assert ab_objective(p,p,p)==1
    # Squaring per-example penalty is different from squaring batch mean.
    eve=torch.stack([p[0],-p[1]])
    assert ab_objective(p,eve,p)==1

def test_random_guess_control():
    p,_=sample(generator(6,'truth'),10000)
    guess,_=sample(generator(6,'guess'),10000)
    m=metrics(guess,p)
    # Broad deterministic sanity bound, not a claim that an Eve leaks no information.
    assert abs(m['bit_error_rate']-.5)<.01
    assert m['normalized_l1']==m['bit_error_rate']
