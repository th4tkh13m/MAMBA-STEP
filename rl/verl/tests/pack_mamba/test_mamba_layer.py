import torch
import itertools
import copy

from verl.models.mamba.hybrid_wrapper import MambaTransformerHybridModelWrapper

class AlignTimer:
    def __init__(self, message='kernel_no_name'):
        self.message = message

    def __enter__(self):
        torch.cuda.synchronize()  
        self.starter = torch.cuda.Event(enable_timing=True)
        self.starter.record()
        return self

    def __exit__(self, type, value, traceback):
        self.ender = torch.cuda.Event(enable_timing=True)
        self.ender.record()
        torch.cuda.synchronize()  
        self.time = self.starter.elapsed_time(self.ender)
        print('{} uses time {:.4f} ms'.format(self.message, self.time))


def unpack(packed_hidden_states, cu_seqlens):
    batch_size = packed_hidden_states.shape[0]
    package_num = cu_seqlens.shape[0] - 1
    seq_len = (cu_seqlens[1:] - cu_seqlens[:-1]).max()
    hidden_dim = packed_hidden_states.shape[2]
    hidden_states = torch.zeros(package_num * batch_size, seq_len, hidden_dim, dtype=packed_hidden_states.dtype, device=packed_hidden_states.device)
    for j in range(batch_size):
        for i in range(package_num):
            line = j * package_num + i
            hidden_states[line, : cu_seqlens[i + 1] - cu_seqlens[i], :] = packed_hidden_states[j, cu_seqlens[i] : cu_seqlens[i + 1], :]
    return hidden_states


def pack(hidden_states, cu_seqlens, batch_size):
    package_num, seq_len, hidden_dim = hidden_states.shape
    seq_len_list = cu_seqlens[1:] - cu_seqlens[:-1]
    seq_len_list_3d = seq_len_list.unsqueeze(1).unsqueeze(2)
    indices_3d = (
        torch.arange(seq_len, device=hidden_states.device)
        .unsqueeze(0)
        .unsqueeze(2)
        .repeat(package_num, 1, hidden_dim)
    )
    mask_3d = indices_3d < seq_len_list_3d.repeat(batch_size, 1, 1)
    packed_hidden_states = hidden_states[mask_3d].view(batch_size,-1, hidden_dim)
    return packed_hidden_states


def generate_random_cu_seqlens(seq_len, packages_num=2):
    if packages_num < 1:
        raise ValueError("packages_num must be at least 1")
    
    # base size of each chunk, and how many get an extra token
    base, rem = divmod(seq_len, packages_num)
    # lengths: e.g. for seq_len=10, packages=3 → [4,3,3]
    lengths = [base + 1 if i < rem else base for i in range(packages_num)]
    
    # split points exclude the final cumulative (seq_len)
    split_points = list(itertools.accumulate(lengths))[:-1]
    
    # cu_seqlens = [0] + split_points + [seq_len]
    cu_seqlens = [0] + split_points + [seq_len]
    
    # index: for each chunk, we emit 0,1,...,length-1
    index = []
    for length in lengths:
        index.extend(range(length))
    
    # sanity check
    assert len(cu_seqlens) - 1 == packages_num
    assert sum(lengths) == seq_len
    assert len(index) == seq_len
    
    return cu_seqlens, index


def test_mamba_block(seq_len = 32768, batch_size = 1, packages_num = 8, layer_id=0):

    pretrained_model_name = "JunxiongWang/M1-3B"  # change as desired
    itype = torch.bfloat16

    # test for mamba
    mamba_model = MambaTransformerHybridModelWrapper.from_pretrained(pretrained_model_name, torch_dtype=itype)
    mamba = mamba_model.model.model.layers[layer_id].mamba
    mamba_ref_model = MambaTransformerHybridModelWrapper.from_pretrained(pretrained_model_name, torch_dtype=itype)
    mamba_ref = mamba_ref_model.model.model.layers[layer_id].mamba
    mamba_ref.load_state_dict(copy.deepcopy(mamba.state_dict()))
    hidden_dim = mamba_model.config.hidden_size

    # config tested with A100
    device='cuda'
    rtol, atol = (6e-4, 2e-3) if itype == torch.float32 else (3e-3, 5e-3)
    if itype == torch.bfloat16:
        rtol, atol = 3e-2, 5e-2
    rtolw, atolw = (1e-3, 1e-3)
    # If we have z, the errors on the weights seem higher
    rtolw = max(rtolw, rtol)
    atolw = max(atolw, atol)
    # Generate random cu_seqlens for testing
    cu_seqlens, index = generate_random_cu_seqlens(seq_len, packages_num = packages_num)
    cu_seqlens = torch.tensor(cu_seqlens).cuda()
    index = torch.tensor(index, dtype=torch.int32).unsqueeze(0).expand(batch_size, -1).contiguous().cuda()

    # Generate packed_hidden_states with random values for testing
    hidden_states_list = [torch.randn(l, hidden_dim, device=device, dtype=itype, requires_grad=True) for l in (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()]
    packed_hidden_states = torch.cat(hidden_states_list, dim=0).unsqueeze(0)
    packed_hidden_states = packed_hidden_states.expand(batch_size, -1, -1).contiguous()
    # hidden_states should be forwarded without cu_seqlens
    hidden_states = unpack(packed_hidden_states, cu_seqlens)

    # Check: sum of seq_len of item in hidden_states_list should be equal to seq_len of packed_hidden_states
    assert sum([hs.shape[0] for hs in hidden_states_list]) == packed_hidden_states.shape[1]
    # Check: max of seq_len of item in hidden_states_list should be equal to seq_len of hidden_states
    assert max([hs.shape[0] for hs in hidden_states_list]) == hidden_states.shape[1]

    # reference output for forwardding hidden_states
    with AlignTimer("pack_fwd"):
        print("packed_hidden_states shape:", packed_hidden_states.shape)
        out = mamba(packed_hidden_states.detach().clone(), position_ids=index)

    with AlignTimer("unpack_fwd"):
        out_ref = mamba_ref(hidden_states.detach().clone())

    out_ref_pack = pack(out_ref, cu_seqlens, batch_size)
    
    diff = (out - out_ref_pack).abs()
    print(f'Output max diff: {diff.max().item()}')
    print(f'Output mean diff: {diff.mean().item()}')

    g = torch.randn(out.shape).to(device)  
    with AlignTimer("pack_bwd"):
        out.backward(g)
    gradients = {name: param.grad.clone() for name, param in mamba.named_parameters() if param.grad is not None}

    g_ref = unpack(g, cu_seqlens)
    with AlignTimer("unpack_bwd"):
        out_ref.backward(g_ref)
    gradients_ref = {name: param.grad.clone() for name, param in mamba_ref.named_parameters() if param.grad is not None}
    
    for name in gradients_ref:
        if name in gradients:
            is_equal = torch.allclose(gradients_ref[name], gradients[name], rtol=rtol, atol=atol)
            print(f"Gradients for {name} are {'equal' if is_equal else 'not equal'}")
            if not is_equal:
                print(f"Gradient difference for {name}: {torch.abs(gradients_ref[name] - gradients[name]).max()}")
                print(f"Gradient difference for {name}: {torch.abs(gradients_ref[name] - gradients[name]).mean()}")
        else:
            print(f"Parameter {name} not found in the second set of gradients")

if __name__ == "__main__":

    for layer_id in [0, 1, 2, 4, 5, 6, 7, 9, 10, 11, 12, 14, 15, 16, 17, 19, 20, 21, 22, 24, 25, 26]:
        test_mamba_block(seq_len = 4096, batch_size = 1, packages_num = 4, layer_id=layer_id)