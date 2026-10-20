"""Run with torchrun inside the production training image; no model downloads."""
from datetime import timedelta
import json
import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel


def main():
    rank = int(os.environ['LOCAL_RANK'])
    assert int(os.environ['WORLD_SIZE']) == 2
    assert torch.cuda.device_count() == 2
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl', timeout=timedelta(seconds=90))
    try:
        value = torch.tensor([rank + 1.0], device=f'cuda:{rank}')
        dist.all_reduce(value)
        assert value.item() == 3.0
        torch.manual_seed(42)
        model = DistributedDataParallel(torch.nn.Linear(8, 1).cuda(rank), device_ids=[rank])
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        before = model.module.weight.detach().clone()
        inputs = torch.ones((4, 8), device=f'cuda:{rank}') * (rank + 1)
        loss = model(inputs).square().mean()
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        assert torch.isfinite(loss).item()
        assert not torch.equal(before, model.module.weight)
        weights = model.module.weight.detach()
        copies = [torch.empty_like(weights) for _ in range(2)]
        dist.all_gather(copies, weights)
        torch.testing.assert_close(copies[0], copies[1])
        if rank == 0:
            print(json.dumps({'passed': True, 'nccl_all_reduce': True,
                              'ddp_training_step': True, 'synchronized_weights': True}), flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
