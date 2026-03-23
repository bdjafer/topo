# VAST.AI Cheatsheet — R-GIN Training for Topo

Operational guide for training the R-GIN model (1.95M params, PyTorch + PyG) on VAST.AI RTX 4090 instances. Covers setup through teardown with exact commands.

---

## 0. Local Prerequisites

```bash
# Install the VAST.AI CLI
pip install vastai

# Set your API key (reads from .env or paste directly)
vastai set api-key $(grep VAST_AI_API_KEY .env | cut -d= -f2)

# Verify
vastai show user

# Add your SSH public key to VAST.AI
cat ~/.ssh/id_ed25519.pub
# Paste at: https://cloud.vast.ai/manage-keys/
# IMPORTANT: Keys only apply to NEW instances created after adding them.
```

---

## 1. Why RTX 4090

Our model is **1.95M parameters**, batch size 32, graphs of ~500-1000 nodes. This is a small model by GPU standards.

| Factor | R-GIN Needs | RTX 4090 Provides |
|---|---|---|
| VRAM | ~4-6 GB (batch of 32 graphs × 1000 nodes × 208d features) | 24 GB GDDR6X |
| Compute | Light — 2-layer GIN, no attention | 16,384 CUDA cores, Ada Lovelace |
| Cost | Budget-conscious research | $0.20-0.45/hr on VAST.AI |
| Training time | ~200 epochs, ~2-6 hours total | Easily handles it |

The 4090 is overkill for this model, which is exactly what we want — headroom for larger batch sizes, debugging, and not having to worry about OOM. An RTX 3090 would also work, but 4090s are plentiful and similarly priced on VAST.AI.

---

## 2. Finding an Instance

### Search Command (Copy-Paste Ready)

```bash
# Best value: reliable, verified, sorted by price
vastai search offers \
  'gpu_name=RTX_4090 num_gpus=1 reliability>0.98 verified=true cpu_ram>=32000 disk_space>=100 inet_down>100' \
  -o 'dph_total'
```

### What to Look For in Results

| Field | Target | Why |
|---|---|---|
| `dph_total` | $0.20–$0.45 | Total hourly cost (GPU + storage) |
| `reliability` | > 0.98 | Machine uptime history. Lower = higher chance of mid-training failure |
| `inet_down` | > 100 Mbps | For uploading dataset and Docker image pull |
| `disk_space` | ≥ 100 GB | Dataset (~5-20 GB) + checkpoints + Docker image |
| `cpu_ram` | ≥ 32 GB | Data loading workers need CPU RAM |
| `duration` | > 86400 (24hr) | Max rental time. Check this! If your training takes 6 hours and the max is 3 hours, you're screwed |

### Interruptible (Cheaper, But Riskier)

```bash
# 50%+ cheaper, but can be interrupted if outbid
vastai search offers \
  'gpu_name=RTX_4090 num_gpus=1 reliability>0.95' \
  -t bid -o 'dph_total'
```

**Use interruptible for**: exploratory runs, hyperparameter sweeps, anything with checkpointing.
**Use on-demand for**: final training run, the one producing the model bundle.

---

## 3. Creating an Instance

### Option A: Quick Start (Onstart Script)

No Docker build needed. Uses VAST.AI's cached PyTorch image + installs PyG at boot:

```bash
OFFER_ID=<from search results>

vastai create instance $OFFER_ID \
  --image vastai/pytorch \
  --disk 100 \
  --ssh --direct \
  --label "topo-rgin-training" \
  --onstart-cmd 'pip install torch-geometric torch-scatter torch-sparse torch-cluster \
    -f https://data.pyg.org/whl/torch-$(python -c "import torch; print(torch.__version__.split(\"+\")[0])")+cu$(python -c "import torch; print(torch.version.cuda.replace(\".\",\"\"))").html \
    && pip install wandb scipy scikit-learn tqdm pyyaml \
    && env >> /etc/environment'
```

**Tradeoff**: Adds 3-5 min to startup (pip install each time). Fine for a single run.

### Option B: Custom Docker Image (Recommended for Repeated Use)

Create `docker/Dockerfile.train`:

```dockerfile
FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

RUN pip install --no-cache-dir \
    torch-geometric \
    torch-scatter \
    torch-sparse \
    torch-cluster \
    -f https://data.pyg.org/whl/torch-2.4.0+cu124.html

RUN pip install --no-cache-dir \
    wandb \
    scipy \
    scikit-learn \
    tqdm \
    pyyaml \
    tensorboard

WORKDIR /workspace
```

Build and push:

```bash
docker build -t <your-dockerhub>/topo-train:latest -f docker/Dockerfile.train .
docker push <your-dockerhub>/topo-train:latest
```

Then:

```bash
vastai create instance $OFFER_ID \
  --image <your-dockerhub>/topo-train:latest \
  --disk 100 \
  --ssh --direct \
  --label "topo-rgin-training" \
  --onstart-cmd 'env >> /etc/environment'
```

### Option C: Interruptible with Bid Price

```bash
vastai create instance $OFFER_ID \
  --image <your-dockerhub>/topo-train:latest \
  --disk 100 \
  --ssh --direct \
  --price 0.25 \
  --label "topo-rgin-sweep-1"
```

### After Creation

The CLI returns an instance ID. Wait 1-2 minutes for it to boot:

```bash
# Check status (wait for "running")
vastai show instances

# Get SSH connection string
vastai ssh-url <INSTANCE_ID>
```

---

## 4. Connecting & Uploading Data

### SSH Into the Instance

```bash
# The ssh-url command gives you the full connection string
ssh -p <PORT> root@<IP> -L 6006:localhost:6006
```

The `-L 6006:localhost:6006` forwards TensorBoard.

### Upload the Training Dataset

From your local machine:

```bash
# Upload processed graph NPZs (from Step 1 output)
scp -P <PORT> -r ./data/processed_graphs/ root@<IP>:/workspace/data/

# Or use vastai copy
vastai copy local:./data/processed_graphs/ <INSTANCE_ID>:/workspace/data/
```

### Upload the Training Code

```bash
# Clone the repo on the instance
ssh -p <PORT> root@<IP> 'cd /workspace && git clone https://github.com/<you>/topo.git'

# Or scp just the training package
scp -P <PORT> -r ./packages/topo-model/ root@<IP>:/workspace/topo-model/
scp -P <PORT> -r ./packages/topo-dataset/ root@<IP>:/workspace/topo-dataset/
```

### Verify GPU Works

```bash
ssh -p <PORT> root@<IP>

nvidia-smi
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()} — {torch.cuda.get_device_name(0)}')
print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
import torch_geometric
print(f'PyG: {torch_geometric.__version__}')
"
```

---

## 5. Running Training

### ALWAYS Use tmux

If your SSH drops, foreground processes die. VAST.AI instances usually start in tmux by default.

```bash
# Create a named session
tmux new -s train

# Inside tmux, run training:
cd /workspace/topo
python -m topo_model.train \
  --data-dir /workspace/data \
  --checkpoint-dir /workspace/checkpoints \
  --epochs 200 \
  --batch-size 32 \
  --lr 1e-3 \
  --save-every 10 \
  --wandb-project topo-rgin

# Detach: Ctrl+B, then D
# Reattach later: tmux attach -t train
```

### Training Duration Estimate

| Dataset Size | Epochs | Est. Time (1x 4090) | Est. Cost (interruptible) |
|---|---|---|---|
| 50 graphs (~50K nodes) | 200 | ~1-2 hours | $0.25-$0.50 |
| 200 graphs (~200K nodes) | 200 | ~3-6 hours | $0.70-$1.50 |
| 500 graphs (~500K nodes) | 200 | ~8-15 hours | $1.80-$3.50 |

These are rough estimates. The R-GIN is small (1.95M params) so training is fast.

### Monitoring

```bash
# From another terminal with port forwarding:
# http://localhost:6006 for TensorBoard

# Or check GPU utilization remotely
vastai execute <INSTANCE_ID> 'nvidia-smi'

# Or check training logs
vastai logs <INSTANCE_ID> --tail 200
```

---

## 6. Checkpointing Strategy (Critical for Interruptible)

The training code MUST checkpoint. This is non-negotiable for interruptible instances.

```python
# Save every 10 epochs + best model
CHECKPOINT_DIR = "/workspace/checkpoints"

if epoch % 10 == 0 or val_metric > best_val_metric:
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "val_metric": val_metric,
        "config": config,
    }, f"{CHECKPOINT_DIR}/checkpoint_epoch_{epoch:04d}.pt")

if val_metric > best_val_metric:
    best_val_metric = val_metric
    torch.save(same_dict, f"{CHECKPOINT_DIR}/best_model.pt")
```

### Resume from Checkpoint

```python
# At training start, check for existing checkpoint
ckpt_path = f"{CHECKPOINT_DIR}/best_model.pt"
if os.path.exists(ckpt_path):
    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    start_epoch = ckpt["epoch"] + 1
    best_val_metric = ckpt["val_metric"]
    print(f"Resumed from epoch {ckpt['epoch']}, val_metric={best_val_metric:.4f}")
```

---

## 7. Retrieving Results

### Download Checkpoints and Artifacts

```bash
# From local machine
vastai copy <INSTANCE_ID>:/workspace/checkpoints/ local:./results/run-N/

# Or with scp
scp -P <PORT> -r root@<IP>:/workspace/checkpoints/ ./results/run-N/
```

### The Final Model Bundle (from STEP_2 spec)

After training completes, the instance should contain:

```
/workspace/checkpoints/
  best_model.pt           # PyTorch checkpoint
  rgin.onnx               # ONNX export (if successful)
  R.npy                    # 32x32 bilinear matrix
  depth_probe_w.npy        # 768d weight vector
  depth_probe_b.npy        # scalar bias
  config.json              # Hyperparameters
  metadata.json            # Training info
  node_type_vocab.json     # Frozen vocabulary
```

Download all of it before destroying.

---

## 8. Shutdown — Don't Explode Cost

### The #1 Rule: DESTROY When Done

**Stopping is NOT enough.** Stopped instances still charge for storage.

```bash
# STOP = pauses GPU billing, storage STILL charges (~$0.01-0.05/hr depending on disk)
vastai stop instance <INSTANCE_ID>

# DESTROY = stops ALL billing. Data is gone forever.
vastai destroy instance <INSTANCE_ID>
```

### Daily Hygiene

```bash
# Check what's running (run this EVERY DAY during training)
vastai show instances

# If anything is sitting idle, destroy it
vastai destroy instance <ID>

# Check your balance
vastai show user

# Check spending
vastai show invoices --start_date 2026-03-01 --end_date 2026-03-31
```

### Cost Control Checklist

- [ ] **Download results BEFORE destroying** — once destroyed, data is gone
- [ ] **Don't over-provision disk** — 100 GB is plenty for R-GIN training. Don't request 500 GB
- [ ] **Set a credit limit** — Only add as much credit as you're willing to spend
- [ ] **Use interruptible for sweeps** — 50%+ cheaper, and checkpointing handles interruptions
- [ ] **Check `duration` field** — Ensure max rental time exceeds your expected training time
- [ ] **Label your instances** — So you know what each one is for: `--label "topo-rgin-run-7"`
- [ ] **Don't leave instances running overnight** unless actively training

### Worst Case Scenario

Forgot a 1x RTX 4090 running for a week:
- On-demand: ~$0.40/hr × 168 hrs = **~$67**
- Plus storage: ~$0.03/hr × 168 hrs = **~$5**
- Total: **~$72** (painful but not catastrophic)

---

## 9. Parallelizing Hyperparameter Sweeps

For the R-GIN, the hyperparameters to sweep (per STEP_2 spec):

| Parameter | Range | Priority |
|---|---|---|
| Learning rate | 5e-4, 1e-3, 2e-3 | High |
| Hidden dim | 128, 256 | Medium |
| Masking ratio | 0.60, 0.65, 0.70 | Medium |
| Dropout | 0.05, 0.10, 0.20 | Low |

### Approach: Multiple Single-GPU Instances

```bash
# Launch one cheap 4090 per trial
for lr in 5e-4 1e-3 2e-3; do
  for hidden in 128 256; do
    OFFER_ID=$(vastai search offers 'gpu_name=RTX_4090 num_gpus=1 reliability>0.95' \
      -t bid -o 'dph_total' --raw | python3 -c "import sys,json; print(json.load(sys.stdin)[0]['id'])")

    vastai create instance $OFFER_ID \
      --image <your-dockerhub>/topo-train:latest \
      --disk 50 --ssh --direct --price 0.25 \
      --label "sweep-lr${lr}-h${hidden}" \
      --onstart-cmd "cd /workspace && git clone <repo> && \
        python -m topo_model.train --lr $lr --hidden-dim $hidden \
        --wandb-project topo-rgin-sweep --wandb-name lr${lr}-h${hidden}"
  done
done
```

**Cost**: 6 trials × ~2 hrs × $0.25/hr = **~$3 total** for a full sweep.

### Alternative: Single 4-GPU Instance

```bash
# Rent one 4x 4090 machine, run 4 trials in parallel
vastai search offers 'gpu_name=RTX_4090 num_gpus=4 reliability>0.95' -o 'dph_total'

# Then on the instance:
CUDA_VISIBLE_DEVICES=0 python train.py --lr 5e-4 --hidden 256 &
CUDA_VISIBLE_DEVICES=1 python train.py --lr 1e-3 --hidden 256 &
CUDA_VISIBLE_DEVICES=2 python train.py --lr 2e-3 --hidden 256 &
CUDA_VISIBLE_DEVICES=3 python train.py --lr 1e-3 --hidden 128 &
wait
```

---

## 10. API Reference (For Scripting)

All endpoints use `https://console.vast.ai/api/v0/` and Bearer auth.

### Search Offers

```bash
curl -X POST "https://console.vast.ai/api/v0/bundles/" \
  -H "Authorization: Bearer $VAST_AI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "limit": 10,
    "type": "ondemand",
    "verified": {"eq": true},
    "rentable": {"eq": true},
    "gpu_name": {"in": ["RTX_4090"]},
    "num_gpus": {"gte": 1},
    "reliability": {"gte": 0.98},
    "dph_total": {"lte": 0.50},
    "disk_space": {"gte": 100},
    "order": [["dph_total", "asc"]]
  }'
```

### Create Instance

```bash
curl -X PUT "https://console.vast.ai/api/v0/asks/$OFFER_ID/" \
  -H "Authorization: Bearer $VAST_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "image": "pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime",
    "disk": 100,
    "runtype": "ssh_direct",
    "onstart": "pip install torch-geometric && env >> /etc/environment"
  }'
```

### List Instances

```bash
curl -H "Authorization: Bearer $VAST_AI_API_KEY" \
  "https://console.vast.ai/api/v0/instances/"
```

### Destroy Instance

```bash
curl -X DELETE "https://console.vast.ai/api/v0/instances/$INSTANCE_ID/" \
  -H "Authorization: Bearer $VAST_AI_API_KEY"
```

### Pro Tip: `--curl` Flag

Any CLI command + `--curl` shows the equivalent REST call:

```bash
vastai search offers 'gpu_name=RTX_4090' --curl
vastai create instance 12345 --image pytorch/pytorch --disk 50 --curl
```

---

## 11. Quick Reference Card

| Task | Command |
|---|---|
| Install CLI | `pip install vastai` |
| Set API key | `vastai set api-key KEY` |
| Search 4090s | `vastai search offers 'gpu_name=RTX_4090 reliability>0.98' -o 'dph_total'` |
| Create instance | `vastai create instance OFFER_ID --image vastai/pytorch --disk 100 --ssh --direct` |
| List instances | `vastai show instances` |
| SSH connect | `ssh -p PORT root@IP` |
| Upload files | `scp -P PORT -r ./data root@IP:/workspace/` |
| Download files | `scp -P PORT root@IP:/workspace/checkpoints/ ./` |
| Execute remote | `vastai execute ID 'nvidia-smi'` |
| View logs | `vastai logs ID --tail 200` |
| Stop (GPU off, storage charged) | `vastai stop instance ID` |
| **Destroy (all billing stops)** | **`vastai destroy instance ID`** |
| Check balance | `vastai show user` |
| Check spending | `vastai show invoices -s 2026-03-01 -e 2026-03-31` |

---

## 12. Typical Training Session (End-to-End)

```bash
# === LOCAL MACHINE ===

# 1. Find a GPU
vastai search offers 'gpu_name=RTX_4090 num_gpus=1 reliability>0.98 cpu_ram>=32000 disk_space>=100' -o 'dph_total'
# Note the OFFER_ID (first column)

# 2. Create instance
vastai create instance <OFFER_ID> --image vastai/pytorch --disk 100 --ssh --direct \
  --label "topo-rgin-run-1" \
  --onstart-cmd 'pip install torch-geometric torch-scatter torch-sparse torch-cluster \
    -f https://data.pyg.org/whl/torch-$(python -c "import torch; print(torch.__version__.split(\"+\")[0])")+cu$(python -c "import torch; print(torch.version.cuda.replace(\".\",\"\"))").html \
    && pip install wandb scipy scikit-learn tqdm pyyaml && env >> /etc/environment'
# Note the INSTANCE_ID

# 3. Wait for boot (~2 min), then get SSH info
vastai show instances
vastai ssh-url <INSTANCE_ID>

# 4. Upload dataset
scp -P <PORT> -r ./data/processed_graphs/ root@<IP>:/workspace/data/

# 5. Upload training code
scp -P <PORT> -r ./packages/topo-model/ root@<IP>:/workspace/topo-model/
scp -P <PORT> -r ./packages/topo-dataset/ root@<IP>:/workspace/topo-dataset/

# === ON THE INSTANCE ===

# 6. SSH in
ssh -p <PORT> root@<IP> -L 6006:localhost:6006

# 7. Verify
nvidia-smi
python -c "import torch; print(torch.cuda.get_device_name(0))"
python -c "import torch_geometric; print('PyG OK')"

# 8. Install local packages
cd /workspace
pip install -e topo-dataset/ -e topo-model/

# 9. Train in tmux
tmux new -s train
python -m topo_model.train \
  --data-dir /workspace/data \
  --checkpoint-dir /workspace/checkpoints \
  --epochs 200 --batch-size 32 --lr 1e-3 --save-every 10
# Ctrl+B, D to detach

# === LOCAL MACHINE (when training is done) ===

# 10. Download results
vastai copy <INSTANCE_ID>:/workspace/checkpoints/ local:./results/run-1/

# 11. DESTROY the instance
vastai destroy instance <INSTANCE_ID>

# 12. Verify nothing is left running
vastai show instances
```

**Total estimated cost for one full training run: $1-5 depending on dataset size and duration.**
