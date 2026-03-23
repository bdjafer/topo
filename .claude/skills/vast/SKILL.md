---
name: vast
description: Manage VAST.AI GPU instances for R-GIN training. Subcommands: search, launch, status, ssh, upload, download, destroy, sweep, nuke.
allowed-tools: Bash, Read, Write, Grep, Glob
user-invocable: true
---

VAST.AI GPU instance manager for topo R-GIN training. Read the API key from the project `.env` file.

## Setup

Before any command, load the API key:

```bash
export VAST_AI_API_KEY=$(grep VAST_AI_API_KEY /Users/bryandjafer/Documents/personal/topo/.env | cut -d= -f2)
```

Ensure the `vastai` CLI is installed. If not:

```bash
pip install vastai 2>/dev/null || pip3 install vastai
```

Set the key for the CLI session:

```bash
vastai set api-key $VAST_AI_API_KEY
```

## Subcommands

Parse the user's argument (passed as `$ARGUMENTS`) to determine which subcommand to run. If no argument is given, run `status`.

### `/vast` or `/vast status`

Show all running instances with cost info:

```bash
vastai show instances
```

Also show current balance:

```bash
vastai show user
```

Report a summary: number of instances, total $/hr burn rate, and a warning if any instances are idle (running but no GPU utilization). Use `vastai execute <ID> 'nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader'` to check GPU util for each running instance.

### `/vast search`

Search for the best RTX 4090 offers:

```bash
vastai search offers 'gpu_name=RTX_4090 num_gpus=1 reliability>0.98 verified=true cpu_ram>=32000 disk_space>=100 inet_down>100' -o 'dph_total'
```

Present the top 5 results in a clean table with: offer ID, price/hr, reliability, location, disk, RAM, max duration.

### `/vast launch`

Launch a new training instance. Steps:

1. Search for cheapest reliable RTX 4090:
```bash
OFFER_ID=$(vastai search offers 'gpu_name=RTX_4090 num_gpus=1 reliability>0.98 verified=true cpu_ram>=32000 disk_space>=100 inet_down>100' -o 'dph_total' --raw | python3 -c "import sys,json; offers=json.load(sys.stdin); print(offers[0]['id']) if offers else print('NONE')")
```

2. If no offers found, report and stop.

3. Show the selected offer (price, location, specs) and ask the user to confirm before creating.

4. Create the instance:
```bash
vastai create instance $OFFER_ID \
  --image vastai/pytorch \
  --disk 100 \
  --ssh --direct \
  --label "topo-rgin-$(date +%Y%m%d-%H%M)" \
  --onstart-cmd 'pip install torch-geometric torch-scatter torch-sparse torch-cluster \
    -f https://data.pyg.org/whl/torch-$(python -c "import torch; print(torch.__version__.split(\"+\")[0])")+cu$(python -c "import torch; print(torch.version.cuda.replace(\".\",\"\"))").html \
    && pip install wandb scipy scikit-learn tqdm pyyaml \
    && env >> /etc/environment'
```

5. Report the instance ID and estimated boot time (~2 min).

### `/vast ssh [INSTANCE_ID]`

Get the SSH connection command for an instance. If no ID given, use the most recent running instance.

```bash
vastai ssh-url <INSTANCE_ID>
```

Print the full SSH command with TensorBoard port forwarding:

```
ssh -p <PORT> root@<IP> -L 6006:localhost:6006
```

### `/vast upload [INSTANCE_ID]`

Upload the training data and code to an instance. If no ID given, use the most recent running instance.

1. Get the SSH connection info:
```bash
vastai ssh-url <INSTANCE_ID>
```

2. Upload the dataset (look for processed graph data):
```bash
# Find the data directory
ls /Users/bryandjafer/Documents/personal/topo/data/
```

3. Upload training packages:
```bash
scp -P <PORT> -r /Users/bryandjafer/Documents/personal/topo/packages/topo-dataset/ root@<IP>:/workspace/topo-dataset/
scp -P <PORT> -r /Users/bryandjafer/Documents/personal/topo/packages/topo-model/ root@<IP>:/workspace/topo-model/
```

4. Report what was uploaded and the install commands to run on the instance.

### `/vast download [INSTANCE_ID]`

Download checkpoints and results from an instance. If no ID given, use the most recent running instance.

```bash
mkdir -p /Users/bryandjafer/Documents/personal/topo/results/$(date +%Y%m%d)
vastai copy <INSTANCE_ID>:/workspace/checkpoints/ local:/Users/bryandjafer/Documents/personal/topo/results/$(date +%Y%m%d)/
```

Report what was downloaded and where.

### `/vast destroy [INSTANCE_ID]`

Destroy a specific instance. If no ID given, ask the user which one.

1. Show the instance details first.
2. **Ask the user to confirm** — this is irreversible and deletes all data.
3. If confirmed:
```bash
vastai destroy instance <INSTANCE_ID>
```
4. Verify destruction:
```bash
vastai show instances
```

### `/vast nuke`

Emergency: destroy ALL running instances. For when you realize you left things running.

1. List all instances.
2. **Ask the user to confirm** — this destroys everything.
3. If confirmed, destroy each instance:
```bash
for id in $(vastai show instances --raw | python3 -c "import sys,json; [print(i['id']) for i in json.load(sys.stdin)]"); do
  vastai destroy instance $id
done
```
4. Verify:
```bash
vastai show instances
```

### `/vast sweep`

Launch multiple interruptible instances for hyperparameter sweep. Ask the user which hyperparameters to sweep, then:

1. Generate the grid of configurations.
2. For each config, find a cheap interruptible 4090:
```bash
vastai search offers 'gpu_name=RTX_4090 num_gpus=1 reliability>0.95' -t bid -o 'dph_total' --raw
```
3. Show the total estimated cost and ask for confirmation.
4. Launch instances with appropriate labels and onstart commands.

### `/vast cost`

Show spending summary:

```bash
vastai show user
vastai show invoices --start_date $(date -v-30d +%Y-%m-%d) --end_date $(date +%Y-%m-%d)
```

Report: current balance, last 30 days spending, current burn rate (from running instances).

## Error Handling

- If `vastai` is not installed, install it with `pip install vastai`.
- If API key is missing from `.env`, tell the user to add `VAST_AI_API_KEY=<key>` to the `.env` file.
- If no offers match the search criteria, suggest relaxing filters (lower reliability, remove verified).
- Always check command exit codes and report errors clearly.

## Safety Rules

- **NEVER** create instances without user confirmation.
- **NEVER** destroy instances without user confirmation.
- **ALWAYS** remind the user to download results before destroying.
- **ALWAYS** show the cost implication of actions ($/hr for launches, total spent for cost checks).
