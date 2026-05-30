# RTX 3090 Tuning (PRD §23)

The miner changes **no** clocks or power unless you set `gpu.apply_overclock = true`. The presets below
are applied with standard tools (`nvidia-smi`, `nvidia-settings`/MSI Afterburner) — the miner only
*reads* telemetry via NVML and throttles on temperature.

RTX 3090 notes specific to this workload:
- The PoUW kernel is **int8 tensor-core + BLAKE3**, compute- and SMEM-bound, not memory-bandwidth-bound
  like Ethash. Memory **overclock** helps less than on hashing coins; the bigger lever is core/SM clock,
  tensor-core occupancy, and the `sm_86` SMEM-tile fit (`docs/cuda-sm86-port.md` §2).
- GDDR6X **VRAM temperature** is the real hazard (PRD §25.4). Watch `vram_temp_c`, not just core temp.

## Presets

### Safe (stability first)
```bash
sudo nvidia-smi -pm 1
sudo nvidia-smi -pl 280                 # power cap 280 W
# conservative clocks; target low temps
sudo nvidia-smi -lgc 0,1600             # lock SM clock ceiling ~1600 MHz
```
`max_temp_c = 75`, `max_vram_temp_c = 94`. Goal: 24/7 stability.

### Balanced (best efficiency — recommended)
```bash
sudo nvidia-smi -pm 1
sudo nvidia-smi -pl 320                 # 320 W
sudo nvidia-smi -lgc 0,1750
```
`max_temp_c = 78`, `max_vram_temp_c = 96`. Goal: best GMAC/W. This is the `config/miner.example.toml`
default envelope.

### Max performance
```bash
sudo nvidia-smi -pl 370                 # up to 350–370 W (PSU + cooling permitting)
sudo nvidia-smi -lgc 0,1900
```
`max_temp_c = 80`, `max_vram_temp_c = 98`. Goal: highest TH/s. Only with strong airflow and a cooled
backplate; expect diminishing returns past ~340 W on this workload.

## Applying memory clock (Linux)
Core (SM) clock uses `nvidia-smi -lgc`. Memory clock offsets need the coolbits path:
```bash
sudo nvidia-xconfig --enable-all-gpus --cool-bits=28
# then, in an X session:
nvidia-settings -a "[gpu:0]/GPUMemoryTransferRateOffsetAllPerformanceLevels=1500"
```
On Windows/WSL, set the memory offset with MSI Afterburner on the host before launching the miner.

## Verifying
```bash
nvidia-smi dmon -s pucvmet -d 1         # power/util/clocks/mem/temp, 1 Hz
scripts/benchmark_3090.sh 300           # captures dmon alongside the run
```
Tune one variable at a time; record TH/s (accepted-proof rate against a node), watts, GPU temp, and
**VRAM temp** in `docs/benchmarking.md`. Back off immediately if `vram_temp_c` approaches the limit.
