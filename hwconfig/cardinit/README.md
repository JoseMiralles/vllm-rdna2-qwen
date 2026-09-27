# cardinit — GPU power cap, clock ceiling and undervolt

Two scripts that put the cards at the operating point described in [`PRODUCTION.md`](../../PRODUCTION.md):
a power cap on every AMD GPU and, for the Radeon PRO V620s, a clock ceiling and a small core-voltage offset.
None of these settings survive a reboot or a driver reload, so run `cardinit` after every boot.

| File | Runs as | Role |
|---|---|---|
| `cardinit` | your user | **Holds the values.** Edit them here; no sudo needed. Calls the helper through `sudo`. |
| `cardinit-apply` | root | **Only validates and applies.** Finds the cards, writes the settings, reads them back and reports. |

## What it sets

```bash
V620_POWER=140      # W, power cap for each V620
OTHER_POWER=140     # W, power cap for every other AMD GPU (e.g. a display card)
V620_SCLK=2100      # MHz, V620 clock ceiling
V620_VOFFSET=-25    # mV, V620 core-voltage offset (undervolt)
```

Leave a value empty (`""`) to leave that setting untouched.

- **Power cap**: written to each card's `hwmon/power1_cap`. The card's firmware holds average power at or
  below it.
- **Clock ceiling**: OverDrive table entry `s 1 <MHz>` in the card's `pp_od_clk_voltage`. Set a little
  below what the card sustains under the cap, the cap stops constantly adjusting clock and voltage, and
  clocks stay flat under load. Idle behaviour is unchanged.
- **Voltage offset**: OverDrive `vo <mV>`, shifting the card's whole voltage–frequency curve. The helper
  only accepts 0 or negative values (−150 to 0). Too much undervolt can produce wrong results without
  crashing, so check model quality after changing it.

The clock ceiling and voltage offset need OverDrive enabled for the V620, which stock kernels don't allow.
Power caps below the V620's factory floor (232 W) also need a patched driver. Both patches are in
[`../kernel-patches/`](../kernel-patches/).

## Finding the cards

Every run scans `/sys/bus/pci/devices` for AMD display-class devices. It picks out V620s by PCI device ID
`1002:73a1` and treats every other AMD GPU as "other". Nothing depends on PCI addresses or card numbers,
which change when cards move slots or the boot order changes. The report ends with how many V620s and other
GPUs it found, and warns if it found no V620.

## Install

```bash
# 1. the helper, root-owned (repeat only after editing cardinit-apply itself)
sudo install -o root -g root -m 0755 cardinit-apply /usr/local/sbin/cardinit-apply

# 2. the user script, anywhere on your PATH
install -m 0755 cardinit ~/bin/cardinit

# 3. passwordless sudo for the helper only
sudo visudo -f /etc/sudoers.d/cardinit
```

The sudoers file needs one line (replace `<user>`):

```
<user> ALL=(root) NOPASSWD: /usr/local/sbin/cardinit-apply
```

The helper takes the values as options, so the rule must not restrict arguments. That's safe because the
helper validates every value (integers only, bounded ranges, undervolt-only offsets, unknown options
refused). **Keep the helper root-owned in a root-owned directory.** If sudo pointed at a file your user can
edit, anything running as your user could rewrite it and gain root.

## Use

```bash
cardinit
```

Example output:

```
0000:0a:00.0 other (0x73df): cap 140 W, perf level auto
0000:0d:00.0 V620 (0x73a1): cap 140 W, perf level auto, sclk max 2100 MHz, voltage offset -25 mV
...
found 4 V620(s), 1 other AMD GPU(s)
```

Any setting that did not take is flagged `[... NOT ...]`, and the exit status is non-zero.

The helper can also be run directly, e.g. to try values once without editing `cardinit`:

```bash
sudo /usr/local/sbin/cardinit-apply --v620-power 150 --v620-sclk 2200 --v620-voffset -25
```

Options: `--v620-power W`, `--other-power W`, `--v620-sclk MHz`, `--v620-voffset mV`. Omitted options
are not touched.

## Details

- If the driver rejects an OverDrive write, the helper switches that card to the `manual` performance
  level, retries, then restores the previous level. Otherwise it never changes the performance level. Use
  `auto`: the ceiling only applies under load, and idle clocks, voltage and power stay low.
- To undo: set `V620_SCLK` back to the card's default ceiling and `V620_VOFFSET=0`, then run `cardinit`.
  Or reboot. Avoid writing `r` to `pp_od_clk_voltage` by hand: it resets the whole OverDrive table.
- `rocm-smi` and `amd-smi` report the firmware's **target** clock, which reads as the ceiling (e.g. 2100)
  even at idle. The effective clock is in `hwmon/freq1_input`, and at idle it is near zero.
- To apply automatically at boot, run the helper from a oneshot systemd unit ordered after the GPU driver
  has loaded.
