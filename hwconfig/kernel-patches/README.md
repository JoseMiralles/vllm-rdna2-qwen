# amdgpu kernel patches for the Radeon PRO V620

Two small patches to the amdgpu driver's power-management code for Navi 21/22 (`sienna_cichlid_ppt.c`).
Together they make the operating point in [`PRODUCTION.md`](../../PRODUCTION.md) possible: power caps well
below the factory floor, plus a clock ceiling and voltage offset through the OverDrive interface. The
settings are then applied at runtime with [`../cardinit/`](../cardinit/).

Neither patch changes behaviour by itself. They only *allow* lower caps and OverDrive edits, and the cards
run at factory defaults until something sets them.

| Patch | What it does |
|---|---|
| `v620-enable-overdrive-caps.patch` | Enables OverDrive on the V620. |
| `amdgpu-powercap-min-100W-v620+navi22.patch` | Lowers the minimum allowed power cap to 100 W on the V620 and on Navi 22 cards. |
| `postcheck.sh` | After booting the patched kernel, prints every card's power-cap range and any "powerfix" kernel notices. |

## `v620-enable-overdrive-caps.patch`

The V620's firmware ships an OverDrive table, but all its capability flags are zeroed, so the driver never
exposes `pp_od_clk_voltage` for editing. The patch sets the four capabilities the table actually supports:
GFX clock limits, GFX clock curve, memory clock limits and power limit. It is gated on PCI device ID
`0x73a1` (V620) only.

Result: `/sys/bus/pci/devices/<card>/pp_od_clk_voltage` becomes writable, so you can set a clock ceiling
(`s 1 <MHz>`) and a core-voltage offset (`vo <mV>`). The driver also needs OverDrive enabled in its feature
mask, via `amdgpu.ppfeaturemask` with bit `0x4000` set; our value is `0xfff77fff` (see `PRODUCTION.md`).

## `amdgpu-powercap-min-100W-v620+navi22.patch`

The board firmware declares a lower bound on the power limit that keeps the minimum cap far above what the
cards can run at: **232 W** on a V620 (default 250 W) and ~174 W on an RX 6700 XT (default 186 W). The patch
adds a small table of PCI-ID matches and lowers the minimum for matching devices:

| Match | New floor |
|---|---|
| `1002:73a1`, subsystem `1002:0e34`: Radeon PRO V620 reference board | 100 W |
| `1002:73df`, any subsystem: Navi 22 (RX 6700 / 6700 XT / 6750 XT) | 100 W |

The kernel logs `powerfix: allowing PPT limit down to 100 W on <device>` for each matched card. The maximum
and the default cap are unchanged. To use a different floor, or add another board, edit the `driver_data`
values or the table entries in the patch.

## Applying and building

The patches were made against upstream **v7.2.6**, via Ubuntu's mainline build tree, and both apply
cleanly there. The same functions exist in nearby kernels (we ran earlier versions on 7.0.x), so expect
them to apply with little or no fuzz. Always check for rejected hunks.

**Full kernel build (what we do):**

```bash
git clone --depth 1 -b cod/mainline/v7.2.6 \
  git://git.launchpad.net/~ubuntu-kernel-test/ubuntu/+source/linux/+git/mainline-crack linux-7.2.6
cd linux-7.2.6
patch -p1 < ../v620-enable-overdrive-caps.patch
patch -p1 < ../amdgpu-powercap-min-100W-v620+navi22.patch
# optional: prepend a changelog entry with a local version suffix (e.g. +v620uncapped) to debian.master/changelog

export CONCURRENCY_LEVEL=$(nproc)
LANG=C fakeroot debian/rules clean
LANG=C fakeroot debian/rules binary-headers binary-generic binary-perarch
```

That produces the usual `linux-image-unsigned`, `linux-modules` and `linux-headers` `.deb`s one directory up.
Install them with `sudo apt install ./linux-image-unsigned-*.deb ./linux-modules-*.deb ./linux-headers-*.deb`,
then reboot into the new kernel.

- The release string matches Ubuntu's own mainline build of the same version. Don't install both.
- The image is unsigned and the modules are signed with a build-time key. This needs Secure Boot off, or
  your own key enrolled.
- `debian/rules` doesn't check build dependencies. If a Rust-related package is flagged as missing, check
  whether a versioned equivalent is already installed.

**Out-of-tree module build (faster, fiddlier):** you can rebuild just `amdgpu` against the installed
headers (`make M=drivers/gpu/drm/amd/amdgpu`). Two traps:
- Start from the *same* kernel source the running kernel was built from, with **both** patches applied.
  Tools that extract a pristine `linux-source` package silently drop the OverDrive patch.
- `amdgpu_trace.h` sets `TRACE_INCLUDE_PATH ../../drivers/gpu/drm/amd/amdgpu`. When the headers' `include/trace`
  is a symlink into another package, `../../` resolves physically to the wrong place. Pass
  `KCFLAGS=-I<dir>/a/b`, where `<dir>/drivers/gpu/drm/amd` is a symlink to the patched amd source.

  With `CONFIG_MODVERSIONS=y`, build against the installed headers' `Module.symvers`.

## Verifying

After rebooting into the patched kernel:

```bash
./postcheck.sh
```

The V620s should show `min= 100W`, and the kernel log should contain a `powerfix` line per matched card
(`sudo dmesg | grep powerfix`). OverDrive is enabled if `cat /sys/bus/pci/devices/<V620>/pp_od_clk_voltage`
prints an `OD_SCLK` / `OD_VDDGFX_OFFSET` table and accepts writes. `../cardinit/` exercises both paths and
reports anything that didn't take.

## Caution

Running below the factory minimum cap, and editing clocks and voltages, is outside what AMD validated for
these boards. We run it because it gives us better stability and efficiency on our platform (see
`PRODUCTION.md`), not because it is guaranteed safe on yours. Keep changes small and test under load.
