# hwconfig — host-side configuration

What the host needs beyond this vLLM fork to run the cards at the operating point in
[`PRODUCTION.md`](../PRODUCTION.md).

| Directory | Contents |
|---|---|
| [`kernel-patches/`](kernel-patches/) | Two amdgpu patches: OverDrive for the Radeon PRO V620, and a 100 W minimum power cap for the V620 and Navi 22. How to apply, build and verify. |
| [`cardinit/`](cardinit/) | Scripts that set the power caps, the V620 clock ceiling and the undervolt after every boot. User-editable values plus a root-owned helper. |

The kernel command line we use is listed in `PRODUCTION.md`.
