#!/bin/bash
# Run after rebooting into the new module. No root needed (dmesg may need it).
echo "== running module =="; modinfo -F filename amdgpu; modinfo -F vermagic amdgpu
echo
for c in /sys/class/drm/card[0-9]; do
  d=$c/device; [ -f "$d/uevent" ] || continue
  id=$(grep -m1 PCI_ID "$d/uevent" | cut -d= -f2)
  for h in "$d"/hwmon/hwmon*; do [ -d "$h" ] || continue
    printf "%-7s %-11s cap=%4sW  min=%4sW  max=%4sW  default=%4sW  now=%sW\n" \
      "$(basename $c)" "$id" \
      $(( $(cat $h/power1_cap)/1000000 )) $(( $(cat $h/power1_cap_min)/1000000 )) \
      $(( $(cat $h/power1_cap_max)/1000000 )) $(( $(cat $h/power1_cap_default)/1000000 )) \
      $(( $(cat $h/power1_average 2>/dev/null || echo 0)/1000000 ))
  done
done
echo
echo "== powerfix notices =="; dmesg 2>/dev/null | grep -i powerfix || echo "(need sudo dmesg, or none logged)"
