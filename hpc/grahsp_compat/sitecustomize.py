"""Runtime compatibility patches for the external GRAHSP/CIGALE checkout."""

from __future__ import annotations

import numpy as np


def _patch_pcigale_sfh_tuple_setter() -> None:
    try:
        from pcigale.sed import SED
    except Exception:
        return

    sfh_property = getattr(SED, "sfh", None)
    if not isinstance(sfh_property, property) or getattr(sfh_property.fset, "_grahsp_tuple_compat", False):
        return

    def sfh(self, value):
        self._sfh = value

        if value is not None:
            if isinstance(value, tuple) and len(value) == 2:
                _, sfh_sfr = value
            else:
                sfh_sfr = value
            sfh_sfr = np.asarray(sfh_sfr, dtype=float)
            self._sfh = sfh_sfr
            self.add_info("sfh.sfr", sfh_sfr[-1], True, force=True)
            self.add_info("sfh.sfr10Myrs", np.mean(sfh_sfr[-10:]), True, force=True)
            self.add_info("sfh.sfr100Myrs", np.mean(sfh_sfr[-100:]), True, force=True)
            self.add_info("sfh.age", sfh_sfr.size, False, force=True)

    sfh._grahsp_tuple_compat = True
    SED.sfh = property(sfh_property.fget, sfh, sfh_property.fdel, sfh_property.__doc__)


_patch_pcigale_sfh_tuple_setter()
