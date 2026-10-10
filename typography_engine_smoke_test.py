#!/usr/bin/env python3
"""Bounded smoke test for the Math-by-Design typography engine."""

from __future__ import annotations

import json

import numpy as np

import typography_engine as te


def main() -> int:
    self_test = te.run_self_tests()
    assert self_test["pass"] is True
    assert self_test["unique_mode_systems"] >= 2

    thesis = te.ProjectThesis(
        label="MBD smoke: technical editorial",
        vector=np.array([0.82, 0.58, 0.18, 0.70, 0.12, 0.08]),
    )

    systems = {}
    for mode in te.PairingMode:
        packet = te.run_engine(
            thesis=thesis,
            catalog=te.DEMO_CATALOG,
            pairing_mode=mode,
            ratio_name="auto",
        )
        assert packet["typography_engine"] == "mbd-type/0.2"
        assert packet["catalog_status"] == "ILLUSTRATIVE_TRAITS_NOT_MEASURED_FROM_FONT_FILES"
        assert packet["selection"]["families"]["reading"]
        assert packet["selection"]["families"]["instrument"]
        assert packet["selection"]["families"]["display"]
        assert "realized_role_truth" in packet["validation"]
        assert "system_identity_test" in packet["validation"]
        assert "realization_error" in packet["validation"]
        json.dumps(packet)
        systems[mode.value] = tuple(packet["selection"]["families"][r] for r in te.ROLES)

    assert len(set(systems.values())) >= 2, systems

    print(json.dumps({
        "pass": True,
        "engine": "mbd-type/0.2",
        "pairing_modes": systems,
        "unique_systems": len(set(systems.values())),
        "self_test": self_test,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
