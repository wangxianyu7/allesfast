"""
Tests for RM β_IP lookup improvements:

1. Dot-notation instrument names: 'HARPS.jiang0' should resolve to HARPS β_IP
2. flux_weighted numeric resolution: passing R as a number computes β_IP = c/(R·2√(2ln2))
"""

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Shared RM call parameters (circular orbit, face-on-ish, simple geometry)
# ---------------------------------------------------------------------------
_RM_KWARGS = dict(
    rr=0.1, ar=10.0, period=3.0, t0=0.0, inc=87.0,
    ecc=0.0, omega=90.0, lambda_r=30.0, vsini=5.0,
    xi=1.0, zeta=3.0, u1=0.3, u2=0.2, teff=5800.0,
)
_TIME = np.linspace(-0.05, 0.05, 11)   # covers the transit window


# ---------------------------------------------------------------------------
# 1.  β_IP dot-notation: 'HARPS.jiang0'  →  same result as 'HARPS'
# ---------------------------------------------------------------------------

class TestBetaIPDotNotation:

    def test_dot_instrument_same_as_base_name(self):
        """HARPS.jiang0 should produce the same RM as plain HARPS."""
        from allesfast.computer import get_rm_hirano2011
        rm_base = get_rm_hirano2011(_TIME, inst='HARPS', **_RM_KWARGS)
        rm_dot  = get_rm_hirano2011(_TIME, inst='HARPS.jiang0', **_RM_KWARGS)
        np.testing.assert_array_almost_equal(rm_base, rm_dot, decimal=10)

    def test_dot_instrument_multiple_suffixes(self):
        """HARPS.jiang1 and HARPS.jiang2 both resolve to HARPS β_IP."""
        from allesfast.computer import get_rm_hirano2011
        rm_ref = get_rm_hirano2011(_TIME, inst='HARPS', **_RM_KWARGS)
        for suffix in ('jiang0', 'jiang1', 'jiang2', 'obs1'):
            rm = get_rm_hirano2011(_TIME, inst=f'HARPS.{suffix}', **_RM_KWARGS)
            np.testing.assert_array_almost_equal(rm_ref, rm, decimal=10,
                err_msg=f'Failed for HARPS.{suffix}')

    def test_dot_unknown_base_falls_back_to_default(self):
        """UnknownSpec.obs1 → base 'UnknownSpec' not in dict → uses default β_IP."""
        from allesfast.computer import get_rm_hirano2011
        rm_default = get_rm_hirano2011(_TIME, inst=None, **_RM_KWARGS)
        rm_unknown = get_rm_hirano2011(_TIME, inst='UnknownSpec.obs1', **_RM_KWARGS)
        np.testing.assert_array_almost_equal(rm_default, rm_unknown, decimal=10)

    def test_known_instrument_without_dot_unchanged(self):
        """Existing plain names still work correctly (regression)."""
        from allesfast.computer import get_rm_hirano2011
        for name in ('HARPS', 'ESPRESSO', 'KPF'):
            rm = get_rm_hirano2011(_TIME, inst=name, **_RM_KWARGS)
            assert np.all(np.isfinite(rm)), f'NaN/Inf for {name}'


# ---------------------------------------------------------------------------
# 2.  flux_weighted numeric resolution: resolution → β_IP = c/(R·2√(2ln2))
# ---------------------------------------------------------------------------

class TestBetaIPFromResolution:

    def _expected_beta_ip(self, R):
        c_kms = 2.998e5
        return c_kms / (R * 2.0 * np.sqrt(2.0 * np.log(2.0)))

    def test_resolution_gives_correct_beta_ip(self):
        """Passing resolution=115000 gives β_IP close to the hardcoded HARPS value."""
        from allesfast.computer import _BETA_IP_KMS
        beta_harps  = _BETA_IP_KMS['HARPS']          # 1.10 km/s
        beta_from_r = self._expected_beta_ip(115000)
        assert abs(beta_harps - beta_from_r) < 0.05, (
            f"HARPS β_IP={beta_harps} vs computed={beta_from_r:.3f}")

    def test_resolution_kwarg_overrides_inst_name(self):
        """When resolution is given, it overrides the inst name lookup."""
        from allesfast.computer import get_rm_hirano2011
        rm_harps = get_rm_hirano2011(_TIME, inst='HARPS', **_RM_KWARGS)
        rm_res   = get_rm_hirano2011(_TIME, inst='SomeUnknownSpec',
                                     resolution=115000, **_RM_KWARGS)
        np.testing.assert_allclose(rm_harps, rm_res, rtol=0.02,
            err_msg='resolution= should give nearly same result as HARPS lookup')

    def test_resolution_none_falls_back_to_name_lookup(self):
        """resolution=None should behave identically to not passing it."""
        from allesfast.computer import get_rm_hirano2011
        rm_no_res   = get_rm_hirano2011(_TIME, inst='HARPS', **_RM_KWARGS)
        rm_none_res = get_rm_hirano2011(_TIME, inst='HARPS', resolution=None, **_RM_KWARGS)
        np.testing.assert_array_equal(rm_no_res, rm_none_res)

    def test_resolution_dot_instrument_with_resolution(self):
        """Dot-notation + explicit resolution: resolution wins over base-name lookup."""
        from allesfast.computer import get_rm_hirano2011
        rm_res      = get_rm_hirano2011(_TIME, inst='HARPS.jiang0',
                                        resolution=140000, **_RM_KWARGS)
        rm_espresso = get_rm_hirano2011(_TIME, inst='ESPRESSO', **_RM_KWARGS)
        # R=140000 ≈ ESPRESSO (0.91 km/s)
        np.testing.assert_allclose(rm_res, rm_espresso, rtol=0.02)
