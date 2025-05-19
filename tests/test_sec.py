import pytest
from polymer_gc.sec import SimSEC


@pytest.mark.parametrize(
    "mn_target, mw_target, pdi_target",
    [
        (10000, 20000, 2.0),
        (50000, 55000, 1.1),
    ],
)
def test_simsec_from_mn_mw_basic(mn_target, mw_target, pdi_target):
    sec = SimSEC.from_mn_mw(
        Mn=mn_target,
        Mw=mw_target,
    )
    assert isinstance(sec, SimSEC)
    assert SimSEC.DEFAULT_VOLUME_COLUMN in sec._raw_data.columns
    assert "signal" in sec._raw_data.columns

    # Check calculated Mn and Mw
    # Need to select a peak. For simulation, assume one dominant peak.
    sec.autodetect_signal_boarders()  # This should find one peak.

    # If autodetect_signal_boarders finds peaks, use the first one.
    # This part depends on secanalysis internal logic heavily.
    # For SimSEC, it's expected one peak is generated.
    if sec.signal_boarders and len(sec.signal_boarders) > 0:
        # Using the first detected peak
        i_start, i_end, _ = sec.signal_boarders[0]
        m_start, m_end = sec.mass_range[[i_end, i_start]]

        params = sec.calc_mass_params(m_start, m_end)
        mn_calc, mw_calc = params["Mn"], params["Mw"]

        assert (
            pytest.approx(mn_calc, rel=0.15) == mn_target
        )  # Allow larger tolerance due to simulation noise & smaller N
        assert pytest.approx(mw_calc, rel=0.15) == mw_target

        assert pytest.approx(mw_calc / mn_calc, rel=0.15) == pdi_target
    else:
        pytest.fail("SimSEC.from_mn_mw did not produce a detectable signal peak.")


def test_simsec_from_mn_mw_errors():
    with pytest.raises(ValueError, match="Mw must exceed Mn."):
        SimSEC.from_mn_mw(Mn=10000, Mw=5000)

    with pytest.raises(ValueError, match="Calibration slope 'a' cannot be zero."):
        SimSEC.from_mn_mw(Mn=1000, Mw=2000, calibration_params=(0.0, 10.0))


def test_simsec_from_mn_mw_with_asymmetry_noise_baseline(monkeypatch):
    # Mock expensive part for speed if needed, or use small N
    sec = SimSEC.from_mn_mw(
        Mn=20000,
        Mw=40000,
        asymmetry=0.5,
    )
    assert isinstance(sec, SimSEC)
    # Basic check: signal should not be all zeros, and should have some values > baseline
    assert sec._raw_data["signal"].max() > 0.05
