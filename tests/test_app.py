"""Headless end-to-end test of the Streamlit app (streamlit.testing AppTest)."""
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app" / "app.py")


def _alerts(at):
    return " ".join(e.value for e in list(at.success) + list(at.warning) + list(at.error))


def test_app_runs_and_recomputes_on_what_if():
    at = AppTest.from_file(APP, default_timeout=180).run()
    assert not at.exception, [e.value for e in at.exception]
    assert len(at.tabs) == 8
    md = [m.value for m in at.markdown if "BRD arrival decision summary" in m.value]
    assert md and "veterinary review" in md[0].lower()
    before = _alerts(at)
    # Make metaphylaxis useless: the recommendation must change and confidence messaging must react
    at.slider(key="p_metaphylaxis_morbidity_rrr").set_value(0.25).run()
    assert not at.exception, [e.value for e in at.exception]
    assert _alerts(at) != before


def test_checkin_tab_and_weather_override():
    at = AppTest.from_file(APP, default_timeout=180).run()
    assert not at.exception, [e.value for e in at.exception]
    text = " ".join(m.value for m in at.markdown)
    assert "Veterinarian to review" in text            # demo pulls produce at least one escalation
    assert any("synthetic" in i.value.lower() for i in at.info)
    risk_before = int([m.value for m in at.metric if m.label == "High risk lots"][0])
    at.selectbox(key="wx_B").set_value("high").run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("Weather forecast override" in i.value for i in at.info)
    at.slider(key="checkin_day").set_value(14).run()
    assert not at.exception, [e.value for e in at.exception]
    assert int([m.value for m in at.metric if m.label == "High risk lots"][0]) >= risk_before


def test_scenario_load_applies_saved_settings():
    at = AppTest.from_file(APP, default_timeout=180).run()
    at.session_state["_pending_scenario"] = {"p_metaphylaxis_morbidity_rrr": 0.3, "wx_D": "high",
                                             "base_rule": "weight_auction", "not_a_setting": 1}
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.slider(key="p_metaphylaxis_morbidity_rrr").value == 0.3
    assert at.selectbox(key="wx_D").value == "high"
    assert at.selectbox(key="base_rule").value == "weight_auction"
    assert "not_a_setting" not in at.session_state


def test_malformed_scenario_is_skipped_not_crashing():
    at = AppTest.from_file(APP, default_timeout=180).run()
    at.session_state["_pending_scenario"] = {"p_metaphylaxis_morbidity_rrr": 5.0, "base_cut": "abc",
                                             "wx_A": "tornado", "r_B": -3, "stepdown": "yes", "h_C": 30.0}
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.slider(key="p_metaphylaxis_morbidity_rrr").value != 5.0
    assert at.number_input(key="h_C").value == 30.0
    assert "Skipped" in at.session_state["_scenario_msg"]
