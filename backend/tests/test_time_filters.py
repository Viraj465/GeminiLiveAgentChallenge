import pytest
from core.vision_loop import _infer_google_qdr, _infer_google_time_phrase, _build_google_filtered_url, _build_direct_time_query_url

def test_infer_google_qdr():
    assert _infer_google_qdr("find papers from the past 2 years") == "y2"
    assert _infer_google_qdr("last 3 weeks of research") == "w3"
    assert _infer_google_qdr("papers from the past month") == "m"
    assert _infer_google_qdr("today's news") == "d"
    assert _infer_google_qdr("past 1 hour") == "h"
    assert _infer_google_qdr("no time constraint") is None

def test_infer_google_time_phrase():
    assert _infer_google_time_phrase("papers from 2020 to 2022") == "2020 to 2022"
    assert _infer_google_time_phrase("since 2018") == "since 2018"
    assert _infer_google_time_phrase("past month") == "past month"
    assert _infer_google_time_phrase("no time") is None

def test_build_google_filtered_url_range():
    url = "https://www.google.com/search?q=test"
    task = "papers from 2020 to 2022"
    filtered = _build_google_filtered_url(url, task)
    assert "tbs=cdr%3A1%2Ccd_min%3A1%2F1%2F2020%2Ccd_max%3A12%2F31%2F2022" in filtered

def test_build_google_filtered_url_since():
    url = "https://www.google.com/search?q=test"
    task = "since 2021"
    filtered = _build_google_filtered_url(url, task)
    assert "tbs=cdr%3A1%2Ccd_min%3A1%2F1%2F2021%2Ccd_max%3A12%2F31%2F2026" in filtered

def test_build_direct_time_query_url_scholar():
    url = "https://scholar.google.com/scholar?q=deep+learning"
    task = "papers from 2019 to 2021"
    # _build_direct_time_query_url calls _build_google_filtered_url first, which returns None for scholar
    # then it hits the scholar path in _build_direct_time_query_url
    filtered = _build_direct_time_query_url(url, task)
    assert "as_ylo=2019" in filtered
    assert "as_yhi=2021" in filtered

def test_build_google_filtered_url_relative():
    url = "https://www.google.com/search?q=test"
    task = "past 3 years"
    filtered = _build_google_filtered_url(url, task)
    assert "tbs=qdr%3Ay3" in filtered
