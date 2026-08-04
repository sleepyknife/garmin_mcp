"""
Integration tests for training module MCP tools

Tests training tools using FastMCP integration with mocked Garmin API responses.
"""
import pytest
from unittest.mock import Mock
from mcp.server.fastmcp import FastMCP

import datetime
import json

from garmin_mcp import training
from tests.fixtures.garmin_responses import (
    MOCK_PROGRESS_SUMMARY,
    MOCK_RECENT_TRAINING_ACTIVITIES,
    MOCK_HRV_DATA,
    MOCK_TRAINING_STATUS,
    MOCK_LACTATE_THRESHOLD,
    MOCK_LACTATE_THRESHOLD_RANGE,
    MOCK_CYCLING_FTP,
    MOCK_ENDURANCE_SCORE,
    MOCK_ACTIVITY_TYPES,
)


@pytest.fixture
def app_with_training(mock_garmin_client):
    """Create FastMCP app with training tools registered"""
    training.configure(mock_garmin_client)
    app = FastMCP("Test Training")
    app = training.register_tools(app)
    return app


@pytest.mark.asyncio
async def test_get_progress_summary_between_dates_tool(app_with_training, mock_garmin_client):
    """Test get_progress_summary_between_dates tool"""
    # Setup mock
    mock_garmin_client.get_progress_summary_between_dates.return_value = MOCK_PROGRESS_SUMMARY

    # Call tool
    result = await app_with_training.call_tool(
        "get_progress_summary_between_dates",
        {
            "start_date": "2024-01-08",
            "end_date": "2024-01-15",
            "metric": "duration"
        }
    )

    # Verify
    assert result is not None
    mock_garmin_client.get_progress_summary_between_dates.assert_called_once_with(
        "2024-01-08", "2024-01-15", "duration"
    )


@pytest.mark.asyncio
async def test_get_hill_score_tool(app_with_training, mock_garmin_client):
    """Test get_hill_score tool"""
    # Setup mock
    hill_score = {
        "hillScore": 75,
        "dateRange": {"start": "2024-01-08", "end": "2024-01-15"}
    }
    mock_garmin_client.get_hill_score.return_value = hill_score

    # Call tool
    result = await app_with_training.call_tool(
        "get_hill_score",
        {"start_date": "2024-01-08", "end_date": "2024-01-15"}
    )

    # Verify
    assert result is not None
    mock_garmin_client.get_hill_score.assert_called_once_with("2024-01-08", "2024-01-15")


@pytest.mark.asyncio
async def test_get_endurance_score_tool(app_with_training, mock_garmin_client):
    """Test get_endurance_score tool with realistic API response"""
    # Setup mocks
    mock_garmin_client.get_endurance_score.return_value = MOCK_ENDURANCE_SCORE
    mock_garmin_client.get_activity_types.return_value = MOCK_ACTIVITY_TYPES

    # Reset the activity type cache to ensure fresh lookup
    training._activity_type_cache = None

    # Call tool
    result = await app_with_training.call_tool(
        "get_endurance_score",
        {"start_date": "2024-01-08", "end_date": "2024-01-15"}
    )

    # Verify API was called correctly
    assert result is not None
    mock_garmin_client.get_endurance_score.assert_called_once_with("2024-01-08", "2024-01-15")

    # Parse the result and verify content
    data = json.loads(result[0][0].text)

    # Check period summary
    assert data["period_avg_score"] == 5631
    assert data["period_max_score"] == 5740

    # Check current score
    assert data["current_score"] == 5712
    assert data["current_date"] == "2024-01-15"
    assert data["classification"] == "intermediate"
    assert data["classification_id"] == 2

    # Check thresholds
    assert "thresholds" in data
    assert data["thresholds"]["trained"] == 5800
    assert data["thresholds"]["well_trained"] == 6500

    # Check contributors have activity type names
    assert "contributors" in data
    contributors = data["contributors"]
    assert len(contributors) == 4

    # Find the hiking contributor
    hiking_contributor = next(
        (c for c in contributors if c.get("activity_type") == "hiking"), None
    )
    assert hiking_contributor is not None
    assert hiking_contributor["contribution_percent"] == 5.49
    assert hiking_contributor["activity_type_id"] == 3

    # Find the yoga contributor
    yoga_contributor = next(
        (c for c in contributors if c.get("activity_type") == "yoga"), None
    )
    assert yoga_contributor is not None
    assert yoga_contributor["contribution_percent"] == 3.13

    # Check weekly breakdown exists
    assert "weekly_breakdown" in data
    assert len(data["weekly_breakdown"]) == 1
    week = data["weekly_breakdown"][0]
    assert week["week_start"] == "2024-01-08"
    assert week["avg_score"] == 5548
    assert week["max_score"] == 5561


@pytest.mark.asyncio
async def test_get_training_effect_tool(app_with_training, mock_garmin_client):
    """Test get_training_effect tool"""
    # Setup mock - get_training_effect uses get_activity internally
    activity_data = {
        "summaryDTO": {
            "trainingEffect": 3.5,
            "anaerobicTrainingEffect": 2.0,
            "trainingEffectLabel": "Highly Improving",
            "activityTrainingLoad": 150,
            "recoveryTime": 720,  # 12 hours in minutes
            "performanceCondition": 95,
        }
    }
    mock_garmin_client.get_activity.return_value = activity_data

    # Call tool
    result = await app_with_training.call_tool(
        "get_training_effect",
        {"activity_id": 12345678901}
    )

    # Verify
    assert result is not None
    mock_garmin_client.get_activity.assert_called_once_with(12345678901)


@pytest.mark.asyncio
async def test_get_hrv_data_tool(app_with_training, mock_garmin_client):
    """Test get_hrv_data tool"""
    # Setup mock
    mock_garmin_client.get_hrv_data.return_value = MOCK_HRV_DATA

    # Call tool
    result = await app_with_training.call_tool(
        "get_hrv_data",
        {"date": "2024-01-15"}
    )

    # Verify
    assert result is not None
    mock_garmin_client.get_hrv_data.assert_called_once_with("2024-01-15")


@pytest.mark.asyncio
async def test_get_fitnessage_data_tool(app_with_training, mock_garmin_client):
    """Test get_fitnessage_data tool"""
    # Setup mock
    fitness_age = {
        "fitnessAge": 25,
        "chronologicalAge": 30,
        "vo2Max": 52.5,
        "date": "2024-01-15"
    }
    mock_garmin_client.get_fitnessage_data.return_value = fitness_age

    # Call tool
    result = await app_with_training.call_tool(
        "get_fitnessage_data",
        {"date": "2024-01-15"}
    )

    # Verify
    assert result is not None
    mock_garmin_client.get_fitnessage_data.assert_called_once_with("2024-01-15")


@pytest.mark.asyncio
async def test_get_cycling_ftp_tool(app_with_training, mock_garmin_client):
    """Test get_cycling_ftp tool returns latest FTP data"""
    mock_garmin_client.get_cycling_ftp.return_value = MOCK_CYCLING_FTP

    result = await app_with_training.call_tool("get_cycling_ftp", {})

    assert result is not None
    mock_garmin_client.get_cycling_ftp.assert_called_once_with()

    data = json.loads(result[0][0].text)
    assert data["sport"] == "CYCLING"
    assert data["functional_threshold_power_watts"] == 294
    assert data["calendar_date"] == "2024-03-15T10:30:00.000"
    assert data["is_stale"] is False
    assert data["biometric_source_type"] == "CHANGE_LOG"


@pytest.mark.asyncio
async def test_request_reload_tool(app_with_training, mock_garmin_client):
    """Test request_reload tool"""
    # Setup mock
    reload_response = {"status": "success", "message": "Data reload requested"}
    mock_garmin_client.request_reload.return_value = reload_response

    # Call tool
    result = await app_with_training.call_tool(
        "request_reload",
        {"date": "2024-01-15"}
    )

    # Verify
    assert result is not None
    mock_garmin_client.request_reload.assert_called_once_with("2024-01-15")


@pytest.mark.asyncio
async def test_get_training_status_tool(app_with_training, mock_garmin_client):
    """Test get_training_status tool returns training status"""
    # Setup mock
    mock_garmin_client.get_training_status.return_value = MOCK_TRAINING_STATUS

    # Call tool
    result = await app_with_training.call_tool(
        "get_training_status",
        {"date": "2024-01-15"}
    )

    # Verify
    assert result is not None
    mock_garmin_client.get_training_status.assert_called_once_with("2024-01-15")


@pytest.mark.asyncio
async def test_get_lactate_threshold_tool_latest(app_with_training, mock_garmin_client):
    """Test get_lactate_threshold tool returns latest lactate threshold data"""
    # Setup mock with latest=True response format
    mock_garmin_client.get_lactate_threshold.return_value = MOCK_LACTATE_THRESHOLD

    # Call tool with no dates (gets latest)
    result = await app_with_training.call_tool(
        "get_lactate_threshold",
        {}
    )

    # Verify API call
    assert result is not None
    mock_garmin_client.get_lactate_threshold.assert_called_once_with(latest=True)

    # Verify output structure
    data = json.loads(result[0][0].text)
    assert data["lactate_threshold_speed_mps"] == 0.32222132
    assert data["lactate_threshold_heart_rate_bpm"] == 169
    assert data["functional_threshold_power_watts"] == 334
    assert data["sport"] == "RUNNING"
    assert data["power_to_weight"] == 4.575


@pytest.mark.asyncio
async def test_get_lactate_threshold_tool_range(app_with_training, mock_garmin_client):
    """Test get_lactate_threshold tool returns lactate threshold data for date range"""
    # Setup mock with date range response format
    mock_garmin_client.get_lactate_threshold.return_value = MOCK_LACTATE_THRESHOLD_RANGE

    # Call tool with date range
    result = await app_with_training.call_tool(
        "get_lactate_threshold",
        {"start_date": "2024-01-08", "end_date": "2024-01-15"}
    )

    # Verify API call
    assert result is not None
    mock_garmin_client.get_lactate_threshold.assert_called_once_with(
        latest=False,
        start_date="2024-01-08",
        end_date="2024-01-15",
    )

    # Verify output structure
    data = json.loads(result[0][0].text)
    assert data["start_date"] == "2024-01-08"
    assert data["end_date"] == "2024-01-15"
    assert "speed_history" in data
    assert len(data["speed_history"]) == 3
    assert data["speed_history"][0]["date"] == "2024-01-08"
    assert "heart_rate_history" in data
    assert len(data["heart_rate_history"]) == 3
    assert "power_history" in data


# Error handling tests
@pytest.mark.asyncio
async def test_get_hrv_data_no_data(app_with_training, mock_garmin_client):
    """Test get_hrv_data tool when no data available"""
    # Setup mock to return None
    mock_garmin_client.get_hrv_data.return_value = None

    # Call tool
    result = await app_with_training.call_tool(
        "get_hrv_data",
        {"date": "2024-01-15"}
    )

    # Verify error message is returned
    assert result is not None


@pytest.mark.asyncio
async def test_get_training_effect_exception(app_with_training, mock_garmin_client):
    """Test get_training_effect tool when API raises exception"""
    # Setup mock to raise exception - get_training_effect uses get_activity internally
    mock_garmin_client.get_activity.side_effect = Exception("API Error")

    # Call tool
    result = await app_with_training.call_tool(
        "get_training_effect",
        {"activity_id": 12345678901}
    )

    # Verify error is handled gracefully
    assert result is not None


@pytest.mark.asyncio
async def test_get_training_status_includes_cycling_vo2_max(app_with_training, mock_garmin_client):
    """Test that cycling VO2 max fields are surfaced when present in API response."""
    mock_garmin_client.get_training_status.return_value = MOCK_TRAINING_STATUS

    result = await app_with_training.call_tool(
        "get_training_status",
        {"date": "2024-01-15"},
    )

    assert result is not None
    import json
    text = result[0][0].text if result and result[0] else str(result)
    try:
        data = json.loads(text)
        assert data.get("cycling_vo2_max") == 55.0
        assert data.get("cycling_vo2_max_precise") == 55.12
    except (json.JSONDecodeError, AttributeError):
        # Tool may return raw text; just check the values appear in output
        assert "55.0" in text or "55.12" in text


@pytest.mark.asyncio
async def test_get_training_status_no_cycling_vo2_when_absent(app_with_training, mock_garmin_client):
    """Test that cycling VO2 fields are omitted when the cycling subkey is missing."""
    status_without_cycling = {
        "mostRecentVO2Max": {
            "generic": {"vo2MaxValue": 52.5, "vo2MaxPreciseValue": 52.47},
        },
    }
    mock_garmin_client.get_training_status.return_value = status_without_cycling

    result = await app_with_training.call_tool(
        "get_training_status",
        {"date": "2024-01-15"},
    )

    assert result is not None
    import json
    text = result[0][0].text if result and result[0] else str(result)
    try:
        data = json.loads(text)
        assert "cycling_vo2_max" not in data
        assert "cycling_vo2_max_precise" not in data
    except (json.JSONDecodeError, AttributeError):
        assert "cycling_vo2_max" not in text


# --- get_recent_training -----------------------------------------------------


@pytest.fixture
def recent_training_client(mock_garmin_client):
    """Mock client wired for the activities endpoint used by get_recent_training"""
    mock_garmin_client.garmin_connect_activities = "/activitylist-service/activities/search/activities"
    mock_garmin_client.connectapi = Mock(return_value=MOCK_RECENT_TRAINING_ACTIVITIES)
    mock_garmin_client.get_training_status = Mock(return_value=MOCK_TRAINING_STATUS)
    return mock_garmin_client


@pytest.fixture
def app_with_recent_training(recent_training_client):
    """FastMCP app registered against the recent-training mock client"""
    training.configure(recent_training_client)
    app = FastMCP("Test Recent Training")
    return training.register_tools(app)


@pytest.mark.asyncio
async def test_get_recent_training_defaults(app_with_recent_training, recent_training_client):
    """Default call returns a 7-day window ending today with a summary and sessions"""
    result = await app_with_recent_training.call_tool("get_recent_training", {})

    assert result is not None
    data = json.loads(result[0][0].text)

    assert data["date_range"]["days"] == 7
    start = datetime.date.fromisoformat(data["date_range"]["start"])
    end = datetime.date.fromisoformat(data["date_range"]["end"])
    assert end == datetime.date.today()
    assert (end - start).days == 6

    # The endpoint is called once with the window and page size, not the
    # library's auto-paginating helper.
    recent_training_client.connectapi.assert_called_once()
    _args, kwargs = recent_training_client.connectapi.call_args
    assert kwargs["params"]["startDate"] == start.isoformat()
    assert kwargs["params"]["endDate"] == end.isoformat()
    assert kwargs["params"]["limit"] == "50"
    assert "activityType" not in kwargs["params"]
    recent_training_client.get_activities_by_date.assert_not_called()

    assert data["truncated"] is False
    assert len(data["sessions"]) == 3
    assert data["sessions"][0]["name"] == "Tempo Run"


@pytest.mark.asyncio
async def test_get_recent_training_summary_totals(app_with_recent_training):
    """Summary aggregates duration, distance, calories and load across sessions"""
    result = await app_with_recent_training.call_tool("get_recent_training", {"days": 7})
    data = json.loads(result[0][0].text)
    summary = data["summary"]

    assert summary["sessions"] == 3
    # Two sessions fall on 2024-01-14, so three sessions span two calendar days.
    assert summary["days_trained"] == 2
    assert summary["rest_days"] == 5

    assert summary["total_duration_seconds"] == 2700 + 5400 + 1800
    assert summary["total_duration_hours"] == 2.75
    assert summary["total_distance_meters"] == 55000
    assert summary["total_distance_km"] == 55.0
    assert summary["total_calories"] == 700 + 950 + 320
    assert summary["total_training_load"] == 390.5
    assert summary["avg_aerobic_training_effect"] == 3.5

    # Ordered by total duration, longest first.
    assert list(summary["by_activity_type"].keys()) == ["cycling", "running", "indoor_cycling"]
    assert summary["by_activity_type"]["running"] == {
        "sessions": 1,
        "duration_seconds": 2700,
        "distance_meters": 10000,
        "training_load": 180.4,
    }
    # The Peloton import reports no distance or load, so those stay at zero.
    assert summary["by_activity_type"]["indoor_cycling"]["distance_meters"] == 0
    assert summary["by_activity_type"]["indoor_cycling"]["training_load"] == 0


@pytest.mark.asyncio
async def test_get_recent_training_omits_missing_session_fields(app_with_recent_training):
    """Fields Garmin did not report are omitted rather than returned as null"""
    result = await app_with_recent_training.call_tool("get_recent_training", {})
    sessions = json.loads(result[0][0].text)["sessions"]

    tempo_run = sessions[0]
    assert tempo_run["training_load"] == 180.4
    assert tempo_run["aerobic_training_effect"] == 3.9
    assert tempo_run["anaerobic_training_effect"] == 1.2
    assert tempo_run["training_effect_label"] == "TEMPO"

    peloton = sessions[2]
    assert peloton["type"] == "indoor_cycling"
    assert "training_load" not in peloton
    assert "aerobic_training_effect" not in peloton
    assert "distance_meters" not in peloton
    assert "avg_hr_bpm" not in peloton


@pytest.mark.asyncio
async def test_get_recent_training_includes_training_status(app_with_recent_training, recent_training_client):
    """Training status snapshot is attached for the last day of the window"""
    result = await app_with_recent_training.call_tool("get_recent_training", {})
    data = json.loads(result[0][0].text)

    recent_training_client.get_training_status.assert_called_once_with(
        datetime.date.today().isoformat()
    )
    status = data["training_status"]
    assert status["training_status"] == "PRODUCTIVE"
    assert status["acute_load"] == 250
    assert status["chronic_load"] == 220
    assert status["load_ratio"] == 1.14
    assert status["acwr_status"] == "OPTIMAL"


@pytest.mark.asyncio
async def test_get_recent_training_can_skip_training_status(app_with_recent_training, recent_training_client):
    """include_training_status=False skips the extra API call"""
    result = await app_with_recent_training.call_tool(
        "get_recent_training", {"include_training_status": False}
    )
    data = json.loads(result[0][0].text)

    assert "training_status" not in data
    recent_training_client.get_training_status.assert_not_called()


@pytest.mark.asyncio
async def test_get_recent_training_tolerates_training_status_failure(
    app_with_recent_training, recent_training_client
):
    """A failing training status call does not fail the whole tool"""
    recent_training_client.get_training_status.side_effect = Exception("503 Server Error")

    result = await app_with_recent_training.call_tool("get_recent_training", {})
    data = json.loads(result[0][0].text)

    assert "training_status" not in data
    assert data["summary"]["sessions"] == 3


@pytest.mark.asyncio
async def test_get_recent_training_filters_by_activity_type(app_with_recent_training, recent_training_client):
    """activity_type is passed through to the endpoint and echoed in the result"""
    recent_training_client.connectapi.return_value = [MOCK_RECENT_TRAINING_ACTIVITIES[0]]

    result = await app_with_recent_training.call_tool(
        "get_recent_training", {"days": 14, "activity_type": "running"}
    )
    data = json.loads(result[0][0].text)

    _args, kwargs = recent_training_client.connectapi.call_args
    assert kwargs["params"]["activityType"] == "running"
    assert data["activity_type_filter"] == "running"
    assert data["date_range"]["days"] == 14
    assert data["summary"]["rest_days"] == 13


@pytest.mark.asyncio
async def test_get_recent_training_flags_truncation(app_with_recent_training, recent_training_client):
    """A full page signals there may be more sessions in the window"""
    recent_training_client.connectapi.return_value = MOCK_RECENT_TRAINING_ACTIVITIES[:2]

    result = await app_with_recent_training.call_tool(
        "get_recent_training", {"max_activities": 2}
    )
    data = json.loads(result[0][0].text)

    _args, kwargs = recent_training_client.connectapi.call_args
    assert kwargs["params"]["limit"] == "2"
    assert data["truncated"] is True


@pytest.mark.asyncio
async def test_get_recent_training_clamps_max_activities(app_with_recent_training, recent_training_client):
    """max_activities is clamped to the endpoint's 200 ceiling"""
    await app_with_recent_training.call_tool("get_recent_training", {"max_activities": 5000})

    _args, kwargs = recent_training_client.connectapi.call_args
    assert kwargs["params"]["limit"] == "200"


@pytest.mark.asyncio
async def test_get_recent_training_rejects_out_of_range_days(app_with_recent_training, recent_training_client):
    """Windows outside 1..90 days are rejected before any API call"""
    for days in (0, 91):
        result = await app_with_recent_training.call_tool("get_recent_training", {"days": days})
        text = result[0][0].text
        assert "days" in text.lower() or "maximum" in text.lower()

    recent_training_client.connectapi.assert_not_called()


@pytest.mark.asyncio
async def test_get_recent_training_no_activities(app_with_recent_training, recent_training_client):
    """An empty window returns a zeroed summary rather than an error"""
    recent_training_client.connectapi.return_value = []

    result = await app_with_recent_training.call_tool("get_recent_training", {"days": 3})
    data = json.loads(result[0][0].text)

    assert data["sessions"] == []
    assert data["summary"]["sessions"] == 0
    assert data["summary"]["days_trained"] == 0
    assert data["summary"]["rest_days"] == 3
    assert data["summary"]["total_training_load"] == 0
    assert "avg_aerobic_training_effect" not in data["summary"]


@pytest.mark.asyncio
async def test_get_recent_training_api_error(app_with_recent_training, recent_training_client):
    """API failures surface as an error message, not an exception"""
    recent_training_client.connectapi.side_effect = Exception("401 Unauthorized")

    result = await app_with_recent_training.call_tool("get_recent_training", {})

    assert "Error retrieving recent training" in result[0][0].text
    assert "401 Unauthorized" in result[0][0].text
