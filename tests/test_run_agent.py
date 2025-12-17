import pytest
from unittest.mock import patch, MagicMock
from run_agent import collect_patient_info, select_doctor, confirm_booking, graph, BookingState


class TestAgentNodes:
    @patch('run_agent.llm')
    def test_collect_patient_info(self, mock_llm):
        # Mock LLM response
        mock_response = MagicMock()
        mock_response.content = "Patient is a 28-year-old with flu symptoms."
        mock_llm.invoke.return_value = mock_response

        state: BookingState = {
            "name": "Alice",
            "age": 28,
            "reason": "Flu",
            "summary": "",
            "doctor_slot": "",
            "confirmation": "",
            "requested_slot": "10:00"
        }

        result = collect_patient_info(state)

        assert result["summary"] == "Patient is a 28-year-old with flu symptoms."
        mock_llm.invoke.assert_called_once()

    @patch('run_agent.llm')
    def test_select_doctor_available_slot(self, mock_llm):
        mock_response = MagicMock()
        mock_response.content = '{"doctor_slot": "10:00"}'
        mock_llm.invoke.return_value = mock_response

        state: BookingState = {
            "name": "Alice",
            "age": 28,
            "reason": "Flu",
            "summary": "Young adult with flu",
            "doctor_slot": "",
            "confirmation": "",
            "requested_slot": "10:00"
        }

        result = select_doctor(state)

        assert result["doctor_slot"] == "10:00"  

    @patch('run_agent.llm')
    def test_select_doctor_conflict(self, mock_llm):
        mock_response = MagicMock()
        mock_response.content = '{"doctor_slot": "10:00"}'
        mock_llm.invoke.return_value = mock_response

        state: BookingState = {
            "name": "Alice",
            "age": 28,
            "reason": "Flu",
            "summary": "Young adult with flu",
            "doctor_slot": "",
            "confirmation": "",
            "requested_slot": "15:00"  
        }

        result = select_doctor(state)

        assert result["doctor_slot"] == "10:00"  

    def test_confirm_booking_success(self):
        state: BookingState = {
            "name": "Alice",
            "age": 28,
            "reason": "Flu",
            "summary": "Young adult with flu",
            "doctor_slot": "10:00",
            "confirmation": "",
            "requested_slot": "10:00"
        }

        result = confirm_booking(state)

        assert result["confirmation"] == "Booking confirmed!"

    def test_confirm_booking_conflict(self):
        state: BookingState = {
            "name": "Alice",
            "age": 28,
            "reason": "Flu",
            "summary": "Young adult with flu",
            "doctor_slot": "11:00",
            "confirmation": "",
            "requested_slot": "10:00"  
        }

        result = confirm_booking(state)

        assert result["confirmation"] == "Conflict! Please choose another time."

    def test_confirm_booking_no_slot(self):
        state: BookingState = {
            "name": "Alice",
            "age": 28,
            "reason": "Flu",
            "summary": "Young adult with flu",
            "doctor_slot": "",
            "confirmation": "",
            "requested_slot": "10:00"
        }

        result = confirm_booking(state)

        assert result["confirmation"] == "No available slots. Please call support."


class TestGraph:
    @patch('run_agent.llm')
    def test_full_graph_success(self, mock_llm):
        # Mock LLM responses
        mock_responses = [
            MagicMock(content="Patient summary"),  
            MagicMock(content='{"doctor_slot": "10:00"}'),  
        ]
        mock_llm.invoke.side_effect = mock_responses

        initial_state: BookingState = {
            "name": "Alice",
            "age": 28,
            "reason": "Flu",
            "summary": "",
            "doctor_slot": "",
            "confirmation": "",
            "requested_slot": "10:00"
        }

        result = graph.invoke(initial_state)

        assert result["doctor_slot"] == "10:00"
        assert result["confirmation"] == "Booking confirmed!"
        assert result["summary"] == "Patient summary"

    @patch('run_agent.llm')
    def test_full_graph_conflict(self, mock_llm):
        # Mock LLM responses
        mock_responses = [
            MagicMock(content="Patient summary"),  
            MagicMock(content='{"doctor_slot": "11:00"}'), 
        ]
        mock_llm.invoke.side_effect = mock_responses

        initial_state: BookingState = {
            "name": "Alice",
            "age": 28,
            "reason": "Flu",
            "summary": "",
            "doctor_slot": "",
            "confirmation": "",
            "requested_slot": "12:00" 
        }

        result = graph.invoke(initial_state)

        assert result["doctor_slot"] == "11:00"
        assert result["confirmation"] == "Conflict! Please choose another time."
