import pytest
from unittest.mock import patch, MagicMock
from booking_evaluator import correctness, slot_conflict_evaluator


class TestEvaluators:
    @patch('booking_evaluator.llm')
    def test_correctness_evaluator(self, mock_llm):
        # Mock LLM response for evaluator
        mock_response = MagicMock()
        mock_response.content = '{"correctness": true, "score": 0.9}'
        mock_llm.invoke.return_value = mock_response

        inputs = {"name": "Alice", "age": 28, "reason": "Flu"}
        outputs = {"doctor_slot": "10:00", "confirmation": "Booking confirmed!"}
        reference_outputs = {"doctor_slot": "10:00", "confirmation": "Booking confirmed!"}

        result = correctness(run=None, example=None, inputs=inputs, outputs=outputs, reference_outputs=reference_outputs)

        assert result["key"] == "correctness"
    def test_slot_conflict_evaluator_available_slot(self):
        # Mock run object
        mock_run = MagicMock()
        mock_run.outputs = {"output": {"doctor_slot": "10:00", "confirmation": "Booking confirmed!"}}

        result = slot_conflict_evaluator(run=mock_run, reference_outputs=None)

        assert result["key"] == "slot_conflict_handling"
        assert result["score"] == 1 

    def test_slot_conflict_evaluator_conflict(self):
    
        mock_run = MagicMock()
        mock_run.outputs = {"output": {"doctor_slot": "10:00", "confirmation": "Conflict! Please choose another time."}}

        result = slot_conflict_evaluator(run=mock_run, reference_outputs=None)

        assert result["key"] == "slot_conflict_handling"
        assert result["score"] == 0  

    def test_slot_conflict_evaluator_correct_confirmation(self):
        mock_run = MagicMock()
        mock_run.outputs = {"output": {"doctor_slot": "10:00", "confirmation": "Booking confirmed!"}}

        result = slot_conflict_evaluator(run=mock_run, reference_outputs=None)

        assert result["key"] == "slot_conflict_handling"
        assert result["score"] == 1
