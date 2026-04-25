class TestGenerateExercises:
    def test_returns_non_empty_string(self, sample_documents):
        from exercises import generate_exercises
        result = generate_exercises(sample_documents)
        assert isinstance(result, str)
        assert len(result.strip()) > 0
