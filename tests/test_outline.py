class TestGenerateOutline:
    def test_returns_non_empty_string(self, sample_documents):
        from outline import generate_outline
        result = generate_outline(sample_documents)
        assert isinstance(result, str)
        assert len(result.strip()) > 0
