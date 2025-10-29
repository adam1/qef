# Minimal Makefile for qef package

.PHONY: test help clean

help:
	@echo "Available targets:"
	@echo "  test   - Run unit tests with pytest"
	@echo "  clean  - Remove Python cache files"
	@echo "  help   - Show this help message"

test:
	pytest tests/ -v

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
