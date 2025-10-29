# Makefile for computing lambda-hat matrix λ̂ for Shor code

# Configuration
PYTHON := python3.11
SCRIPTS_DIR := ../../scripts
PARTIAL_SCRIPT := $(SCRIPTS_DIR)/compute_lambdaHat_partial.py
MERGE_SCRIPT := $(SCRIPTS_DIR)/merge_lambdaHat.py

# Output files
OUTPUT_DIR := output
FINAL_OUTPUT := lambdaHat.txt

# Matrix parameters
MATRIX_DIM := 28  # Dimension of P_{9,1}
NUM_WORKERS := 4   # Number of parallel workers (adjust based on CPU cores)

# Compute column ranges for each worker
# We divide 28 columns (0-27) into NUM_WORKERS chunks
# Worker 0: columns 0-6   (7 columns)
# Worker 1: columns 7-13  (7 columns)
# Worker 2: columns 14-20 (7 columns)
# Worker 3: columns 21-27 (7 columns)

COLS_PER_WORKER := $(shell echo $$(($(MATRIX_DIM) / $(NUM_WORKERS))))

# Generate list of partial output files
PARTIAL_FILES := $(foreach i,$(shell seq 0 $$(($(NUM_WORKERS) - 1))),$(OUTPUT_DIR)/partial_$(i).txt)

.PHONY: all clean lambdaHat help

help:
	@echo "Makefile for computing lambda-hat matrix λ̂"
	@echo ""
	@echo "Targets:"
	@echo "  lambdaHat    - Compute the lambda-hat matrix λ̂ (parallelized)"
	@echo "  clean        - Remove output files"
	@echo "  help         - Show this help message"
	@echo ""
	@echo "Configuration:"
	@echo "  Matrix dimension: $(MATRIX_DIM)"
	@echo "  Number of workers: $(NUM_WORKERS)"
	@echo "  Output directory: $(OUTPUT_DIR)"
	@echo "  Final output: $(FINAL_OUTPUT)"

all: lambdaHat

# Main target: compute lambdaHat
lambdaHat: $(FINAL_OUTPUT)

# Create output directory
$(OUTPUT_DIR):
	mkdir -p $(OUTPUT_DIR)

# Merge partial files into final output
$(FINAL_OUTPUT): $(PARTIAL_FILES) $(MERGE_SCRIPT)
	@echo "========================================"
	@echo "Merging partial results..."
	@echo "========================================"
	$(PYTHON) $(MERGE_SCRIPT) $(PARTIAL_FILES) -o $(FINAL_OUTPUT) --dimension $(MATRIX_DIM)

# Pattern rule for computing partial files
# This uses shell arithmetic to compute column ranges
$(OUTPUT_DIR)/partial_%.txt: $(PARTIAL_SCRIPT) | $(OUTPUT_DIR)
	@WORKER_ID=$*; \
	COL_START=$$(($$WORKER_ID * $(COLS_PER_WORKER))); \
	if [ $$WORKER_ID -eq $$(($(NUM_WORKERS) - 1)) ]; then \
		COL_END=$$(($(MATRIX_DIM) - 1)); \
	else \
		COL_END=$$(($$COL_START + $(COLS_PER_WORKER) - 1)); \
	fi; \
	echo "========================================"; \
	echo "Worker $$WORKER_ID: Computing columns $$COL_START-$$COL_END"; \
	echo "========================================"; \
	$(PYTHON) $(PARTIAL_SCRIPT) $$COL_START $$COL_END -o $@

# Clean generated files
clean:
	rm -rf $(OUTPUT_DIR)
	rm -f $(FINAL_OUTPUT)
