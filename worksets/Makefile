# Makefile for computing lambda-hat matrix λ̂ and D̂ matrix for Shor code

# Configuration
PYTHON := python3.11
SCRIPTS_DIR := ../../scripts

# Scripts for lambdaHat computation (make-based parallelization)
LAMBDAHAT_PARTIAL_SCRIPT := $(SCRIPTS_DIR)/compute_lambdaHat_partial.py
LAMBDAHAT_MERGE_SCRIPT := $(SCRIPTS_DIR)/merge_lambdaHat.py

# Scripts for D̂ computation (multiprocessing-based parallelization)
DHAT_PARALLEL_SCRIPT := $(SCRIPTS_DIR)/compute_Dhat_parallel.py
DHAT_MERGE_SCRIPT := $(SCRIPTS_DIR)/merge_Dhat.py

# Scripts for rank and signature computation
RANK_SCRIPT := $(SCRIPTS_DIR)/compute_rank.py
SIGNATURE_NUMERICAL_SCRIPT := $(SCRIPTS_DIR)/compute_signature_numerical.py

# Scripts for B̂_{λ,E,F} analysis
ALL_BHAT_ON_M_SCRIPT := $(SCRIPTS_DIR)/compute_all_Bhat_on_M.py
ALL_BHAT_HERMITIAN_SCRIPT := $(SCRIPTS_DIR)/compute_all_Bhat_hermitian_check.py
JHAT_SCRIPT := $(SCRIPTS_DIR)/compute_Jhat.py
CHECK_HERMITIAN_SCRIPT := $(SCRIPTS_DIR)/check_hermitian.py

# Output files
OUTPUT_DIR := output
LAMBDAHAT_FINAL_OUTPUT := $(OUTPUT_DIR)/lambdaHat.txt
DHAT_FINAL_OUTPUT := $(OUTPUT_DIR)/Dhat.txt
JHAT_FINAL_OUTPUT := $(OUTPUT_DIR)/Jhat.txt

# Matrix parameters
LAMBDAHAT_MATRIX_DIM := 28  # Dimension of P_{9,1} basis
DHAT_MATRIX_DIM := 512      # Dimension of computational basis for 9 qubits
NUM_WORKERS := 6            # Number of parallel workers (adjust based on CPU cores)
N_QUBITS := 9               # Number of qubits
MAX_WEIGHT := 1             # Maximum weight t for P_{n,t} basis

# Run identification (timestamp and PID for this make invocation)
RUN_ID := $(shell date '+%s')_$$$$

# Compute column ranges for each worker (for lambdaHat)
# We divide 28 columns (0-27) into NUM_WORKERS chunks
# Worker 0: columns 0-6   (7 columns)
# Worker 1: columns 7-13  (7 columns)
# Worker 2: columns 14-20 (7 columns)
# Worker 3: columns 21-27 (7 columns)

LAMBDAHAT_COLS_PER_WORKER := $(shell echo $$(($(LAMBDAHAT_MATRIX_DIM) / $(NUM_WORKERS))))

# Generate list of partial output files for lambdaHat
LAMBDAHAT_PARTIAL_FILES := $(foreach i,$(shell seq 0 $$(($(NUM_WORKERS) - 1))),$(OUTPUT_DIR)/lambdaHat_partial_$(i).txt)

# Generate list of partial output files for D̂
DHAT_PARTIAL_FILES := $(foreach i,$(shell seq 0 $$(($(NUM_WORKERS) - 1))),$(OUTPUT_DIR)/dhat_partial_$(i).txt)

.PHONY: all clean lambdaHat Dhat Jhat clean-lambdaHat clean-Dhat clean-Jhat rank-lambdaHat rank-Dhat rank-Jhat signature-lambdaHat signature-Dhat signature-Jhat check-Jhat-hermitian Bhat-on-M Bhat-hermitian help

help:
	@echo "Makefile for computing quantum error correction matrices"
	@echo ""
	@echo "Targets:"
	@echo "  lambdaHat            - Compute the lambda-hat matrix λ̂ (make-based parallelization)"
	@echo "  Dhat                 - Compute the total error form matrix D̂ (multiprocessing)"
	@echo "  Jhat                 - Compute the sum matrix Ĵ = Σ_{E,F} B̂_{λ,E,F}"
	@echo "  rank-lambdaHat       - Compute the rank of λ̂ (symbolic)"
	@echo "  rank-Dhat            - Compute the rank of D̂ (symbolic)"
	@echo "  rank-Jhat            - Compute the rank of Ĵ (symbolic)"
	@echo "  signature-lambdaHat  - Compute the signature of λ̂ (numerical, via eigenvalues)"
	@echo "  signature-Dhat       - Compute the signature of D̂ (numerical, via eigenvalues)"
	@echo "  signature-Jhat       - Compute the signature of Ĵ (numerical, via eigenvalues)"
	@echo "  check-Jhat-hermitian - Check if Ĵ is Hermitian (symbolic)"
	@echo "  Bhat-on-M            - Verify B̂_{λ,E,F} = 0 on code space M for all pairs in P_{9,1}"
	@echo "  Bhat-hermitian       - Check Hermiticity of B̂_{λ,E,F} for all pairs in P_{9,1}"
	@echo "  clean-lambdaHat      - Remove lambdaHat output files and logs"
	@echo "  clean-Dhat           - Remove D̂ output files and logs"
	@echo "  clean-Jhat           - Remove Ĵ output files and logs"
	@echo "  clean                - Remove all output files and logs"
	@echo "  help                 - Show this help message"
	@echo ""
	@echo "Configuration:"
	@echo "  λ̂ matrix dimension: $(LAMBDAHAT_MATRIX_DIM) (P_{$(N_QUBITS),$(MAX_WEIGHT)} basis)"
	@echo "  D̂ matrix dimension: $(DHAT_MATRIX_DIM)×$(DHAT_MATRIX_DIM) (computational basis)"
	@echo "  Number of workers: $(NUM_WORKERS)"
	@echo "  Output directory: $(OUTPUT_DIR)"
	@echo ""
	@echo "Usage:"
	@echo "  make lambdaHat          - Sequential λ̂ computation"
	@echo "  make -j4 lambdaHat      - Parallel λ̂ with 4 workers"
	@echo "  make Dhat               - Compute D̂ with $(NUM_WORKERS) workers"
	@echo ""
	@echo "Output files:"
	@echo "  λ̂: $(LAMBDAHAT_FINAL_OUTPUT)"
	@echo "  D̂: $(DHAT_FINAL_OUTPUT)"
	@echo "  Ĵ: $(JHAT_FINAL_OUTPUT)"
	@echo ""
	@echo "Logs:"
	@echo "  λ̂ workers: $(OUTPUT_DIR)/lambdaHat_worker_<timestamp>_<pid>_<worker_id>.log"
	@echo "  D̂ computation: $(OUTPUT_DIR)/dhat_<timestamp>_<pid>.log"

all: lambdaHat

# Create output directory
$(OUTPUT_DIR):
	mkdir -p $(OUTPUT_DIR)

#==============================================================================
# λ̂ (lambda-hat) computation targets
#==============================================================================

# Main target: compute lambdaHat
lambdaHat: $(LAMBDAHAT_FINAL_OUTPUT)

# Merge partial files into final output
$(LAMBDAHAT_FINAL_OUTPUT): $(LAMBDAHAT_PARTIAL_FILES) $(LAMBDAHAT_MERGE_SCRIPT)
	@echo "========================================"
	@echo "Merging partial λ̂ results..."
	@echo "========================================"
	$(PYTHON) $(LAMBDAHAT_MERGE_SCRIPT) $(LAMBDAHAT_PARTIAL_FILES) -o $(LAMBDAHAT_FINAL_OUTPUT) --dimension $(LAMBDAHAT_MATRIX_DIM)

# Pattern rule for computing partial lambdaHat files
# This uses shell arithmetic to compute column ranges
$(OUTPUT_DIR)/lambdaHat_partial_%.txt: $(LAMBDAHAT_PARTIAL_SCRIPT) | $(OUTPUT_DIR)
	@WORKER_ID=$*; \
	COL_START=$$(($$WORKER_ID * $(LAMBDAHAT_COLS_PER_WORKER))); \
	if [ $$WORKER_ID -eq $$(($(NUM_WORKERS) - 1)) ]; then \
		COL_END=$$(($(LAMBDAHAT_MATRIX_DIM) - 1)); \
	else \
		COL_END=$$(($$COL_START + $(LAMBDAHAT_COLS_PER_WORKER) - 1)); \
	fi; \
	LOG_FILE=$(OUTPUT_DIR)/lambdaHat_worker_$(RUN_ID)_$$WORKER_ID.log; \
	echo "Worker $$WORKER_ID: Computing λ̂ columns $$COL_START-$$COL_END (log: $$LOG_FILE)"; \
	$(PYTHON) $(LAMBDAHAT_PARTIAL_SCRIPT) $$COL_START $$COL_END -o $@ > $$LOG_FILE 2>&1

#==============================================================================
# D̂ (error form) computation targets
#==============================================================================

# Main target: compute D̂ using multiprocessing
Dhat: $(DHAT_FINAL_OUTPUT)

$(DHAT_FINAL_OUTPUT): $(DHAT_PARTIAL_FILES) $(DHAT_MERGE_SCRIPT)
	@echo "========================================"
	@echo "Merging partial D̂ results..."
	@echo "========================================"
	$(PYTHON) $(DHAT_MERGE_SCRIPT) $(DHAT_PARTIAL_FILES) -o $(DHAT_FINAL_OUTPUT) \
		--dimension $(DHAT_MATRIX_DIM) -n $(N_QUBITS) -t $(MAX_WEIGHT)

# Compute all partial D̂ files using multiprocessing script
# Note: The script handles parallelization internally, so we don't use make -j here
$(DHAT_PARTIAL_FILES): $(DHAT_PARALLEL_SCRIPT) | $(OUTPUT_DIR)
	@echo "========================================"
	@echo "Computing D̂ matrix with $(NUM_WORKERS) workers..."
	@echo "========================================"
	@LOG_FILE=$(OUTPUT_DIR)/dhat_$(RUN_ID).log; \
	echo "Log file: $$LOG_FILE"; \
	$(PYTHON) $(DHAT_PARALLEL_SCRIPT) -o $(OUTPUT_DIR) -w $(NUM_WORKERS) -n $(N_QUBITS) -t $(MAX_WEIGHT) 2>&1 | tee $$LOG_FILE

#==============================================================================
# Ĵ (sum of B̂_{λ,E,F}) computation targets
#==============================================================================

# Main target: compute Ĵ = Σ_{E,F} B̂_{λ,E,F}
Jhat: $(JHAT_FINAL_OUTPUT)

$(JHAT_FINAL_OUTPUT): $(JHAT_SCRIPT) | $(OUTPUT_DIR)
	@echo "========================================"
	@echo "Computing Ĵ = Σ_{E,F} B̂_{λ,E,F}..."
	@echo "========================================"
	@LOG_FILE=$(OUTPUT_DIR)/Jhat_$(RUN_ID).log; \
	echo "Log file: $$LOG_FILE"; \
	$(PYTHON) $(JHAT_SCRIPT) -o $(JHAT_FINAL_OUTPUT) -n $(N_QUBITS) -t $(MAX_WEIGHT) 2>&1 | tee $$LOG_FILE

#==============================================================================
# Rank computation targets
#==============================================================================

# Compute rank of λ̂
rank-lambdaHat: $(LAMBDAHAT_FINAL_OUTPUT)
	@echo "========================================"
	@echo "Computing rank of λ̂..."
	@echo "========================================"
	$(PYTHON) $(RANK_SCRIPT) $(LAMBDAHAT_FINAL_OUTPUT)

# Compute rank of D̂
rank-Dhat: $(DHAT_FINAL_OUTPUT)
	@echo "========================================"
	@echo "Computing rank of D̂..."
	@echo "========================================"
	$(PYTHON) $(RANK_SCRIPT) $(DHAT_FINAL_OUTPUT)

# Compute rank of Ĵ
rank-Jhat: $(JHAT_FINAL_OUTPUT)
	@echo "========================================"
	@echo "Computing rank of Ĵ..."
	@echo "========================================"
	$(PYTHON) $(RANK_SCRIPT) $(JHAT_FINAL_OUTPUT)

# Compute signature of λ̂ (numerical)
signature-lambdaHat: $(LAMBDAHAT_FINAL_OUTPUT)
	@echo "========================================"
	@echo "Computing signature of λ̂ (numerical)..."
	@echo "========================================"
	@LOG_FILE=$(OUTPUT_DIR)/signature_lambdaHat_$(RUN_ID).log; \
	echo "Log file: $$LOG_FILE"; \
	$(PYTHON) $(SIGNATURE_NUMERICAL_SCRIPT) $(LAMBDAHAT_FINAL_OUTPUT) 2>&1 | tee $$LOG_FILE

# Compute signature of D̂ (numerical)
signature-Dhat: $(DHAT_FINAL_OUTPUT)
	@echo "========================================"
	@echo "Computing signature of D̂ (numerical)..."
	@echo "========================================"
	@LOG_FILE=$(OUTPUT_DIR)/signature_dhat_$(RUN_ID).log; \
	echo "Log file: $$LOG_FILE"; \
	$(PYTHON) $(SIGNATURE_NUMERICAL_SCRIPT) $(DHAT_FINAL_OUTPUT) 2>&1 | tee $$LOG_FILE

# Compute signature of Ĵ (numerical)
signature-Jhat: $(JHAT_FINAL_OUTPUT)
	@echo "========================================"
	@echo "Computing signature of Ĵ (numerical)..."
	@echo "========================================"
	@LOG_FILE=$(OUTPUT_DIR)/signature_jhat_$(RUN_ID).log; \
	echo "Log file: $$LOG_FILE"; \
	$(PYTHON) $(SIGNATURE_NUMERICAL_SCRIPT) $(JHAT_FINAL_OUTPUT) 2>&1 | tee $$LOG_FILE

# Check if Ĵ is Hermitian (symbolic)
check-Jhat-hermitian: $(JHAT_FINAL_OUTPUT) $(CHECK_HERMITIAN_SCRIPT)
	@echo "========================================"
	@echo "Checking if Ĵ is Hermitian (symbolic)..."
	@echo "========================================"
	$(PYTHON) $(CHECK_HERMITIAN_SCRIPT) $(JHAT_FINAL_OUTPUT)

#==============================================================================
# B̂_{λ,E,F} analysis targets
#==============================================================================

# Verify B̂_{λ,E,F} = 0 on code space M for all pairs in P_{9,1}
Bhat-on-M: $(ALL_BHAT_ON_M_SCRIPT) | $(OUTPUT_DIR)
	@echo "========================================"
	@echo "Verifying B̂_{λ,E,F} = 0 on code space M for all pairs in P_{$(N_QUBITS),$(MAX_WEIGHT)}..."
	@echo "========================================"
	@LOG_FILE=$(OUTPUT_DIR)/Bhat_on_M_$(RUN_ID).log; \
	echo "Log file: $$LOG_FILE"; \
	$(PYTHON) $(ALL_BHAT_ON_M_SCRIPT) -o $(OUTPUT_DIR) -n $(N_QUBITS) -t $(MAX_WEIGHT) 2>&1 | tee $$LOG_FILE

# Check Hermiticity of B̂_{λ,E,F} for all pairs in P_{9,1}
Bhat-hermitian: $(ALL_BHAT_HERMITIAN_SCRIPT) | $(OUTPUT_DIR)
	@echo "========================================"
	@echo "Checking Hermiticity of B̂_{λ,E,F} for all pairs in P_{$(N_QUBITS),$(MAX_WEIGHT)}..."
	@echo "========================================"
	@LOG_FILE=$(OUTPUT_DIR)/Bhat_hermitian_$(RUN_ID).log; \
	echo "Log file: $$LOG_FILE"; \
	$(PYTHON) $(ALL_BHAT_HERMITIAN_SCRIPT) -o $(OUTPUT_DIR) -n $(N_QUBITS) -t $(MAX_WEIGHT) 2>&1 | tee $$LOG_FILE

#==============================================================================
# Clean targets
#==============================================================================

# Clean lambdaHat files
clean-lambdaHat:
	rm -f $(OUTPUT_DIR)/lambdaHat_partial_*.txt
	rm -f $(OUTPUT_DIR)/lambdaHat_worker_*.log
	rm -f $(OUTPUT_DIR)/signature_lambdaHat_*.log
	rm -f $(LAMBDAHAT_FINAL_OUTPUT)

# Clean D̂ files
clean-Dhat:
	rm -f $(OUTPUT_DIR)/dhat_partial_*.txt
	rm -f $(OUTPUT_DIR)/dhat_*.log
	rm -f $(OUTPUT_DIR)/signature_dhat_*.log
	rm -f $(DHAT_FINAL_OUTPUT)

# Clean Ĵ files
clean-Jhat:
	rm -f $(OUTPUT_DIR)/Jhat_*.log
	rm -f $(OUTPUT_DIR)/signature_jhat_*.log
	rm -f $(JHAT_FINAL_OUTPUT)

# Clean all generated files
clean: clean-lambdaHat clean-Dhat clean-Jhat
	rmdir $(OUTPUT_DIR) 2>/dev/null || true
