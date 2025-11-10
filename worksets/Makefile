# Makefile for computing lambda-hat matrix λ̂ and D̂ matrix for Shor code

# Configuration
PYTHON := python3.11
SCRIPTS_DIR := ../../scripts

# Scripts for lambdaHat computation
LAMBDAHAT_PARALLEL_SCRIPT := $(SCRIPTS_DIR)/compute_lambdaHat_parallel.py
LAMBDAHAT_MERGE_SCRIPT := $(SCRIPTS_DIR)/merge_lambdaHat.py

# Scripts for D̂ computation
DHAT_SCRIPT := $(SCRIPTS_DIR)/compute_Dhat.py

# Scripts for rank and signature computation
RANK_SCRIPT := $(SCRIPTS_DIR)/compute_rank.py
SIGNATURE_NUMERICAL_SCRIPT := $(SCRIPTS_DIR)/compute_signature_numerical.py

# Scripts for B̂_{λ,E,F} analysis
ALL_BHAT_ON_M_SCRIPT := $(SCRIPTS_DIR)/compute_all_Bhat_on_M.py
ALL_BHAT_HERMITIAN_SCRIPT := $(SCRIPTS_DIR)/compute_all_Bhat_hermitian_check.py
CHECK_HERMITIAN_SCRIPT := $(SCRIPTS_DIR)/check_hermitian.py
EVAL_DHAT_QUADRATIC_SCRIPT := $(SCRIPTS_DIR)/evaluate_Dhat_quadratic_form.py

# Scripts for isotropic extension
ISOTROPIC_EXTENSION_SCRIPT := $(SCRIPTS_DIR)/compute_isotropic_extension.py

# Output files
OUTPUT_DIR := output
LAMBDAHAT_FINAL_OUTPUT := $(OUTPUT_DIR)/lambdaHat.txt
DHAT_FINAL_OUTPUT := $(OUTPUT_DIR)/Dhat.txt
EXTENSION_VECTOR_OUTPUT := $(OUTPUT_DIR)/extension_vector.txt

# Matrix parameters
LAMBDAHAT_MATRIX_DIM := 28  # Dimension of P_{9,1} basis
NUM_WORKERS := 6            # Number of parallel workers (adjust based on CPU cores)
N_QUBITS := 9               # Number of qubits
MAX_WEIGHT := 1             # Maximum weight t for P_{n,t} basis

# Run identification (timestamp and PID for this make invocation)
RUN_ID := $(shell date '+%s')_$$$$

# Generate list of partial output files for lambdaHat
LAMBDAHAT_PARTIAL_FILES := $(foreach i,$(shell seq 0 $$(($(NUM_WORKERS) - 1))),$(OUTPUT_DIR)/lambdaHat_partial_$(i).txt)


.PHONY: all clean lambdaHat Dhat clean-lambdaHat clean-Dhat rank-lambdaHat rank-Dhat signature-lambdaHat signature-Dhat check-Dhat-hermitian eval-Dhat-quadratic Bhat-on-M Bhat-hermitian isotropic-extension help

help:
	@echo "Makefile for computing quantum error correction matrices"
	@echo ""
	@echo "Targets:"
	@echo "  lambdaHat            - Compute the lambda-hat matrix λ̂ (multiprocessing)"
	@echo "  Dhat                 - Compute the sum matrix D̂ = Σ_{E,F} B̂_{λ,E,F}"
	@echo "  rank-lambdaHat       - Compute the rank of λ̂ (symbolic)"
	@echo "  rank-Dhat            - Compute the rank of D̂ (symbolic)"
	@echo "  signature-lambdaHat  - Compute the signature of λ̂ (numerical, via eigenvalues)"
	@echo "  signature-Dhat       - Compute the signature of D̂ (numerical, via eigenvalues)"
	@echo "  check-Dhat-hermitian - Check if D̂ is Hermitian (symbolic)"
	@echo "  eval-Dhat-quadratic  - Evaluate quadratic form Q(v) = ⟨v|D̂|v⟩ on |0_L⟩ and |1_L⟩"
	@echo "  Bhat-on-M            - Verify B̂_{λ,E,F} = 0 on code space M for all pairs in P_{9,1}"
	@echo "  Bhat-hermitian       - Check Hermiticity of B̂_{λ,E,F} for all pairs in P_{9,1}"
	@echo "  isotropic-extension  - Compute isotropic extension M' = M ⊕ span(u+v) using Grove's algorithm"
	@echo "  clean-lambdaHat      - Remove lambdaHat output files and logs"
	@echo "  clean-Dhat           - Remove D̂ output files and logs"
	@echo "  clean                - Remove all output files and logs"
	@echo "  help                 - Show this help message"
	@echo ""
	@echo "Configuration:"
	@echo "  λ̂ matrix dimension: $(LAMBDAHAT_MATRIX_DIM) (P_{$(N_QUBITS),$(MAX_WEIGHT)} basis)"
	@echo "  Number of workers: $(NUM_WORKERS)"
	@echo "  Output directory: $(OUTPUT_DIR)"
	@echo ""
	@echo "Usage:"
	@echo "  make lambdaHat          - Compute λ̂ with $(NUM_WORKERS) workers"
	@echo "  make Dhat               - Compute D̂"
	@echo ""
	@echo "Output files:"
	@echo "  λ̂: $(LAMBDAHAT_FINAL_OUTPUT)"
	@echo "  D̂: $(DHAT_FINAL_OUTPUT)"
	@echo ""
	@echo "Logs:"
	@echo "  λ̂ computation: $(OUTPUT_DIR)/lambdaHat_<timestamp>_<pid>.log"
	@echo "  D̂ computation: $(OUTPUT_DIR)/Dhat_<timestamp>_<pid>.log"

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

# Compute all partial lambdaHat files using multiprocessing script
# Note: The script handles parallelization internally, so we don't use make -j here
$(LAMBDAHAT_PARTIAL_FILES): $(LAMBDAHAT_PARALLEL_SCRIPT) | $(OUTPUT_DIR)
	@echo "========================================"
	@echo "Computing λ̂ matrix with $(NUM_WORKERS) workers..."
	@echo "========================================"
	@LOG_FILE=$(OUTPUT_DIR)/lambdaHat_$(RUN_ID).log; \
	echo "Log file: $$LOG_FILE"; \
	$(PYTHON) $(LAMBDAHAT_PARALLEL_SCRIPT) -o $(OUTPUT_DIR) -w $(NUM_WORKERS) -n $(N_QUBITS) -t $(MAX_WEIGHT) 2>&1 | tee $$LOG_FILE


#==============================================================================
# D̂ (sum of B̂_{λ,E,F}) computation targets
#==============================================================================

# Main target: compute D̂ = Σ_{E,F} B̂_{λ,E,F}
Dhat: $(DHAT_FINAL_OUTPUT)

$(DHAT_FINAL_OUTPUT): $(DHAT_SCRIPT) | $(OUTPUT_DIR)
	@echo "========================================"
	@echo "Computing D̂ = Σ_{E,F} B̂_{λ,E,F}..."
	@echo "========================================"
	@LOG_FILE=$(OUTPUT_DIR)/Dhat_$(RUN_ID).log; \
	echo "Log file: $$LOG_FILE"; \
	$(PYTHON) $(DHAT_SCRIPT) -o $(DHAT_FINAL_OUTPUT) -n $(N_QUBITS) -t $(MAX_WEIGHT) 2>&1 | tee $$LOG_FILE

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

# Check if D̂ is Hermitian (symbolic)
check-Dhat-hermitian: $(DHAT_FINAL_OUTPUT) $(CHECK_HERMITIAN_SCRIPT)
	@echo "========================================"
	@echo "Checking if D̂ is Hermitian (symbolic)..."
	@echo "========================================"
	$(PYTHON) $(CHECK_HERMITIAN_SCRIPT) $(DHAT_FINAL_OUTPUT)

# Evaluate quadratic form Q(v) = ⟨v|D̂|v⟩ on code space basis
eval-Dhat-quadratic: $(DHAT_FINAL_OUTPUT) $(EVAL_DHAT_QUADRATIC_SCRIPT)
	@echo "========================================"
	@echo "Evaluating quadratic form Q(v) = ⟨v|D̂|v⟩ on |0_L⟩ and |1_L⟩..."
	@echo "========================================"
	$(PYTHON) $(EVAL_DHAT_QUADRATIC_SCRIPT) -i $(DHAT_FINAL_OUTPUT)

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
# Isotropic extension targets
#==============================================================================

# Compute isotropic extension M' = M ⊕ span(u+v)
isotropic-extension: $(EXTENSION_VECTOR_OUTPUT)

$(EXTENSION_VECTOR_OUTPUT): $(DHAT_FINAL_OUTPUT) $(ISOTROPIC_EXTENSION_SCRIPT) | $(OUTPUT_DIR)
	@echo "========================================"
	@echo "Computing isotropic extension using Grove's algorithm..."
	@echo "========================================"
	@LOG_FILE=$(OUTPUT_DIR)/isotropic_extension_$(RUN_ID).log; \
	echo "Log file: $$LOG_FILE"; \
	$(PYTHON) $(ISOTROPIC_EXTENSION_SCRIPT) -i $(DHAT_FINAL_OUTPUT) -o $(EXTENSION_VECTOR_OUTPUT) -n $(N_QUBITS) -t $(MAX_WEIGHT) 2>&1 | tee $$LOG_FILE

#==============================================================================
# Clean targets
#==============================================================================

# Clean lambdaHat files
clean-lambdaHat:
	rm -f $(OUTPUT_DIR)/lambdaHat_partial_*.txt
	rm -f $(OUTPUT_DIR)/lambdaHat_*.log
	rm -f $(OUTPUT_DIR)/signature_lambdaHat_*.log
	rm -f $(LAMBDAHAT_FINAL_OUTPUT)

# Clean D̂ files
clean-Dhat:
	rm -f $(OUTPUT_DIR)/Dhat_*.log
	rm -f $(OUTPUT_DIR)/signature_dhat_*.log
	rm -f $(DHAT_FINAL_OUTPUT)

# Clean all generated files
clean: clean-lambdaHat clean-Dhat
	rmdir $(OUTPUT_DIR) 2>/dev/null || true
