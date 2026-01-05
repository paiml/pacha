# Pacha Makefile
# Certeza Methodology - Tiered Quality Gates
#
# PERFORMANCE TARGETS (Toyota Way: Zero Defects, Fast Feedback)
# - make test-fast: < 30 seconds (unit tests only)
# - make test:      < 2 minutes (all tests)
# - make coverage:  < 5 minutes (coverage report)

SHELL := /bin/bash
.SUFFIXES:
.DELETE_ON_ERROR:
.ONESHELL:

.PHONY: all build test test-fast test-full lint fmt fmt-check clean doc bench coverage coverage-open tier1 tier2 tier3 check book book-build book-serve examples

# Default target
all: tier2

# Build
build:
	cargo build --release

# ============================================================================
# TEST TARGETS
# ============================================================================

# Fast tests (<30s): Unit tests only
test-fast: ## Fast unit tests (<30s target)
	@echo "⚡ Running fast tests (target: <30s)..."
	@if command -v cargo-nextest >/dev/null 2>&1; then \
		time cargo nextest run --lib \
			--status-level skip \
			--failure-output immediate; \
	else \
		echo "💡 Install cargo-nextest for faster tests: cargo install cargo-nextest"; \
		time cargo test --lib; \
	fi
	@echo "✅ Fast tests passed"

# Standard tests (<2min): All tests including integration
test: ## Standard tests (<2min target)
	@echo "🧪 Running standard tests (target: <2min)..."
	@if command -v cargo-nextest >/dev/null 2>&1; then \
		time cargo nextest run \
			--status-level skip \
			--failure-output immediate; \
	else \
		time cargo test; \
	fi
	@echo "✅ Standard tests passed"

# Full comprehensive tests: All features
test-full: ## Comprehensive tests (all features)
	@echo "🔬 Running full comprehensive tests..."
	@if command -v cargo-nextest >/dev/null 2>&1; then \
		time cargo nextest run --all-features; \
	else \
		time cargo test --all-features; \
	fi
	@echo "✅ Full tests passed"

# Linting
lint:
	cargo clippy -- -D warnings

# Format
fmt:
	cargo fmt

fmt-check:
	cargo fmt --check

# Clean
clean:
	cargo clean

# Documentation
doc:
	cargo doc --no-deps --open

# Benchmarks
bench:
	cargo bench

# ============================================================================
# COVERAGE TARGETS (Two-Phase Pattern - cargo-llvm-cov)
# ============================================================================
# CRITICAL: mold linker breaks LLVM coverage instrumentation
# Solution: Temporarily move ~/.cargo/config.toml during coverage runs

coverage: ## Generate HTML coverage report (target: <5 min)
	@echo "📊 Running coverage analysis (target: <5 min)..."
	@echo "🔍 Checking for cargo-llvm-cov and cargo-nextest..."
	@which cargo-llvm-cov > /dev/null 2>&1 || (echo "📦 Installing cargo-llvm-cov..." && cargo install cargo-llvm-cov --locked)
	@which cargo-nextest > /dev/null 2>&1 || (echo "📦 Installing cargo-nextest..." && cargo install cargo-nextest --locked)
	@echo "🧹 Cleaning old coverage data..."
	@mkdir -p target/coverage
	@echo "🧪 Phase 1: Running tests with instrumentation (no report)..."
	@cargo llvm-cov --no-report nextest --no-tests=warn --no-fail-fast --all-features
	@echo "📊 Phase 2: Generating coverage reports..."
	@cargo llvm-cov report --html --output-dir target/coverage/html
	@cargo llvm-cov report --lcov --output-path target/coverage/lcov.info
	@echo ""
	@echo "📊 Coverage Summary:"
	@echo "=================="
	@cargo llvm-cov report --summary-only
	@echo ""
	@echo "💡 Reports:"
	@echo "- HTML: target/coverage/html/index.html"
	@echo "- LCOV: target/coverage/lcov.info"

coverage-open: ## Open HTML coverage report in browser
	@if [ -f target/coverage/html/index.html ]; then \
		xdg-open target/coverage/html/index.html 2>/dev/null || \
		open target/coverage/html/index.html 2>/dev/null || \
		echo "Open: target/coverage/html/index.html"; \
	else \
		echo "❌ Run 'make coverage' first"; \
	fi

# ============================================================================
# TIERED QUALITY GATES
# ============================================================================

# Tier 1: On-save (<1 second)
tier1:
	@echo "Running Tier 1: Fast feedback..."
	@cargo fmt --check
	@cargo clippy -- -W clippy::all
	@cargo check
	@echo "✅ Tier 1: PASSED"

# Tier 2: Pre-commit (<5 seconds)
tier2:
	@echo "Running Tier 2: Pre-commit checks..."
	@cargo test --lib
	@cargo clippy -- -D warnings
	@echo "✅ Tier 2: PASSED"

# Tier 3: Pre-push (full validation)
tier3:
	@echo "Running Tier 3: Full validation..."
	@cargo test --all
	@cargo clippy -- -D warnings
	@echo "✅ Tier 3: PASSED"

# Quick check (compile only)
check:
	cargo check --all

# ============================================================================
# BOOK TARGETS (mdBook)
# ============================================================================

book: book-build ## Build and open the book

book-build: ## Build the book
	@echo "📚 Building Pacha book..."
	@if command -v mdbook >/dev/null 2>&1; then \
		mdbook build book; \
		echo "✅ Book built: book/book/index.html"; \
	else \
		echo "❌ mdbook not found. Install with: cargo install mdbook"; \
		exit 1; \
	fi

book-serve: ## Serve the book locally for development
	@echo "📖 Serving book at http://localhost:3000..."
	@mdbook serve book --open

# ============================================================================
# EXAMPLES
# ============================================================================

examples: ## Run all examples
	@echo "🎯 Running all examples..."
	@for example in examples/*.rs; do \
		name=$$(basename "$$example" .rs); \
		echo "  Running $$name..."; \
		cargo run --example "$$name" --quiet 2>/dev/null && echo "    ✅ $$name passed" || echo "    ❌ $$name failed"; \
	done
	@echo "✅ All examples complete"
