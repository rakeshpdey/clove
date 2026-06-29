# Clove Framework Makefile

.PHONY: all build test lint format clean doc

all: format lint test

build:
	cargo build --release

test:
	cargo test

lint:
	cargo clippy --all-targets --all-features -- -D warnings

format:
	cargo fmt

doc:
	cargo doc --no-deps --open

clean:
	cargo clean

