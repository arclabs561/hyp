default:
    @just --list

test:
    if command -v cargo-nextest >/dev/null 2>&1; then cargo nextest run; else cargo test; fi

check:
    cargo fmt --all -- --check
    cargo clippy --all-targets -- -D warnings
    just test

