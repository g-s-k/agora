# agora: signal processing orchestrator

This library is a minimal implementation of an "audio graph" - in a general
sense, this means a framework for sequencing and executing signal processing
operations.

### More information

Documentation and tests are included alongside the code. To run the tests or
view a friendlier form of documentation, you need to have the Rust toolchain
installed. See [rustup.rs](https://rustup.rs) for instructions and guidance on
how to obtain that.

Once that is taken care of, you can clone this repository and run the following
commands:

- `cargo test` will run the tests and documentation examples
- `cargo doc --open` will build the HTML documentation and open it in your
  default browser.
- `cargo run --example tone` will run an example application that produces a
  pair of test tones.
