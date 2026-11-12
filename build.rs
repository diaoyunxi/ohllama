// Build script: compile the Slint UI into generated Rust code.
// The generated module is pulled into `src/gui.rs` via `slint::include_modules!()`.
fn main() {
    slint_build::compile("src/ui/ohllama.slint").unwrap();
}
