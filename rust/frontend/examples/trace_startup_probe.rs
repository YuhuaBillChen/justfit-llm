use justfit_rust_frontend::engine::Engine;
fn main(){let command:Vec<String>=std::env::args().skip(1).collect();assert!(Engine::spawn(&command,1).is_err());}
