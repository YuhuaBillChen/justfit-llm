use justfit_rust_frontend::download;
use std::path::Path;
fn main()->Result<(),Box<dyn std::error::Error>> {
    let path=std::env::args().nth(1).expect("owned test output");
    let count=download::fetch("https://huggingface.co/incoai/Qwen3.8-27B-Splash/resolve/9d27070b71f7142c6b6025f03ac011d70a73cb48/tokenizer/tokenizer_config.json",Path::new(&path),65536,"792fa3f0cb88b111e54ef3134c873531008c4df471d108da17903426e308aa7b")?;
    assert_eq!(count,1165);println!("verified trusted HTTPS transfer: {count} bytes");Ok(())
}
