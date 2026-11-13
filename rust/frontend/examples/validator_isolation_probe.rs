use justfit_rust_frontend::grammar::Factory;
use serde_json::json;
use std::{path::PathBuf,sync::Arc};
#[tokio::main]
async fn main()->Result<(),Box<dyn std::error::Error>>{
    let factory=Arc::new(Factory::new(PathBuf::from(std::env::args().nth(1).unwrap())));
    let a=json!({"type":"object","propertyNames":{"const":"alpha"}});
    let b=json!({"type":"object","propertyNames":{"const":"beta"}});
    let first=factory.create(a.clone(),false).await.map_err(|e|e.1)?;
    let second=factory.create(b.clone(),false).await.map_err(|e|e.1)?;
    let doc=json!({"alpha":1});
    let first_valid=first.lock().unwrap().validator.is_valid(&doc);
    let second_valid=second.lock().unwrap().validator.is_valid(&doc);
    let mask_a=first.lock().unwrap().masks(1,1,&[]).map_err(|e|e.1)?;
    let mask_b=second.lock().unwrap().masks(1,1,&[]).map_err(|e|e.1)?;
    let mut repeats=vec![];
    for(index,schema)in [a.clone(),b.clone(),a,b].into_iter().enumerate(){
        let session=factory.create(schema,false).await.map_err(|e|e.1)?;
        repeats.push(json!({"index":index,"valid":session.lock().unwrap().validator.is_valid(&doc)}));
    }
    println!("{}",json!({"first_valid":first_valid,"second_valid":second_valid,"same_compiled_mask":mask_a==mask_b,"repeats":repeats,"cache":factory.snapshot()}));Ok(())
}
