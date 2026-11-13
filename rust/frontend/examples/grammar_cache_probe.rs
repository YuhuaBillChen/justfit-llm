use justfit_rust_frontend::{grammar,tools};
use serde_json::{json,Value};
use sha2::{Digest,Sha256};
use std::{io::{self,BufRead},path::PathBuf,sync::Arc};
#[tokio::main]
async fn main()->Result<(),Box<dyn std::error::Error>>{
    let factory=Arc::new(grammar::Factory::new(PathBuf::from(std::env::args().nth(1).unwrap())));
    for line in io::stdin().lock().lines(){let body:Value=serde_json::from_str(&line?)?;
        let schema=grammar::response_schema(&body);
        let result=match schema{
            Err(error)=>Err(error),
            Ok(schema)=>match tools::normalize(&body){
                Err(error)=>Err(error),
                Ok((_,Some(policy)))=>factory.create_tools(Arc::new(policy),schema,body["thinking"]==true).await,
                Ok((_,None))=>factory.create(schema.unwrap_or(json!({})),body["thinking"]==true).await,
            }
        };
        let value=match result{Ok(session)=>{let mask=session.lock().unwrap().masks(1,1,&[]).map_err(|e|e.1)?;json!({"status":200,"mask_sha256":format!("{:x}",Sha256::digest(mask)),"stats":factory.snapshot()})},Err(error)=>json!({"status":error.0,"error":error.body(),"stats":factory.snapshot()})};println!("{value}");
    }Ok(())
}
