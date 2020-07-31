//use std::io::prelude::*;
use std::fs::File;
use std::io::{Write, Error};
use std::fs::OpenOptions;

#[allow(dead_code)]
pub fn write_to_file_append(data: &Vec<f64>, file_path: String, append: bool ) -> Result<(), Error> {
    let strings: Vec<String> = data.iter().map(|n| n.to_string()).collect();

    //let mut file = File:: create(file_path)?;
    let mut file = OpenOptions::new().create(true).append(append).open(file_path).expect("error");
    writeln!(file, "{}", strings.join(", "))?;
    //writeln!(file, "{}", "\n")?;
    Ok(())
}

#[allow(dead_code)]
pub fn write_to_file(data: &Vec<f64>, file_path: String ) {
    let strings: Vec<String> = data.iter().map(|n| n.to_string()).collect();

    let mut file = File::create(file_path).expect("File could not be created");
    writeln!(file, "{}", strings.join(", ")).expect("data could not be written");
}

