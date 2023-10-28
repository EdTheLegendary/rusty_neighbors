use std::error::Error;
use csv;
use serde::Deserialize;
use rand::Rng;
use rand::distributions::Alphanumeric;
use std::collections::HashMap;

#[derive(Debug, Clone, Deserialize)]
struct Flower {
    sepal_length: f64,
    sepal_width: f64,
    petal_length: f64,
    petal_width: f64,
    class: String,
}

#[derive(Debug, Clone, PartialEq, PartialOrd)]
struct MeasuredFlower {
    distance: f64,
    class: String,
}

impl Flower {
    fn rowify(&self) -> Vec<f64> {
        vec!(self.sepal_length, self.sepal_width, self.petal_length, self.petal_width)
    }
    fn rand_flower() -> Flower {
        let mut rng = rand::thread_rng();
        let s: String = rand::thread_rng()
        .sample_iter(&Alphanumeric)
        .take(7)
        .map(char::from)
        .collect();
        Flower {
            sepal_length: rng.gen_range(0.1..7.0),
            sepal_width: rng.gen_range(0.1..7.0),
            petal_length: rng.gen_range(0.1..7.0),
            petal_width: rng.gen_range(0.1..7.0),
            class: String::from(s),
        }
    }
}

fn predict_classification<'a>(train: &'a Vec<Flower>, test_row: &'a Flower, num_neighbors: usize) -> String {
    let neighbors = get_neighbors(train, test_row, num_neighbors);
    let mut m: HashMap<String, usize> = HashMap::new();
    for x in neighbors {
        *m.entry(x).or_default() += 1;
    }
    let max = m.into_iter().max_by_key(|(_, v)| *v).map(|(k, _)| k);

    let max = max.unwrap();
    // println!("Max thing{:?}", max.clone());
    max
}

fn get_neighbors<'a>(train: &'a Vec<Flower>, test_row: &'a Flower, num_neighbors: usize) -> Vec<String> {
    let mut distances = Vec::new();

    for train_row in train {
        let dist = euclidean_distance(test_row, train_row);
        distances.push(MeasuredFlower {distance: dist, class: train_row.class.clone()});
    }

    distances.sort_by(|a, b| b.distance.partial_cmp(&a.distance).unwrap());
    distances.reverse();
    // println!("Distances and classes {:?}", distances.clone());

    let mut neighbors = Vec::new();
    for i in 0..num_neighbors {
        let item = &distances[i];
        neighbors.push(item.class.clone());
    }
    neighbors
}

fn euclidean_distance(row1: &Flower, row2: &Flower) -> f64 {
    let mut distance = 0.0;

    let measure1 = row1.rowify();
    let measure2 = row2.rowify();
    let mut measurements1 = measure1.iter();
    let mut measurements2 = measure2.iter();

    while let Some(measurement) = measurements1.next() {
        distance += (measurement - measurements2.next().unwrap()).powi(2);
    }

    distance.sqrt()
}

fn read_from_file(path: &str) -> Result<Vec<Flower>, Box<dyn Error>> {
    let mut reader = csv::Reader::from_path(path)?;

    let headers = reader.headers()?;
    println!("{:?}", headers);

    let mut last = Vec::new();

    for result in reader.deserialize() {
        let record: Flower = result?;
        last.push(record);
        //println!("{:?}", record);
    }
    Ok(last)
}

fn normalize_dataset() -> TODO {

}

fn accuracy_metric -> TODO {

}

fn cross_validation_split() -> TODO {

}

fn evaluate_algorithm(dataset, algorithm, n_folds, num_neighbors) -> Vec<> {
    
}

fn k_nearest_neighbors<'a>(train: &'a Vec<Flower>, test: &'a Vec<Flower>, num_neighbors: usize) -> Vec<String> {
    let mut predictions = Vec::new();
    for row in test {
        let output = predict_classification(train, row, num_neighbors);
        predictions.push(output);
    }
    predictions

}

fn main() {

    let dataset = read_from_file("../iris.csv").unwrap();

    let n_folds = 5;
    let num_neighbors = 5
    
    let scores = evaluate_algorithm(dataset, k_nearest_neighbors, n_folds, num_neighbors);
    println!("Scores: {}" scores);
    // println!("Mean Accuracy: {}", sum(scores)/len(scores);

    //let mut dataset = Vec::new();
    //for _x in 0..10 {
    //    dataset.push(Flower::rand_flower());
    //}
    //for x in dataset.clone().iter() {
    //    println!("{:?}\n", x);
    //}

    //let neighbors = get_neighbors(&dataset, &dataset[0], 3);
    //let prediction = predict_classification(&dataset, &dataset[0], 3);
    // println!("Expected {}, Got {}.", &dataset[0].class, prediction);
    
    // println!("The item to compare is: {:?}", &dataset[0]);

    //for neighbor in neighbors {
    //    println!("{:?}\n", neighbor);
    //}

    //if let Err(e) = read_from_file("../iris.csv") {
    //    eprintln!("{}", e);
    //}
}
