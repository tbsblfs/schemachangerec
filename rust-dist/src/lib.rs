use std::collections::HashMap;
use std::iter::zip;
use std::ptr;
use std::time::SystemTime;

use ndarray::Array1;
use ndarray_linalg::Norm;
use numpy::{PyArray2, PyReadonlyArray1};
use pyo3::prelude::*;
use rayon::prelude::*;

use chrono::prelude::{DateTime, Utc};
use minstore::MinStore;

mod minstore;

type ResType = [f32; 4];

const NAN_RESULT: ResType = [f32::NAN, f32::NAN, f32::NAN, f32::NAN];

#[pyfunction]
fn get_mrr(
    labels: PyReadonlyArray1<i32>,
    predictions: PyReadonlyArray1<f64>,
    groups: PyReadonlyArray1<i64>,
) -> f64 {
    let labels = labels.as_array();
    let predictions = predictions.as_array();
    let groups = groups.as_array();
    let mut sum = 0.0;
    let mut start : usize = 0;
    let mut count = 0;
    for group in groups {
        if let Some(best_prediction) = (start..start + *group as usize)
            .filter_map(|i| if labels[i] >= 1 {
                Some(predictions[i])
            } else {
                None
            })
            .max_by(|a, b| a.partial_cmp(b).unwrap()) {
            let larger = (start..start + *group as usize)
                .filter(|i| labels[*i] == 0 && predictions[*i] >= best_prediction)
                .count();
            sum += 1.0 / (larger + 1) as f64;
            count += 1;
        }
        // let mut best_prediction : f64 = 0.0;
        // for i  in start..start + *group as usize {
        //     if labels[i] >= 1 {
        //         best_prediction = f64::max(best_prediction, predictions[i]);
        //     }
        // }
        //let larger = (start..start + *group as usize)
        //    .filter(|i| labels[*i] == 0 && predictions[*i] >= best_prediction)
        //    .count();
        // let mut larger = 0;
        // for i in start..start + *group as usize {
        //     if labels[i] == 0 && predictions[i] >= best_prediction {
        //         larger += 1;
        //     }
        // }
        //
        //
        // let group_labels = labels.slice(s![start..start + group]);
        // let group_predictions = predictions.slice(s![start..start + group]);
        // start += group;
        // let mut best_prediction : f64 = 0.0;
        // for (label, &prediction) in group_labels.iter().zip(group_predictions.iter()) {
        //     if *label > 0 {
        //         best_prediction = f64::max(best_prediction, prediction);
        //     }
        // }
        // let mut larger = 0;
        // for (label, prediction) in group_labels.iter().zip(group_predictions.iter()) {
        //     if *label == 0 && *prediction >= best_prediction {
        //         larger += 1;
        //     }
        // }
        //sum += 1.0 / (larger + 1) as f64;
        start += *group as usize;
    }
    sum / count as f64
}

#[pyfunction]
fn get_minimum_timediff(
    valid_from: Vec<f64>,
    revisions: Vec<PyReadonlyArray1<f64>>,
) -> Vec<f64> {
    valid_from
        .into_iter()
        .zip(revisions)
        .map(|(valid_from, revisions)| {
            revisions.as_array()
                .into_iter()
                .map(|x| (*x - valid_from).abs())
                .min_by(|&a, &b| a.partial_cmp(&b).unwrap())
                .unwrap_or(f64::NAN)
        })
        .collect::<Vec<_>>()
}

#[pyfunction]
fn get_distances_chunks(
    page_titles: Vec<&str>,
    source_pages: Vec<Vec<&str>>,
    page_dict: HashMap<&str, PyReadonlyArray1<f32>>,
    chunk_size: usize,
) -> PyResult<Py<PyArray2<f32>>> {
    if page_titles.len() != source_pages.len() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "page_titles and source_pages must have the same length",
        ));
    }

    let now: DateTime<Utc> = SystemTime::now().into();
    println!("Current time {:?}", now.to_rfc3339());
    let page_dict = build_dict(page_dict);

    let tasks = zip(page_titles.into_iter(), source_pages.into_iter()).collect::<Vec<_>>();

    let start_time = std::time::Instant::now();
    let res: Vec<_> = tasks
        .par_chunks(chunk_size)
        .flat_map_iter(|chunk| {
            chunk.iter().map(|(page_target, page_source)| {
                get_distance(page_target, page_source, &page_dict)
            })
        })
        .collect::<Vec<_>>();
    println!("time: {:?}", start_time.elapsed());

    let dims = (res.len(), 4);
    // turn res into a numpy 2d-array
    let res = Python::with_gil(|py| unsafe {
        let array = PyArray2::<f32>::new(py, dims, false);
        let mut data_ptr = array.data();
        for v in res {
            ptr::copy_nonoverlapping(v.as_ptr(), data_ptr, 4);
            data_ptr = data_ptr.add(4);
        }

        array.to_owned()
    });
    let now: DateTime<Utc> = SystemTime::now().into();
    println!("Current time {:?}", now.to_rfc3339());
    Ok(res)
}

fn build_dict<'a>(
    page_dict: HashMap<&'a str, PyReadonlyArray1<f32>>,
) -> HashMap<&'a str, Array1<f32>> {
    println!("Build page dict");
    let start_dict = std::time::Instant::now();
    let page_dict: HashMap<&str, Array1<f32>> = page_dict
        .into_iter()
        .map(|(k, v)| (k, v.as_array().to_owned()))
        .collect();
    println!("time: {:?}", start_dict.elapsed());
    page_dict
}

fn get_distance(
    page_target: &str,
    page_source: &[&str],
    page_dict: &HashMap<&str, Array1<f32>>,
) -> ResType {
    page_dict
        .get(page_target)
        .map(|page_target_vec| {
            let mut dists_euclidean = MinStore::new();
            let mut dists_cosine = MinStore::new();

            let target_length = page_target_vec.norm_l2();
            page_source
                .iter()
                .filter_map(|&x| page_dict.get(x))
                .for_each(|page_source_vec| {
                    let euclidean = (page_target_vec - page_source_vec).norm_l2();
                    if !euclidean.is_nan() {
                        dists_euclidean.push(euclidean.try_into().unwrap());
                    }

                    let cosine = 1.0
                        - page_target_vec.dot(page_source_vec)
                            / (target_length * page_source_vec.norm_l2());
                    if !cosine.is_nan() {
                        dists_cosine.push(cosine.try_into().unwrap());
                    }
                });

            [
                dists_euclidean.min(),
                dists_euclidean.mean(),
                dists_cosine.min(),
                dists_cosine.mean(),
            ]
        })
        .unwrap_or(NAN_RESULT)
}

#[pymodule]
fn rust_dist(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(get_distances_chunks, m)?)?;
    m.add_function(wrap_pyfunction!(get_mrr, m)?)?;
    m.add_function(wrap_pyfunction!(get_minimum_timediff, m)?)?;
    Ok(())
}
