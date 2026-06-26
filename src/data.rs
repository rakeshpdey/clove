use ndarray::Array2;
use std::fs::{self, File};
use std::io::Read;
use tokenizers::Tokenizer as HfTokenizer;
use std::thread;
use std::sync::mpsc::{sync_channel, Receiver};
use std::sync::Arc;
use rand::RngExt;
use rayon::prelude::*;

// ========================================================================
// 1. DATA AUGMENTATION PIPELINES (VISION & NLP)
// ========================================================================
pub trait Transform: Send + Sync {
    fn apply(&self, data: &mut [f32]);
}

/// Chained pipeline for applying multiple augmentations in parallel
#[derive(Clone)]
pub struct DataPipeline {
    transforms: Vec<Arc<dyn Transform>>,
}

impl Default for DataPipeline {
    fn default() -> Self {
        Self::new()
    }
}

impl DataPipeline {
    pub fn new() -> Self {
        Self { transforms: Vec::new() }
    }

    pub fn add_transform<T: Transform + 'static>(mut self, transform: T) -> Self {
        self.transforms.push(Arc::new(transform));
        self
    }

    pub fn process_batch(&self, batch: &mut Array2<f32>) {
        // Parallelize augmentations across the CPU cores
        let cols = batch.ncols();
        batch.as_slice_mut().unwrap().par_chunks_mut(cols).for_each(|row| {
            for transform in &self.transforms {
                transform.apply(row);
            }
        });
    }
}

// --- Specific Transforms ---

pub struct Normalize {
    pub mean: f32,
    pub std: f32,
}

impl Transform for Normalize {
    fn apply(&self, data: &mut [f32]) {
        for val in data.iter_mut() {
            *val = (*val - self.mean) / self.std;
        }
    }
}

pub struct RandomNoise {
    pub factor: f32,
}

impl Transform for RandomNoise {
    fn apply(&self, data: &mut [f32]) {
        let mut rng = rand::rng();
        for val in data.iter_mut() {
            let noise: f32 = rng.random_range(-self.factor..self.factor);
            *val += noise;
        }
    }
}

pub struct RandomTokenMasking {
    pub mask_token_id: f32,
    pub probability: f32,
}

impl Transform for RandomTokenMasking {
    fn apply(&self, data: &mut [f32]) {
        let mut rng = rand::rng();
        for val in data.iter_mut() {
            let roll: f32 = rng.random();
            if roll < self.probability {
                *val = self.mask_token_id;
            }
        }
    }
}

// ========================================================================
// 2. DATA LOADING INTERFACES
// ========================================================================
pub fn load_mnist_images(path: &str) -> Vec<Array2<f32>> {
    let mut file = File::open(path).expect("Failed to find MNIST image file. Is it in the /data folder?");
    
    let mut magic = [0u8; 4]; file.read_exact(&mut magic).unwrap();
    let mut num_imgs = [0u8; 4]; file.read_exact(&mut num_imgs).unwrap();
    let mut rows = [0u8; 4]; file.read_exact(&mut rows).unwrap();
    let mut cols = [0u8; 4]; file.read_exact(&mut cols).unwrap();

    let count = u32::from_be_bytes(num_imgs) as usize;
    let mut images = Vec::with_capacity(count);

    println!("[DATA] Unpacking {} images...", count);

    for _ in 0..count {
        let mut buffer = [0u8; 784];
        file.read_exact(&mut buffer).unwrap();
        
        let float_data: Vec<f32> = buffer.iter().map(|&x| x as f32 / 255.0).collect();
        images.push(Array2::from_shape_vec((1, 784), float_data).unwrap());
    }
    images
}

pub fn load_mnist_labels(path: &str) -> Vec<f32> {
    let mut file = File::open(path).expect("Failed to find MNIST label file.");
    
    let mut magic = [0u8; 4]; file.read_exact(&mut magic).unwrap();
    let mut num_labels = [0u8; 4]; file.read_exact(&mut num_labels).unwrap();

    let count = u32::from_be_bytes(num_labels) as usize;
    let mut labels = Vec::with_capacity(count);

    for _ in 0..count {
        let mut buffer = [0u8; 1];
        file.read_exact(&mut buffer).unwrap();
        labels.push(buffer[0] as f32);
    }
    labels
}

pub struct Tokenizer {
    inner: HfTokenizer,
    pub vocab_size: usize,
}

impl Tokenizer {
    pub fn from_pretrained(identifier: &str) -> Self {
        let inner = HfTokenizer::from_pretrained(identifier, None)
            .unwrap_or_else(|_| panic!("Failed to download/load tokenizer: {}", identifier));
        let vocab_size = inner.get_vocab_size(true);
        Self { inner, vocab_size }
    }

    pub fn from_file(path: &str) -> Self {
        let inner = HfTokenizer::from_file(path)
            .unwrap_or_else(|_| panic!("Failed to load tokenizer from file: {}", path));
        let vocab_size = inner.get_vocab_size(true);
        Self { inner, vocab_size }
    }

    pub fn encode(&self, text: &str) -> Vec<usize> {
        let encoding = self.inner.encode(text, false).expect("Encoding failed");
        encoding.get_ids().iter().map(|&id| id as usize).collect()
    }

    pub fn decode(&self, ids: &[usize]) -> String {
        let u32_ids: Vec<u32> = ids.iter().map(|&id| id as u32).collect();
        self.inner.decode(&u32_ids, false).unwrap_or_else(|_| String::new())
    }
}

pub struct DataLoader {
    pub tokenizer: Tokenizer,
    pub seq_len: usize,
    pub batch_size: usize,
    // PRE-FETCHING RING BUFFER: Holds batches generated asynchronously by the worker thread!
    receiver: Receiver<(Array2<f32>, Array2<f32>)>, 
}

impl DataLoader {
    pub fn from_file(file_path: &str, seq_len: usize, batch_size: usize, pipeline: Option<DataPipeline>) -> Self {
        println!("Reading dataset into RAM...");
        let text = fs::read_to_string(file_path).unwrap_or_else(|_| {
            println!("WARNING: input.txt not found. Using fallback text.");
            String::from("To be, or not to be, that is the question:")
        });
        
        println!("Loading Hugging Face Tokenizer...");
        let tokenizer = Tokenizer::from_pretrained("gpt2");
        
        println!("Tokenizing dataset...");
        let encoded_data = tokenizer.encode(&text);
        println!("Dataset loaded! Total tokens: {}", encoded_data.len());
        
        // 1. Setup the Bounded Queue (Limits RAM usage to 10 cached batches)
        let (sender, receiver) = sync_channel(10);
        let slen = seq_len;
        let bsize = batch_size;

        // 2. Spawn the Asynchronous Worker Thread
        thread::spawn(move || {
            let mut current_idx = 0;
            loop {
                let mut x_batch = Vec::with_capacity(bsize * slen);
                let mut y_batch = Vec::with_capacity(bsize * slen);

                for _ in 0..bsize {
                    if current_idx + slen + 1 >= encoded_data.len() {
                        current_idx = 0; 
                    }

                    let x_seq = &encoded_data[current_idx .. current_idx + slen];
                    let y_seq = &encoded_data[current_idx + 1 .. current_idx + slen + 1];

                    x_batch.extend(x_seq.iter().map(|&id| id as f32));
                    y_batch.extend(y_seq.iter().map(|&id| id as f32));

                    current_idx += slen;
                }

                let mut x_array = Array2::from_shape_vec((bsize, slen), x_batch).unwrap();
                let y_array = Array2::from_shape_vec((bsize, slen), y_batch).unwrap();

                // Apply Augmentations in the background thread BEFORE sending to GPU!
                if let Some(pipe) = &pipeline {
                    pipe.process_batch(&mut x_array);
                }

                // 3. Push batch to the queue. If queue hits 10, the thread automatically sleeps!
                if sender.send((x_array, y_array)).is_err() {
                    break; // Exit if the DataLoader was dropped in the main thread
                }
            }
        });

        Self {
            tokenizer,
            seq_len,
            batch_size,
            receiver,
        }
    }

    pub fn next_batch(&mut self) -> (Array2<f32>, Array2<f32>) {
        // INSTANT RETRIEVAL!
        // We no longer read from the array on the main thread.
        // We just pop the pre-computed, augmented batch instantly from RAM!
        self.receiver.recv().expect("Failed to fetch pre-loaded batch from worker thread!")
    }
}